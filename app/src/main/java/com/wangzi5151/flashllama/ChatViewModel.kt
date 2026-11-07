package com.wangzi5151.flashllama

import android.app.Application
import android.content.ContentResolver
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.File
import java.util.Locale

data class ChatMessage(
    val role: String,
    val content: String,
    val streaming: Boolean = false,
)

data class GenParams(
    val nThreads: Int,
    val nCtx: Int = 4096,
    val nBatch: Int = 256,
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val maxTokens: Int = 512,
    val kvType: Int = 0, // 0 = f16, 2 = q8_0, 3 = q4_0
    val gpuLayers: Int = 0, // 0 = CPU only, 99 = offload everything (experimental)
    val useMlock: Boolean = false,
    val useChatTemplate: Boolean = true,
)

data class ModelInfo(
    val name: String,
    val desc: String,
    val nParams: Long,
    val sizeBytes: Long,
    val nCtxTrain: Int,
    val nVocab: Int,
    val nCtx: Int,
    val llamaVersion: String,
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val TAG = "FlashLlama"
    private val contentResolver: ContentResolver = app.contentResolver
    private val nativeLibDir: String = app.applicationInfo.nativeLibraryDir

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _params = MutableStateFlow(GenParams(nThreads = defaultThreadCount()))
    val params: StateFlow<GenParams> = _params.asStateFlow()

    private val _modelInfo = MutableStateFlow<ModelInfo?>(null)
    val modelInfo: StateFlow<ModelInfo?> = _modelInfo.asStateFlow()

    private val _status = MutableStateFlow("请选择 GGUF 模型文件")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    private val _generating = MutableStateFlow(false)
    val generating: StateFlow<Boolean> = _generating.asStateFlow()

    private val _speed = MutableStateFlow("")
    val speed: StateFlow<String> = _speed.asStateFlow()

    /**
     * Guards [handle], [modelPfd] and the in-flight native jobs. Native work is
     * tracked so a release can wait for it before calling `nativeFree`; doing
     * otherwise frees the llama context while a decode is still running (UAF).
     */
    private val sessionLock = Any()
    private val releaseScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var releaseJob: Job? = null
    private var generationJob: Job? = null
    private var resetJob: Job? = null
    private var sessionEpoch = 0L

    private var handle: Long = 0L
    private var modelPfd: ParcelFileDescriptor? = null
    private var pendingPrompt: String? = null

    private data class SessionTicket(
        val handle: Long,
        val generation: Job?,
        val reset: Job?,
    )

    /** Queues a prompt to be sent automatically once the current model is ready. */
    fun queuePrompt(prompt: String) {
        pendingPrompt = prompt
    }

    val isReady: Boolean get() = synchronized(sessionLock) { handle != 0L }

    fun updateParams(transform: (GenParams) -> GenParams) {
        _params.update(transform)
    }

    fun loadModel(uri: Uri) {
        if (_loading.value || _generating.value) return
        _loading.value = true
        _speed.value = ""
        _status.value = "正在加载模型…"
        val epoch = synchronized(sessionLock) { sessionEpoch }
        viewModelScope.launch(Dispatchers.Default) {
            var newPfd: ParcelFileDescriptor? = null
            var committed = false
            try {
                val name = queryDisplayName(uri)
                val resolved = resolveModelPath(uri)
                newPfd = resolved.second

                releaseCurrentBlocking()

                val p = _params.value
                LlamaBridge.nativeSetLogPath(logFilePath())
                diag("loadModel resolved=${resolved.first} pfd=${newPfd != null}")

                var h = 0L
                if (newPfd != null) {
                    h = LlamaBridge.nativeInitFd(
                        newPfd.fd, nativeLibDir, p.nThreads, p.nCtx, p.nBatch,
                        nGpuLayers = p.gpuLayers, mLock = p.useMlock, kvType = p.kvType,
                    )
                    if (h == 0L) {
                        // Some providers do not support loading straight from the fd.
                        diag("fd load failed, copying to cache and retrying")
                        newPfd.close()
                        newPfd = null
                        val copied = copyToCache(uri)
                        if (copied != null) {
                            h = LlamaBridge.nativeInit(
                                copied.absolutePath, nativeLibDir, p.nThreads, p.nCtx, p.nBatch,
                                nGpuLayers = p.gpuLayers, mLock = p.useMlock, kvType = p.kvType,
                            )
                        }
                    }
                } else {
                    h = LlamaBridge.nativeInit(
                        resolved.first, nativeLibDir, p.nThreads, p.nCtx, p.nBatch,
                        nGpuLayers = p.gpuLayers, mLock = p.useMlock, kvType = p.kvType,
                    )
                }
                diag("native init returned handle=$h")
                if (h == 0L) {
                    _status.value = "模型加载失败，请确认是有效的 GGUF 文件"
                } else {
                    // Read metadata before publishing the handle so unload()
                    // can never free it while nativeInfo is still running.
                    val info = parseModelInfo(name, LlamaBridge.nativeInfo(h))
                    val devices = LlamaBridge.nativeDevices()
                    val accepted = synchronized(sessionLock) {
                        if (sessionEpoch != epoch) {
                            false
                        } else {
                            handle = h
                            modelPfd = newPfd
                            true
                        }
                    }
                    if (accepted) {
                        committed = true
                        diag("devices=$devices")
                        _modelInfo.value = info
                        _status.value = "模型已就绪"
                        _messages.value = emptyList()
                    } else {
                        diag("load abandoned: session was unloaded")
                        LlamaBridge.nativeFree(h)
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "load failed", t)
                diag("EXCEPTION ${t.javaClass.name}: ${t.message}")
                t.stackTrace.take(6).forEach { diag("  at $it") }
                _status.value = "加载出错：${t.message}"
            } finally {
                if (!committed) {
                    runCatching { newPfd?.close() }
                    newPfd = null
                }
                _loading.value = false
            }
        }
    }

    /** Loads a model from an absolute filesystem path (used by automation / "open with"). */
    fun loadModelFromPath(path: String) {
        if (_loading.value || _generating.value) return
        _loading.value = true
        _speed.value = ""
        _status.value = "正在加载模型…"
        Log.i(TAG, "loadModelFromPath: $path")
        diag("loadModelFromPath path=$path nativeLibDir=$nativeLibDir")
        val epoch = synchronized(sessionLock) { sessionEpoch }
        viewModelScope.launch(Dispatchers.Default) {
            try {
                releaseCurrentBlocking()
                val p = _params.value
                LlamaBridge.nativeSetLogPath(logFilePath())
                val h = LlamaBridge.nativeInit(
                    path, nativeLibDir, p.nThreads, p.nCtx, p.nBatch,
                    nGpuLayers = p.gpuLayers, mLock = p.useMlock, kvType = p.kvType,
                )
                diag("nativeInit returned handle=$h")
                if (h == 0L) {
                    _status.value = "模型加载失败，请确认是有效的 GGUF 文件"
                } else {
                    val info = parseModelInfo(File(path).name, LlamaBridge.nativeInfo(h))
                    val devices = LlamaBridge.nativeDevices()
                    val accepted = synchronized(sessionLock) {
                        if (sessionEpoch != epoch) {
                            false
                        } else {
                            handle = h
                            true
                        }
                    }
                    if (accepted) {
                        diag("devices=$devices")
                        _modelInfo.value = info
                        _status.value = "模型已就绪"
                        _messages.value = emptyList()
                        pendingPrompt?.let { prompt -> pendingPrompt = null; send(prompt) }
                    } else {
                        diag("load abandoned: session was unloaded")
                        LlamaBridge.nativeFree(h)
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "load failed", t)
                diag("EXCEPTION ${t.javaClass.name}: ${t.message}")
                t.stackTrace.take(6).forEach { diag("  at $it") }
                _status.value = "加载出错：${t.message}"
            } finally {
                _loading.value = false
            }
        }
    }

    private fun diag(msg: String) {
        try {
            val dir = getApplication<Application>().getExternalFilesDir(null) ?: return
            File(dir, "diag.txt").appendText(msg + "\n")
        } catch (_: Throwable) {
        }
    }

    private fun logFilePath(): String {
        val dir = getApplication<Application>().getExternalFilesDir(null)
        return File(dir ?: getApplication<Application>().cacheDir, "llama.log").absolutePath
    }

    private fun copyToCache(uri: Uri): File? = try {
        val dst = File(getApplication<Application>().cacheDir, "model.gguf")
        contentResolver.openInputStream(uri)?.use { input ->
            dst.outputStream().use { output -> input.copyTo(output) }
        }
        diag("copied model to ${dst.absolutePath} (${dst.length()} bytes)")
        dst
    } catch (t: Throwable) {
        diag("copy failed: ${t.message}")
        null
    }

    fun send(input: String) {
        val text = input.trim()
        if (text.isEmpty() || _generating.value) return
        val h = synchronized(sessionLock) { handle }
        if (h == 0L) return

        val history = _messages.value + ChatMessage("user", text)
        val assistantIndex = history.size

        // LAZY so the native call cannot start until the handle is registered
        // as the in-flight generation under sessionLock.
        val job = viewModelScope.launch(Dispatchers.Default, start = CoroutineStart.LAZY) {
            val started = System.currentTimeMillis()
            val buffer = StringBuilder()
            var lastPublish = 0L
            var tokenCount = 0
            try {
                val prompt = buildPrompt(history)
                val p = _params.value
                val callback = TokenCallback { piece ->
                    tokenCount++
                    buffer.append(piece)
                    val now = System.currentTimeMillis()
                    if (now - lastPublish >= 50) {
                        lastPublish = now
                        val snapshot = buffer.toString()
                        publishAssistant(assistantIndex, snapshot, streaming = true)
                    }
                }
                LlamaBridge.nativeGenerate(
                    h, prompt, p.maxTokens, p.temperature, p.topK, p.topP,
                    p.minP, p.repeatPenalty, 256, -1, callback,
                )
            } catch (t: Throwable) {
                buffer.append("\n[错误] ${t.message}")
            } finally {
                val elapsed = (System.currentTimeMillis() - started).coerceAtLeast(1)
                publishAssistant(assistantIndex, buffer.toString(), streaming = false)
                val tps = tokenCount * 1000f / elapsed
                _speed.value = "%.1f tok/s · %d tokens".format(tps, tokenCount)
                diag("gen tokens=$tokenCount elapsedMs=$elapsed tps=%.2f".format(tps))
                _generating.value = false
            }
        }

        synchronized(sessionLock) {
            if (handle != h) {
                job.cancel()
                return
            }
            generationJob = job
            _messages.value = history + ChatMessage("assistant", "", streaming = true)
            _generating.value = true
            _speed.value = ""
            job.start()
        }
    }

    fun stop() {
        synchronized(sessionLock) {
            val h = handle
            if (h != 0L) LlamaBridge.nativeCancel(h)
        }
    }

    fun clearChat() {
        _messages.value = emptyList()
        _speed.value = ""
        if (_generating.value) return
        val h = synchronized(sessionLock) { handle }
        if (h == 0L) return
        val job = viewModelScope.launch(Dispatchers.Default, start = CoroutineStart.LAZY) {
            LlamaBridge.nativeReset(h)
        }
        synchronized(sessionLock) {
            if (handle != h) {
                job.cancel()
                return
            }
            resetJob = job
            job.start()
        }
    }

    fun unload() {
        val ticket = synchronized(sessionLock) {
            sessionEpoch++
            takeSessionLocked()
        }
        _modelInfo.value = null
        _status.value = "请选择 GGUF 模型文件"
        _speed.value = ""
        scheduleRelease(ticket)
    }

    private fun takeSessionLocked(): SessionTicket {
        val ticket = SessionTicket(handle, generationJob, resetJob)
        handle = 0L
        generationJob = null
        resetJob = null
        return ticket
    }

    private fun takeSession(): SessionTicket = synchronized(sessionLock) { takeSessionLocked() }

    private fun closeModelPfd() {
        runCatching { modelPfd?.close() }
        modelPfd = null
    }

    /** Releases the current session on a background coroutine after it goes idle. */
    private fun scheduleRelease(ticket: SessionTicket) {
        closeModelPfd()
        if (ticket.handle == 0L) return
        LlamaBridge.nativeCancel(ticket.handle)
        synchronized(sessionLock) {
            val previous = releaseJob
            releaseJob = releaseScope.launch {
                previous?.join()
                ticket.generation?.join()
                ticket.reset?.join()
                LlamaBridge.nativeFree(ticket.handle)
            }
        }
    }

    /** Blocking variant used by model loading (already off the main thread). */
    private fun releaseCurrentBlocking() {
        val ticket = takeSession()
        closeModelPfd()
        if (ticket.handle == 0L) return
        LlamaBridge.nativeCancel(ticket.handle)
        runBlocking {
            synchronized(sessionLock) { releaseJob }?.join()
            ticket.generation?.join()
            ticket.reset?.join()
            LlamaBridge.nativeFree(ticket.handle)
        }
    }

    private fun publishAssistant(index: Int, content: String, streaming: Boolean) {
        _messages.update { list ->
            if (index !in list.indices) return@update list
            list.toMutableList().also {
                it[index] = it[index].copy(content = content, streaming = streaming)
            }
        }
    }

    private fun buildPrompt(history: List<ChatMessage>): String {
        val p = _params.value
        val h = synchronized(sessionLock) { handle }
        val applier: ((Array<String>, Array<String>, Boolean) -> String)? =
            if (p.useChatTemplate && h != 0L) {
                { roles, contents, add -> LlamaBridge.nativeApplyChatTemplate(h, roles, contents, add) }
            } else {
                null
            }
        return composePrompt(history, p.useChatTemplate, applier)
    }

    private fun resolveModelPath(uri: Uri): Pair<String, ParcelFileDescriptor?> {
        if (uri.scheme == "file") {
            return (uri.path ?: "") to null
        }
        val pfd = contentResolver.openFileDescriptor(uri, "r")
        if (pfd != null && pfd.statSize > 0) {
            return "/proc/self/fd/${pfd.fd}" to pfd
        }
        // Non-seekable provider: copy into the cache so the runtime can mmap it.
        val dst = File(getApplication<Application>().cacheDir, "model.gguf")
        contentResolver.openInputStream(uri)?.use { input ->
            dst.outputStream().use { output -> input.copyTo(output) }
        }
        pfd?.close()
        return dst.absolutePath to null
    }

    private fun queryDisplayName(uri: Uri): String {
        if (uri.scheme == "file") return File(uri.path ?: "").name
        return try {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst() && !c.isNull(0)) c.getString(0) else "model.gguf"
            } ?: "model.gguf"
        } catch (t: Throwable) {
            "model.gguf"
        }
    }

    override fun onCleared() {
        val ticket = synchronized(sessionLock) {
            sessionEpoch++
            takeSessionLocked()
        }
        closeModelPfd()
        val pendingRelease = synchronized(sessionLock) { releaseJob }
        if (ticket.handle != 0L) {
            // Cancel first, then wait for the native call to return. nativeFree
            // also waits for any remaining lease before deleting the session.
            LlamaBridge.nativeCancel(ticket.handle)
            runBlocking {
                pendingRelease?.join()
                ticket.generation?.join()
                ticket.reset?.join()
                LlamaBridge.nativeFree(ticket.handle)
            }
        } else {
            runBlocking { pendingRelease?.join() }
        }
        releaseScope.cancel()
        super.onCleared()
    }
}

/**
 * Pure prompt assembly used by [ChatViewModel.send]. [applyTemplate] is null when
 * the model has no chat template (raw completion mode).
 */
internal fun composePrompt(
    history: List<ChatMessage>,
    useChatTemplate: Boolean,
    applyTemplate: ((Array<String>, Array<String>, Boolean) -> String)?,
): String {
    if (useChatTemplate && applyTemplate != null) {
        val roles = ArrayList<String>(history.size)
        val contents = ArrayList<String>(history.size)
        for (m in history) {
            if (m.content.isBlank()) continue
            roles.add(m.role)
            contents.add(m.content)
        }
        if (roles.isNotEmpty()) {
            val formatted = applyTemplate(roles.toTypedArray(), contents.toTypedArray(), true)
            if (formatted.isNotEmpty()) return formatted
        }
    }
    // Raw completion mode: continue from the latest user message.
    return history.lastOrNull { it.role == "user" }?.content ?: ""
}

/** Parses `desc|nParams|size|nCtxTrain|nVocab|nCtx|version` from the JNI layer. */
internal fun parseModelInfo(name: String, raw: String): ModelInfo {
    val parts = raw.split("|")
    return ModelInfo(
        name = name,
        desc = parts.getOrNull(0)?.takeIf { it.isNotBlank() } ?: "unknown",
        nParams = parts.getOrNull(1)?.toLongOrNull() ?: 0L,
        sizeBytes = parts.getOrNull(2)?.toLongOrNull() ?: 0L,
        nCtxTrain = parts.getOrNull(3)?.toIntOrNull() ?: 0,
        nVocab = parts.getOrNull(4)?.toIntOrNull() ?: 0,
        nCtx = parts.getOrNull(5)?.toIntOrNull() ?: 0,
        llamaVersion = parts.getOrNull(6) ?: "",
    )
}

/**
 * Picks a sensible thread count from per-core max frequencies. Oversubscribing
 * little cores makes llama.cpp much slower, so use the "big" cluster only.
 */
internal fun pickThreadCount(freqs: List<Long>, availableProcessors: Int): Int {
    if (freqs.isEmpty()) return availableProcessors.coerceIn(2, 4)
    val max = freqs.max()
    return freqs.count { it >= max * 9 / 10 }.coerceIn(2, 8)
}

fun defaultThreadCount(cpuRoot: File = File("/sys/devices/system/cpu")): Int {
    return try {
        val freqs = cpuRoot
            .listFiles { f -> f.name.matches(Regex("cpu\\d+")) }
            ?.mapNotNull { cpu ->
                File(cpu, "cpufreq/cpuinfo_max_freq")
                    .takeIf { it.exists() }
                    ?.readText()?.trim()?.toLongOrNull()
            }
            ?: emptyList()
        pickThreadCount(freqs, Runtime.getRuntime().availableProcessors())
    } catch (_: Throwable) {
        Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
    }
}

fun formatBytes(bytes: Long): String {
    if (bytes <= 0) return "-"
    val units = arrayOf("B", "KB", "MB", "GB", "TB")
    var value = bytes.toDouble()
    var idx = 0
    while (value >= 1024 && idx < units.lastIndex) {
        value /= 1024
        idx++
    }
    return if (idx == 0) "%d %s".format(Locale.US, bytes, units[idx])
    else "%.2f %s".format(Locale.US, value, units[idx])
}

fun formatParams(n: Long): String {
    if (n <= 0) return "-"
    return when {
        n >= 1_000_000_000L -> "%.2fB".format(Locale.US, n / 1_000_000_000.0)
        n >= 1_000_000L -> "%.2fM".format(Locale.US, n / 1_000_000.0)
        else -> "$n"
    }
}
