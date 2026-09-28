package com.wangzi5151.flashllama

import android.app.Application
import android.content.ContentResolver
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class ChatMessage(
    val role: String,
    val content: String,
    val streaming: Boolean = false,
)

data class GenParams(
    val nThreads: Int,
    val nCtx: Int = 4096,
    val nBatch: Int = 512,
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val maxTokens: Int = 512,
    val kvType: Int = 0, // 0 = f16, 2 = q8_0, 3 = q4_0
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

    private val contentResolver: ContentResolver = app.contentResolver
    private val nativeLibDir: String = app.applicationInfo.nativeLibraryDir

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _params = MutableStateFlow(
        GenParams(nThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8))
    )
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

    private var handle: Long = 0L
    private var modelPfd: ParcelFileDescriptor? = null

    val isReady: Boolean get() = handle != 0L

    fun updateParams(transform: (GenParams) -> GenParams) {
        _params.update(transform)
    }

    fun loadModel(uri: Uri) {
        if (_loading.value || _generating.value) return
        _loading.value = true
        _speed.value = ""
        _status.value = "正在加载模型…"
        viewModelScope.launch(Dispatchers.Default) {
            var newPfd: ParcelFileDescriptor? = null
            try {
                val name = queryDisplayName(uri)
                val resolved = resolveModelPath(uri)
                newPfd = resolved.second

                withContext(Dispatchers.Main) { releaseHandle() }
                modelPfd = newPfd

                val p = _params.value
                val h = LlamaBridge.nativeInit(
                    resolved.first, nativeLibDir, p.nThreads, p.nCtx, p.nBatch,
                    nGpuLayers = 0, mLock = p.useMlock, kvType = p.kvType,
                )
                if (h == 0L) {
                    _status.value = "模型加载失败，请确认是有效的 GGUF 文件"
                    newPfd?.close()
                    modelPfd = null
                } else {
                    handle = h
                    _modelInfo.value = parseInfo(name, LlamaBridge.nativeInfo(h))
                    _status.value = "模型已就绪"
                    _messages.value = emptyList()
                }
            } catch (t: Throwable) {
                _status.value = "加载出错：${t.message}"
                newPfd?.close()
                modelPfd = null
            } finally {
                _loading.value = false
            }
        }
    }

    fun send(input: String) {
        val text = input.trim()
        if (text.isEmpty() || handle == 0L || _generating.value) return

        val history = _messages.value + ChatMessage("user", text)
        val assistantIndex = history.size
        _messages.value = history + ChatMessage("assistant", "", streaming = true)
        _generating.value = true
        _speed.value = ""

        viewModelScope.launch(Dispatchers.Default) {
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
                    handle, prompt, p.maxTokens, p.temperature, p.topK, p.topP,
                    p.minP, p.repeatPenalty, 256, -1, callback,
                )
            } catch (t: Throwable) {
                buffer.append("\n[错误] ${t.message}")
            } finally {
                val elapsed = (System.currentTimeMillis() - started).coerceAtLeast(1)
                publishAssistant(assistantIndex, buffer.toString(), streaming = false)
                val tps = tokenCount * 1000f / elapsed
                _speed.value = "%.1f tok/s · %d tokens".format(tps, tokenCount)
                _generating.value = false
            }
        }
    }

    fun stop() {
        val h = handle
        if (h != 0L) LlamaBridge.nativeCancel(h)
    }

    fun clearChat() {
        _messages.value = emptyList()
        _speed.value = ""
        val h = handle
        if (h != 0L) {
            viewModelScope.launch(Dispatchers.Default) { LlamaBridge.nativeReset(h) }
        }
    }

    fun unload() {
        releaseHandle()
        _modelInfo.value = null
        _status.value = "请选择 GGUF 模型文件"
        _speed.value = ""
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
        if (p.useChatTemplate && handle != 0L) {
            val roles = ArrayList<String>(history.size)
            val contents = ArrayList<String>(history.size)
            for (m in history) {
                if (m.content.isBlank()) continue
                roles.add(m.role)
                contents.add(m.content)
            }
            if (roles.isNotEmpty()) {
                val formatted = LlamaBridge.nativeApplyChatTemplate(
                    handle, roles.toTypedArray(), contents.toTypedArray(), true,
                )
                if (formatted.isNotEmpty()) return formatted
            }
        }
        // Raw completion mode: continue from the latest user message.
        return history.lastOrNull { it.role == "user" }?.content ?: ""
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

    private fun parseInfo(name: String, raw: String): ModelInfo {
        val parts = raw.split("|")
        return ModelInfo(
            name = name,
            desc = parts.getOrElse(0) { "unknown" },
            nParams = parts.getOrNull(1)?.toLongOrNull() ?: 0L,
            sizeBytes = parts.getOrNull(2)?.toLongOrNull() ?: 0L,
            nCtxTrain = parts.getOrNull(3)?.toIntOrNull() ?: 0,
            nVocab = parts.getOrNull(4)?.toIntOrNull() ?: 0,
            nCtx = parts.getOrNull(5)?.toIntOrNull() ?: 0,
            llamaVersion = parts.getOrNull(6) ?: "",
        )
    }

    private fun releaseHandle() {
        val h = handle
        if (h != 0L) {
            LlamaBridge.nativeFree(h)
            handle = 0L
        }
        modelPfd?.close()
        modelPfd = null
    }

    override fun onCleared() {
        releaseHandle()
        super.onCleared()
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
    return if (idx == 0) "%d %s".format(bytes, units[idx])
    else "%.2f %s".format(value, units[idx])
}

fun formatParams(n: Long): String {
    if (n <= 0) return "-"
    return when {
        n >= 1_000_000_000L -> "%.2fB".format(n / 1_000_000_000.0)
        n >= 1_000_000L -> "%.2fM".format(n / 1_000_000.0)
        else -> "$n"
    }
}
