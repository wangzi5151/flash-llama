package com.wangzi5151.flashllama

/** Streams decoded pieces of text back to the caller while generation runs. */
fun interface TokenCallback {
    fun onToken(text: String)
}

/**
 * Thin JNI surface over the bundled llama.cpp runtime. Everything here maps
 * 1:1 to `gguf_jni.cpp`.
 */
object LlamaBridge {

    init {
        System.loadLibrary("ggufjni")
    }

    /** Routes llama.cpp/ggml log output to the given file. */
    external fun nativeSetLogPath(path: String)

    /** Loads a model and creates a reusable context. Returns a handle, or 0 on failure. */
    external fun nativeInit(
        modelPath: String,
        libDir: String,
        nThreads: Int,
        nCtx: Int,
        nBatch: Int,
        nGpuLayers: Int,
        mLock: Boolean,
        kvType: Int,
    ): Long

    /** Same as [nativeInit] but reads from an already-open file descriptor. */
    external fun nativeInitFd(
        fd: Int,
        libDir: String,
        nThreads: Int,
        nCtx: Int,
        nBatch: Int,
        nGpuLayers: Int,
        mLock: Boolean,
        kvType: Int,
    ): Long

    external fun nativeFree(handle: Long)

    external fun nativeCancel(handle: Long)

    external fun nativeReset(handle: Long)

    /** Returns "desc|nParams|size|nCtxTrain|nVocab|nCtx|version". */
    external fun nativeInfo(handle: Long): String

    /** Applies the model's built-in chat template (falls back to ChatML). */
    external fun nativeApplyChatTemplate(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>,
        addAssistant: Boolean,
    ): String

    /** Runs a full generation. Returns the number of tokens produced. */
    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temp: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        callback: TokenCallback,
    ): Int
}
