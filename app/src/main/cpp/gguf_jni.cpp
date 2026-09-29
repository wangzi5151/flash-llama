// FlashLlama - JNI bridge between the Android app and llama.cpp (GGUF runtime).
//
// The native side keeps a loaded llama_model + llama_context alive across many
// generation calls. Prompt processing, sampling and detokenization all happen
// here; tokens are streamed back to Kotlin through a callback so the UI can
// render text while it is produced.

#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"

#define TAG "FlashLlama"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct Session {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    FILE *              file  = nullptr; // kept open for lazily-read tensors
    int                 n_ctx = 0;

    std::atomic<bool> cancel{false};
    std::mutex        mtx;
};

std::atomic<bool> g_backends_loaded{false};

std::mutex  g_log_mtx;
std::string g_log_path;

void log_to_file(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (g_log_path.empty()) {
        return;
    }
    char lvl = 'I';
    if (level == GGML_LOG_LEVEL_ERROR) lvl = 'E';
    else if (level == GGML_LOG_LEVEL_WARN) lvl = 'W';
    else if (level == GGML_LOG_LEVEL_DEBUG) lvl = 'D';
    std::lock_guard<std::mutex> lock(g_log_mtx);
    FILE * f = fopen(g_log_path.c_str(), "a");
    if (f == nullptr) {
        return;
    }
    fprintf(f, "[%c] %s", lvl, text);
    fclose(f);
}

void ensure_backends(const char * dir) {
    bool expected = false;
    if (!g_backends_loaded.compare_exchange_strong(expected, true)) {
        return;
    }
    if (dir != nullptr && dir[0] != '\0') {
        LOGI("loading ggml backends from %s", dir);
        ggml_backend_load_all_from_path(dir);
    } else {
        LOGI("loading ggml backends from default paths");
        ggml_backend_load_all();
    }
    LOGI("backend registries=%zu devices=%zu",
         ggml_backend_reg_count(), ggml_backend_dev_count());

    // Some Android linker namespaces refuse the directory scan; load the CPU
    // backend by explicit path as a fallback.
    if (ggml_backend_dev_count() == 0 && dir != nullptr && dir[0] != '\0') {
        std::string cpu = std::string(dir) + "/libggml-cpu.so";
        LOGW("no backend device found, explicit load: %s", cpu.c_str());
        ggml_backend_load(cpu.c_str());
        LOGI("after explicit load registries=%zu devices=%zu",
             ggml_backend_reg_count(), ggml_backend_dev_count());
    }
    if (ggml_backend_dev_count() == 0) {
        LOGE("no ggml backend device available!");
    }
}

bool abort_cb(void * data) {
    return static_cast<Session *>(data)->cancel.load();
}

// Number of leading bytes of `s` that form complete, valid UTF-8 sequences.
size_t utf8_complete_len(const std::string & s) {
    size_t i    = 0;
    size_t last = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t need;
        if (c < 0x80) {
            need = 1;
        } else if ((c >> 5) == 0x6) {
            need = 2;
        } else if ((c >> 4) == 0xE) {
            need = 3;
        } else if ((c >> 3) == 0x1E) {
            need = 4;
        } else {
            ++i; // stray continuation / invalid byte, skip it
            last = i;
            continue;
        }
        if (i + need > s.size()) {
            break; // incomplete trailing sequence -> wait for more bytes
        }
        bool ok = true;
        for (size_t k = 1; k < need; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            ++i;
            last = i;
            continue;
        }
        i += need;
        last = i;
    }
    return last;
}

jstring utf8_to_jstring(JNIEnv * env, const std::string & s) {
    std::vector<jchar> w;
    w.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp;
        size_t   len;
        if (c < 0x80) {
            cp = c; len = 1;
        } else if ((c >> 5) == 0x6) {
            cp = c & 0x1F; len = 2;
        } else if ((c >> 4) == 0xE) {
            cp = c & 0x0F; len = 3;
        } else if ((c >> 3) == 0x1E) {
            cp = c & 0x07; len = 4;
        } else {
            ++i; continue;
        }
        if (i + len > s.size()) {
            break;
        }
        bool ok = true;
        for (size_t k = 1; k < len; ++k) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { ++i; continue; }
        i += len;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            w.push_back(static_cast<jchar>(0xD800 + (cp >> 10)));
            w.push_back(static_cast<jchar>(0xDC00 + (cp & 0x3FF)));
        } else {
            w.push_back(static_cast<jchar>(cp));
        }
    }
    return env->NewString(w.data(), static_cast<jsize>(w.size()));
}

std::string jstring_to_utf8(JNIEnv * env, jstring js) {
    if (js == nullptr) {
        return {};
    }
    const jchar * u = env->GetStringChars(js, nullptr);
    const jsize   n = env->GetStringLength(js);
    std::string out;
    out.reserve(static_cast<size_t>(n));
    for (jsize i = 0; i < n; ++i) {
        uint32_t c = u[i];
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < n && u[i + 1] >= 0xDC00 && u[i + 1] < 0xE000) {
            c = 0x10000 + ((c - 0xD800) << 10) + (u[i + 1] - 0xDC00);
            ++i;
        }
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (c >> 12)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (c >> 18)));
            out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    env->ReleaseStringChars(js, u);
    return out;
}

ggml_type kv_type_from_int(jint v) {
    switch (v) {
        case 1: return GGML_TYPE_F16;
        case 2: return GGML_TYPE_Q8_0;
        case 3: return GGML_TYPE_Q4_0;
        default: return GGML_TYPE_COUNT; // keep library default
    }
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               add_special, true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           add_special, true);
    }
    if (n < 0) {
        tokens.clear();
        return tokens;
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

} // namespace

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *, void *) {
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeInit(
        JNIEnv * env, jobject /*thiz*/,
        jstring jmodelPath, jstring jlibDir,
        jint nThreads, jint nCtx, jint nBatch,
        jint nGpuLayers, jboolean mLock, jint kvType) {

    const std::string modelPath = jstring_to_utf8(env, jmodelPath);
    const std::string libDir    = jstring_to_utf8(env, jlibDir);

    ensure_backends(libDir.c_str());
    llama_backend_init();

    auto * s = new Session();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = nGpuLayers;
    mp.load_mode    = mLock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP;

    LOGI("loading model from %s (threads=%d ctx=%d batch=%d)", modelPath.c_str(), nThreads, nCtx, nBatch);
    s->model = llama_model_load_from_file(modelPath.c_str(), mp);
    if (s->model == nullptr) {
        LOGE("failed to load model: %s", modelPath.c_str());
        delete s;
        return 0;
    }
    s->vocab = llama_model_get_vocab(s->model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = static_cast<uint32_t>(nCtx > 0 ? nCtx : 4096);
    cp.n_batch         = static_cast<uint32_t>(nBatch > 0 ? nBatch : 512);
    cp.n_ubatch        = cp.n_batch;
    cp.n_threads       = nThreads;
    cp.n_threads_batch = nThreads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO; // CPU flash-attention when supported
    cp.no_perf         = true;
    cp.abort_callback      = abort_cb;
    cp.abort_callback_data = s;

    const ggml_type kvt = kv_type_from_int(kvType);
    if (kvt != GGML_TYPE_COUNT) {
        cp.type_k = kvt;
        cp.type_v = kvt;
    }

    s->ctx = llama_init_from_model(s->model, cp);
    if (s->ctx == nullptr) {
        LOGE("failed to create context");
        llama_model_free(s->model);
        delete s;
        return 0;
    }
    s->n_ctx = static_cast<int>(llama_n_ctx(s->ctx));

    LOGI("model ready: %s (n_ctx=%d, threads=%d)", modelPath.c_str(), s->n_ctx, nThreads);
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeSetLogPath(
        JNIEnv * env, jobject /*thiz*/, jstring jpath) {
    std::lock_guard<std::mutex> lock(g_log_mtx);
    g_log_path = jstring_to_utf8(env, jpath);
    llama_log_set(log_to_file, nullptr);
}

namespace {

llama_context * make_context(Session * s, int nThreads, int nCtx, int nBatch, jint kvType) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = static_cast<uint32_t>(nCtx > 0 ? nCtx : 4096);
    cp.n_batch         = static_cast<uint32_t>(nBatch > 0 ? nBatch : 512);
    cp.n_ubatch        = cp.n_batch;
    cp.n_threads       = nThreads;
    cp.n_threads_batch = nThreads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.no_perf         = true;
    cp.abort_callback      = abort_cb;
    cp.abort_callback_data = s;

    const ggml_type kvt = kv_type_from_int(kvType);
    if (kvt != GGML_TYPE_COUNT) {
        cp.type_k = kvt;
        cp.type_v = kvt;
    }
    return llama_init_from_model(s->model, cp);
}

} // namespace

// Loads a model from an already-open file descriptor (SAF / content:// URIs).
// Re-opening /proc/self/fd/N is unreliable inside the app sandbox, so hand the
// fd straight to llama.cpp instead.
JNIEXPORT jlong JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeInitFd(
        JNIEnv * env, jobject /*thiz*/, jint fd, jstring jlibDir,
        jint nThreads, jint nCtx, jint nBatch, jint nGpuLayers, jboolean mLock, jint kvType) {

    const std::string libDir = jstring_to_utf8(env, jlibDir);
    ensure_backends(libDir.c_str());
    llama_backend_init();

    auto * s = new Session();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = nGpuLayers;
    mp.load_mode    = mLock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP;

    const int dfd = dup(fd);
    if (dfd < 0) {
        LOGE("dup(fd=%d) failed", fd);
        delete s;
        return 0;
    }
    s->file = fdopen(dfd, "rb");
    if (s->file == nullptr) {
        LOGE("fdopen(fd=%d) failed", dfd);
        close(dfd);
        delete s;
        return 0;
    }

    LOGI("loading model from fd=%d (threads=%d ctx=%d batch=%d)", fd, nThreads, nCtx, nBatch);
    s->model = llama_model_load_from_file_ptr(s->file, mp);
    if (s->model == nullptr) {
        LOGE("failed to load model from fd=%d", fd);
        fclose(s->file);
        delete s;
        return 0;
    }
    s->vocab = llama_model_get_vocab(s->model);

    s->ctx = make_context(s, nThreads, nCtx, nBatch, kvType);
    if (s->ctx == nullptr) {
        LOGE("failed to create context (fd path)");
        llama_model_free(s->model);
        fclose(s->file);
        delete s;
        return 0;
    }
    s->n_ctx = static_cast<int>(llama_n_ctx(s->ctx));
    LOGI("model ready (fd path): n_ctx=%d", s->n_ctx);
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeFree(JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    auto * s = reinterpret_cast<Session *>(handle);
    if (s == nullptr) {
        return;
    }
    if (s->ctx != nullptr) {
        llama_free(s->ctx);
    }
    if (s->model != nullptr) {
        llama_model_free(s->model);
    }
    if (s->file != nullptr) {
        fclose(s->file);
    }
    delete s;
}

JNIEXPORT void JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeCancel(JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    auto * s = reinterpret_cast<Session *>(handle);
    if (s != nullptr) {
        s->cancel.store(true);
    }
}

JNIEXPORT void JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeReset(JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    auto * s = reinterpret_cast<Session *>(handle);
    if (s != nullptr && s->ctx != nullptr) {
        std::lock_guard<std::mutex> lock(s->mtx);
        llama_memory_clear(llama_get_memory(s->ctx), true);
    }
}

JNIEXPORT jstring JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeInfo(JNIEnv * env, jobject /*thiz*/, jlong handle) {
    auto * s = reinterpret_cast<Session *>(handle);
    if (s == nullptr || s->model == nullptr) {
        return env->NewStringUTF("");
    }
    char desc[256] = {0};
    llama_model_desc(s->model, desc, sizeof(desc));
    char out[768];
    std::snprintf(out, sizeof(out), "%s|%llu|%llu|%d|%d|%d|%s",
                  desc,
                  static_cast<unsigned long long>(llama_model_n_params(s->model)),
                  static_cast<unsigned long long>(llama_model_size(s->model)),
                  llama_model_n_ctx_train(s->model),
                  llama_vocab_n_tokens(s->vocab),
                  s->n_ctx,
                  llama_version());
    return utf8_to_jstring(env, out);
}

JNIEXPORT jstring JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeApplyChatTemplate(
        JNIEnv * env, jobject /*thiz*/, jlong handle,
        jobjectArray roles, jobjectArray contents, jboolean addAss) {

    auto * s = reinterpret_cast<Session *>(handle);
    if (s == nullptr || s->model == nullptr) {
        return env->NewStringUTF("");
    }

    const jsize n = env->GetArrayLength(roles);
    std::vector<std::string>       r(n), c(n);
    std::vector<llama_chat_message> msgs(n);
    for (jsize i = 0; i < n; ++i) {
        auto jr = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto jc = static_cast<jstring>(env->GetObjectArrayElement(contents, i));
        r[i] = jstring_to_utf8(env, jr);
        c[i] = jstring_to_utf8(env, jc);
        env->DeleteLocalRef(jr);
        env->DeleteLocalRef(jc);
        msgs[i].role    = r[i].c_str();
        msgs[i].content = c[i].c_str();
    }

    const char * tmpl = llama_model_chat_template(s->model, nullptr);
    if (tmpl == nullptr) {
        tmpl = "chatml";
    }

    int32_t need = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), addAss, nullptr, 0);
    if (need <= 0) {
        return env->NewStringUTF("");
    }
    std::vector<char> buf(static_cast<size_t>(need) + 1);
    int32_t written = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), addAss, buf.data(),
                                                static_cast<int32_t>(buf.size()));
    if (written <= 0) {
        return env->NewStringUTF("");
    }
    return utf8_to_jstring(env, std::string(buf.data(), static_cast<size_t>(written)));
}

JNIEXPORT jint JNICALL
Java_com_wangzi5151_flashllama_LlamaBridge_nativeGenerate(
        JNIEnv * env, jobject /*thiz*/, jlong handle, jstring jprompt,
        jint maxTokens, jfloat temp, jint topK, jfloat topP, jfloat minP,
        jfloat repeatPenalty, jint repeatLastN, jint seed, jobject callback) {

    auto * s = reinterpret_cast<Session *>(handle);
    if (s == nullptr || s->ctx == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(s->mtx);
    s->cancel.store(false);

    const std::string prompt = jstring_to_utf8(env, jprompt);

    llama_memory_t mem = llama_get_memory(s->ctx);
    llama_memory_clear(mem, true);

    std::vector<llama_token> tokens = tokenize(s->vocab, prompt, true);
    if (tokens.empty()) {
        tokens.push_back(llama_vocab_bos(s->vocab));
    }

    // Keep the prompt within the context window while leaving room to generate.
    const int reserve = maxTokens > 0 ? maxTokens : 256;
    int keep = s->n_ctx - reserve - 4;
    if (keep < 1) {
        keep = s->n_ctx / 2;
    }
    if (static_cast<int>(tokens.size()) > keep) {
        tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - keep));
    }

    const int n_batch = static_cast<int>(llama_n_batch(s->ctx));
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const int chunk = static_cast<int>(std::min<size_t>(n_batch, tokens.size() - i));
        llama_batch b = llama_batch_get_one(tokens.data() + i, chunk);
        const int ret = llama_decode(s->ctx, b);
        if (ret != 0) {
            LOGE("prompt decode failed: %d", ret);
            return -2;
        }
        if (s->cancel.load()) {
            return -3;
        }
    }

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sp);

    if (repeatPenalty > 0.0f && repeatPenalty != 1.0f && repeatLastN > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                llama_vocab_n_tokens(s->vocab), repeatLastN, repeatPenalty, 0.0f, 0.0f));
    }
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (topK > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
        }
        if (topP > 0.0f && topP < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
        }
        if (minP > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(minP, 1));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        const uint32_t s_seed = (seed < 0) ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed);
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(s_seed));
    }

    jclass    cbClass  = env->GetObjectClass(callback);
    jmethodID cbMethod = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    env->DeleteLocalRef(cbClass);

    std::string pending;
    int generated = 0;
    llama_token cur = llama_sampler_sample(smpl, s->ctx, -1);

    while (generated < maxTokens) {
        if (s->cancel.load()) {
            break;
        }
        if (cur == LLAMA_TOKEN_NULL || llama_vocab_is_eog(s->vocab, cur)) {
            break;
        }

        char piece[128];
        int np = llama_token_to_piece(s->vocab, cur, piece, sizeof(piece), 0, false);
        if (np < 0) {
            std::vector<char> big(static_cast<size_t>(-np));
            np = llama_token_to_piece(s->vocab, cur, big.data(), static_cast<int>(big.size()), 0, false);
            if (np > 0) {
                pending.append(big.data(), static_cast<size_t>(np));
            }
        } else if (np > 0) {
            pending.append(piece, static_cast<size_t>(np));
        }

        const size_t complete = utf8_complete_len(pending);
        if (complete > 0) {
            const std::string out = pending.substr(0, complete);
            pending.erase(0, complete);
            jstring js = utf8_to_jstring(env, out);
            if (js != nullptr) {
                env->CallVoidMethod(callback, cbMethod, js);
                env->DeleteLocalRef(js);
            }
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                break;
            }
        }

        ++generated;

        llama_batch b = llama_batch_get_one(&cur, 1);
        const int ret = llama_decode(s->ctx, b);
        if (ret != 0) {
            LOGW("decode failed during generation: %d", ret);
            break;
        }
        cur = llama_sampler_sample(smpl, s->ctx, -1);
    }

    if (!pending.empty()) {
        const size_t complete = utf8_complete_len(pending);
        if (complete > 0) {
            jstring js = utf8_to_jstring(env, pending.substr(0, complete));
            if (js != nullptr) {
                env->CallVoidMethod(callback, cbMethod, js);
                env->DeleteLocalRef(js);
            }
        }
    }

    llama_sampler_free(smpl);
    return generated;
}

} // extern "C"
