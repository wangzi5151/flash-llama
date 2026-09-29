# 极速 GGUF · FlashLlama

一个轻量、原生、**速度优先**的 Android 本地大模型运行器。直接在手机上加载
`.gguf` 模型（Llama / Qwen / DeepSeek / Phi / Gemma 等），离线推理、流式输出。

底层基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `libllama` +
`libggml`，通过 JNI 调用，纯 CPU 推理并开启 arm64 NEON / DOTPROD / CPU
Flash-Attention 加速。

---

## ✨ 特性

- **真·原生 GGUF 推理**：不依赖服务端、不联网，模型全部在手机上运行。
- **速度优先**：
  - arm64 NEON + `i8mm`/`dotprod` 量化矩阵乘；
  - 自动启用 CPU Flash-Attention；
  - Prompt 批处理（默认 batch 512）；
  - `mmap` 零拷贝加载模型，启动快、内存占用低；
  - 多线程（默认使用设备大核数量）。
- **流式输出**：逐字/逐 token 实时显示，带 tokens/s 速度统计。
- **完整对话能力**：自动套用模型内置 chat 模板（无模板时回退 ChatML），支持系统提示词与多轮上下文。
- **无需复制模型**：直接通过 `content://` 的文件描述符加载，几十 GB 模型也不必先拷贝一份。
- **可调参数**：线程数、上下文长度、最大生成长度、温度、top-k / top-p / min-p、重复惩罚、KV 缓存精度（F16 / Q8_0 / Q4_0）、mlock 锁定内存。
- **流式可中断**：生成过程中可随时点击「停止」。
- **深色/浅色主题**：Material 3 + 动态取色。

## 🧠 性能说明

运行库是**在设备上原生编译**的（`GGML_NATIVE=ON`），CPU 后端会带上本机 SoC
支持的指令集（NEON / dotprod / fp16 …）。相比通用预编译包，量化矩阵乘可以快
**2～2.5 倍**。

实测（联发科 MT6878 / Dimensity 7300，Llama-3.2-1B-Instruct Q4_K_M，约 0.76GB）：

| 配置 | 生成速度 |
| --- | --- |
| 2 线程 + Flash-Attention | ~14.5 tok/s |
| 4 线程 + Flash-Attention | ~13 tok/s |
| 8 线程（错误示范） | ~1.5 tok/s |

> 线程数并非越多越快：超过大核数量后会互相抢核、明显掉速。应用默认取「大核频率
> 对应的核心数」。

实际速度取决于 SoC 与模型大小。建议使用 `Q4_K_M` 量化。

## 🏗️ 架构

```
app/src/main/
├── java/com/wangzi5151/flashllama/
│   ├── LlamaBridge.kt      # JNI 方法声明（init / generate / cancel / ...）
│   ├── ChatViewModel.kt    # 模型加载、对话状态、参数、tokens/s 统计
│   ├── MainActivity.kt     # Jetpack Compose 界面
│   └── ui/Theme.kt
├── cpp/gguf_jni.cpp        # JNI 桥接：tokenize / decode / 采样 / UTF-8 流式回调
└── jniLibs/arm64-v8a/      # 预编译的 llama.cpp 运行库
    ├── libggufjni.so       # 本项目编译的 JNI 桥
    ├── libllama.so
    ├── libggml.so
    ├── libggml-base.so
    ├── libggml-cpu.so      # CPU 后端（ggml 动态后端机制按需加载）
    └── libc++_shared.so
```

- `libggml.so` 依赖 `libggml-cpu.so`，CPU 后端在库加载时自动注册。JNI 仍会
  调用 `ggml_backend_load_all_from_path(<nativeLibraryDir>)` 以兼容动态后端
  构建；APK 设置了 `android:extractNativeLibs="true"` 以得到真实的库目录。
- `libc++_shared.so` 随构建工具链一起打包，保证与运行库 ABI 完全匹配。

## 🔧 构建

### 方式一：在 Termux 中直接构建（本仓库的开发方式）

需要 Android SDK（build-tools、platform 34）、JDK 17+。

```bash
pkg install -y cmake ninja patchelf python openjdk-17
```

1. 构建原生运行库（从源码编译，带本机 CPU 优化）：

   ```bash
   ./build_native.sh
   ```

   脚本会通过 PyPI 拉取 llama.cpp 源码（`llama-cpp-python` 的 sdist 内置完整
   源码），用 Termux clang 以 `GGML_NATIVE=ON` 编译出 `libllama.so` /
   `libggml*.so`，修正 soname 后连同 JNI 桥一起放入
   `app/src/main/jniLibs/arm64-v8a/`。

   > 首次构建会下载约 75MB 源码并编译数分钟。由于使用 `GGML_NATIVE`，请在你
   > 要运行的设备（或同架构设备）上编译以获得最佳指令集。

2. 配置 `local.properties`：

   ```properties
   sdk.dir=/path/to/android-sdk
   ```

3. 打包：

   ```bash
   ./gradlew :app:assembleRelease
   ```

   > 在 Termux aarch64 上，Gradle 自带的 `aapt2`（x86_64）无法运行，需在
   > `gradle.properties` 中设置 `android.aapt2FromMavenOverride` 指向 Termux
   > 的 aapt2，或 `pkg install aapt`。本仓库已包含该配置。

### 方式二：Android Studio / 标准 NDK

标准环境也可直接构建，只要 `jniLibs/arm64-v8a/` 中已有运行库；
本项目默认不启用 `externalNativeBuild`，JNI 由 `build_native.sh` 预先编译。

## 📱 使用

1. 安装 APK（仅支持 `arm64-v8a`）。
2. 点击「选择 GGUF 模型」，通过系统文件选择器选取 `.gguf` 文件（可使用 SAF，
   无需把模型复制到内部存储）。
3. 等待模型加载完成后，输入消息即可开始对话；顶部会显示实时 tokens/s。
4. 「参数设置」中可调整线程数、上下文长度、采样参数等。

## ⚠️ 已知限制

- 仅打包 `arm64-v8a`（绝大多数现代 Android 手机）。
- 分片（split）GGUF 暂不支持通过 SAF 直接加载，请使用单文件模型。
- 未启用 GPU / NPU 后端；纯 CPU 推理，兼容性最好、功耗可控。

## 📄 许可

本项目代码采用 [MIT License](LICENSE)。

打包的 llama.cpp / ggml 运行库同样为 MIT 许可，详见
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)。
