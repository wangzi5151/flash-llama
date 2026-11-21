# 参与贡献

欢迎提 issue 和 PR。先说清这个仓库的特殊之处，省得白费功夫：

## 原生库不在普通 CI 里构建

`app/src/main/jniLibs/arm64-v8a/` 里的 llama.cpp 运行库是用 Termux clang
以 `GGML_NATIVE=ON` 在真机/ARM64 设备上编译的（见 README「构建」一节和
`build_native.sh`）。GitHub 托管 runner **不**构建原生库，所以：

- 只改 Kotlin / 资源 / Gradle 的 PR，靠 `android-unit-tests.yml` 跑 JVM
  单测验证即可，不需要碰原生库。
- 改 `app/src/main/cpp/gguf_jni.cpp` 或 `build_native.sh` 的 PR，必须说明
  你在哪台设备上重新构建并实测过（设备型号 + 用的模型 + tok/s 变化），
  没实测过的原生改动不会合并。

## 开发环境

- JDK 17+、Android SDK（platform 34、build-tools）
- 桌面环境用 Android Studio 或命令行均可；Termux 构建的注意事项
  （aapt2 覆盖等）见 README

## 提交前自查

```bash
./gradlew testDebugUnitTest   # 纯逻辑单测，不需要设备和模型文件
```

单测位于 `app/src/test/`，覆盖 prompt 组装、模型信息解析、容量格式化、
大核线程数选择。新增纯逻辑时请顺手补测。

## 提 issue

- 崩溃/加载失败：带上设备型号、Android 版本、模型名与量化等级、
  文件大小，以及是点选文件后立刻失败还是生成中途失败。
- 速度问题：带上参数设置页的线程数。先看 README「性能说明」——
  线程数超过大核数会明显掉速，这不是 bug。
