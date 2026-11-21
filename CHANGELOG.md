# Changelog

All notable changes to FlashLlama are documented here.

## [Unreleased]

### Added

- README: CI / license badges, an English summary, a model download &
  selection guide (where to get GGUF files, Q4_K_M size vs. device RAM
  table, why split files won't load), and explicit device requirements
  (Android 8.0+, arm64-v8a) in the usage section.
- `CONTRIBUTING.md` (native-library build constraints, how to run the
  JVM unit tests, what to include in issues) and `SECURITY.md`
  (vulnerability reporting, trusted-model guidance).
- Repository topics: android, gguf, jni, kotlin, llama-cpp, llm,
  on-device-ai.

### Fixed

- `gradle.properties` no longer hardcodes the Termux-local
  `android.aapt2FromMavenOverride` path. That path only exists on the
  maintainer's Termux device, so GitHub-hosted CI always failed in
  `:app:processDebugResources` ("Specified AAPT2 executable does not
  exist"). Termux builds now pass it via `~/.gradle/gradle.properties`
  or `-P` on the command line, as the README's test example already did.
- Prevent use-after-free when a model is unloaded while a generation is still
  running: `unload()` / `onCleared()` now cancel the native decode, wait for the
  in-flight JNI call, and only then call `nativeFree()`.
- Native sessions are reference-counted (`SessionLease`): `nativeFree()` flips a
  closing flag and waits for all active operations (generate / reset / info /
  chat template / cancel) before destroying the llama model and context.
- Model loading no longer publishes a handle before metadata parsing, and a load
  racing with `unload()` is abandoned and freed instead of resurrecting the old
  session.

### Added

- JVM unit tests under `app/src/test/` for prompt composition, model info
  parsing, byte/param formatting, and big-core thread selection.
- GitHub Actions workflow that runs `./gradlew testDebugUnitTest` on push/PR,
  plus a manual `workflow_dispatch` workflow for the on-device native build.
- This changelog and expanded build/test/limitations documentation.

### Changed

- `build_native.sh` now pins `llama-cpp-python==0.3.36` and verifies the sdist
  SHA-256 before extracting the vendored llama.cpp tree, making native builds
  reproducible.

### Tests

- `./gradlew testDebugUnitTest`
