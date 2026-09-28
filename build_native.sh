#!/data/data/com.termux/files/usr/bin/bash
# Build the FlashLlama JNI bridge and stage all native libraries into jniLibs.
#
# The llama.cpp runtime is provided by the Termux `llama-cpp` package, which is
# compiled against Android bionic and only depends on system libraries, so its
# shared objects can be bundled straight into the APK.
#
#   pkg install -y llama-cpp
#   ./build_native.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
CXX="$PREFIX/bin/clang++"
JNI_OUT="$ROOT/app/src/main/jniLibs/arm64-v8a"
SRC="$ROOT/app/src/main/cpp/gguf_jni.cpp"

LIBS=(libllama.so libggml.so libggml-base.so libggml-cpu.so libc++_shared.so)

mkdir -p "$JNI_OUT"

echo ">> compiling libggufjni.so"
"$CXX" -shared -fPIC -O3 -std=c++17 -DNDEBUG \
    -fvisibility=hidden \
    -I"$PREFIX/include" \
    "$SRC" -o "$JNI_OUT/libggufjni.so" \
    -L"$PREFIX/lib" -lllama -lggml -lggml-base -llog

echo ">> staging runtime libraries"
for lib in "${LIBS[@]}"; do
    src="$PREFIX/lib/$lib"
    if [ -f "$src" ]; then
        cp -f "$src" "$JNI_OUT/$lib"
    else
        echo "!! missing $src (run: pkg install llama-cpp)" >&2
        exit 1
    fi
done

echo ">> jniLibs:"
ls -la "$JNI_OUT"
