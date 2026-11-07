#!/data/data/com.termux/files/usr/bin/bash
# Build the FlashLlama native runtime (llama.cpp + ggml + JNI bridge) for
# Android arm64 and stage everything into app/src/main/jniLibs/arm64-v8a.
#
# The runtime is compiled *on device* with GGML_NATIVE=ON so the CPU backend is
# built with the full instruction set of the host SoC (NEON, dotprod, ...).
# This is what makes generation several times faster than the generic
# prebuilt package.
#
# Requirements (Termux):
#   pkg install -y cmake ninja patchelf python
#
# Usage:
#   ./build_native.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
JNI_OUT="$ROOT/app/src/main/jniLibs/arm64-v8a"
SRC="$ROOT/app/src/main/cpp/gguf_jni.cpp"
THIRD_PARTY="$ROOT/third_party"
LLAMA_SRC="$THIRD_PARTY/llama.cpp"
BUILD="$THIRD_PARTY/build-android"

CXX="$PREFIX/bin/clang++"
JOBS="$(nproc 2>/dev/null || echo 4)"

# Pinned for reproducible builds. The sdist filename and its SHA-256 are checked
# so a compromised/mutated mirror cannot silently swap the vendored llama.cpp.
LLAMA_CPP_PY_VERSION="0.3.36"
LLAMA_CPP_PY_SHA256="832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e"
LLAMA_CPP_PY_SDIST="$THIRD_PARTY/dl/llama_cpp_python-${LLAMA_CPP_PY_VERSION}.tar.gz"

# ---------------------------------------------------------------------------
# 1. Fetch the llama.cpp source. PyPI is a fast, reliable mirror because the
#    llama-cpp-python sdist vendors the full llama.cpp tree.
# ---------------------------------------------------------------------------
if [ ! -f "$LLAMA_SRC/include/llama.h" ]; then
    echo ">> fetching llama.cpp source via PyPI (llama-cpp-python ${LLAMA_CPP_PY_VERSION})"
    mkdir -p "$THIRD_PARTY/dl"
    if [ ! -f "$LLAMA_CPP_PY_SDIST" ]; then
        python -m pip download --no-deps --no-binary :all: \
            --dest "$THIRD_PARTY/dl" "llama-cpp-python==${LLAMA_CPP_PY_VERSION}"
    fi
    echo "${LLAMA_CPP_PY_SHA256}  ${LLAMA_CPP_PY_SDIST}" | sha256sum -c -
    EXTRACT_DIR="$THIRD_PARTY/dl/src"
    rm -rf "$EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"
    tar xzf "$LLAMA_CPP_PY_SDIST" -C "$EXTRACT_DIR"
    rm -rf "$LLAMA_SRC"
    mv "$EXTRACT_DIR/llama_cpp_python-${LLAMA_CPP_PY_VERSION}/vendor/llama.cpp" "$LLAMA_SRC"
    if [ ! -f "$LLAMA_SRC/include/llama.h" ]; then
        echo "!! vendored llama.cpp source is missing or incomplete" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. Configure and build the shared libraries with native CPU optimizations.
# ---------------------------------------------------------------------------
echo ">> configuring llama.cpp"
cmake -S "$LLAMA_SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER="$PREFIX/bin/clang" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DGGML_NATIVE=ON \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF >/dev/null

echo ">> building llama.cpp (this can take a while)"
cmake --build "$BUILD" -j "$JOBS" --target llama

# ---------------------------------------------------------------------------
# 3. Stage the libraries with version-less sonames so Android packages them.
# ---------------------------------------------------------------------------
echo ">> staging native libraries"
mkdir -p "$JNI_OUT"

copy_lib() {
    local base="$1"
    local real
    real="$(readlink -f "$BUILD/bin/$base")"
    if [ -z "$real" ] || [ ! -f "$real" ]; then
        echo "!! cannot find $base in $BUILD/bin" >&2
        exit 1
    fi
    cp -f "$real" "$JNI_OUT/$base"
    patchelf --set-soname "$base" "$JNI_OUT/$base"
}

copy_lib libllama.so
copy_lib libggml.so
copy_lib libggml-base.so
copy_lib libggml-cpu.so

for f in libllama.so libggml.so libggml-cpu.so; do
    patchelf --replace-needed libggml.so.0      libggml.so      "$JNI_OUT/$f" 2>/dev/null || true
    patchelf --replace-needed libggml-cpu.so.0  libggml-cpu.so  "$JNI_OUT/$f" 2>/dev/null || true
    patchelf --replace-needed libggml-base.so.0 libggml-base.so "$JNI_OUT/$f" 2>/dev/null || true
done

# The libc++ that matches the build is the one shipped with the toolchain.
cp -f "$PREFIX/lib/libc++_shared.so" "$JNI_OUT/libc++_shared.so"

# ---------------------------------------------------------------------------
# 4. Build the JNI bridge against the freshly built libraries.
# ---------------------------------------------------------------------------
echo ">> compiling libggufjni.so"
"$CXX" -shared -fPIC -O3 -std=c++17 -DNDEBUG \
    -fvisibility=hidden \
    -I"$LLAMA_SRC/include" \
    -I"$LLAMA_SRC/ggml/include" \
    "$SRC" -o "$JNI_OUT/libggufjni.so" \
    -L"$JNI_OUT" -lllama -lggml -lggml-base -llog

echo ">> done"
ls -la "$JNI_OUT"
