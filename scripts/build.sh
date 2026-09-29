#!/usr/bin/env bash
# Build the pinned Prism llama.cpp runtime and Shingi's native readout into $SHINGI_HOME.
# Skips all work when the readout was already built from the same revision, architectures and source.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHINGI_HOME="${SHINGI_HOME:-$HOME/.cache/shingi-27b}"
PRISM_URL=https://github.com/PrismML-Eng/llama.cpp.git
PRISM_REVISION=d8f26eec76da6d09bb708bcba51ef64b8cd868a3
ARCHITECTURES="${SHINGI_CUDA_ARCHITECTURES:-86;89;120;121}"
PRISM="$SHINGI_HOME/prism"
READOUT="$SHINGI_HOME/bin/readout"
STAMP="$READOUT.stamp"

stamp="$PRISM_REVISION $ARCHITECTURES $(sha256sum "$ROOT/src/native/readout.cpp" | cut -d' ' -f1)"
if [ -x "$READOUT" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$stamp" ]; then
    echo "shingi-27b: runtime already built at $PRISM_REVISION"
    exit 0
fi

echo "shingi-27b: building the Prism runtime at $PRISM_REVISION for CUDA architectures $ARCHITECTURES"
echo "shingi-27b: the first build compiles CUDA kernels and can take a while"
if [ ! -e "$PRISM" ]; then
    mkdir -p "$SHINGI_HOME"
    git clone --quiet --no-checkout "$PRISM_URL" "$PRISM"
fi
if [ -n "$(git -C "$PRISM" status --porcelain --untracked-files=no)" ]; then
    echo "shingi-27b: $PRISM has local changes; move it away and run again" >&2
    exit 1
fi
if [ "$(git -C "$PRISM" rev-parse HEAD 2>/dev/null)" != "$PRISM_REVISION" ]; then
    git -C "$PRISM" fetch --quiet origin "$PRISM_REVISION"
    git -C "$PRISM" checkout --quiet --detach "$PRISM_REVISION"
fi

jobs="$(nproc)"
[ "$jobs" -le 8 ] || jobs=8  # CUDA kernel compilation needs several GB of RAM per job.
cmake -S "$PRISM" -B "$PRISM/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHITECTURES" -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$PRISM/build" --target llama --parallel "$jobs"

mkdir -p "$SHINGI_HOME/bin"
c++ -std=c++17 -O2 -Wall -Wextra "$ROOT/src/native/readout.cpp" \
    -I"$PRISM/include" -I"$PRISM/ggml/include" -I"$PRISM/vendor" \
    -L"$PRISM/build/bin" -Wl,-rpath,"$PRISM/build/bin" \
    -lllama -lggml -lggml-base -o "$READOUT"
printf '%s\n' "$stamp" > "$STAMP"
echo "shingi-27b: built $READOUT"
