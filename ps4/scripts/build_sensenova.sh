#!/bin/bash
set -euo pipefail
export PATH="$HOME/opt/cmake-3.31.12-linux-x86_64/bin:$HOME/.local/bin:$PATH"
cd ~/llm/llama.cpp-sn
echo "== commit: $(git rev-parse HEAD)"
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_PREFIX_PATH="$HOME/opt/vkdev" \
  -DVulkan_INCLUDE_DIR="$HOME/opt/vkdev/include"
time ninja -C build -j6 llama-cli llama-server llama-bench test-backend-ops llama-quantize "$@" 2>&1
echo "== BUILD DONE"
