#!/bin/bash
cd ~/llm/llama.cpp-sn
export GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456
./build/bin/llama-imatrix -m ~/llm/models/SenseNova-Loop-9B-Q4_K_M.gguf -f ~/llm/calib_small.txt \
  -o ~/llm/imatrix-sn-loop.gguf -ngl 99 --no-host -c 512 -b 512 -ub 128 --chunks 100 -t 8 -ofreq 10
echo "IMATRIX_EXIT=$?"
