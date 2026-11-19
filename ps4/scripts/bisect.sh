#!/bin/bash
cd ~/llm/llama.cpp-sn
export GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456
OT='output\.weight=CPU,blk\.(0|1|2|3|52|53|54|55)\.ffn_(gate|up|down)\.weight=CPU'
run() {
  local name="$1"; shift
  local log=~/llm/bisect_$name.log
  env "$@" timeout 900 ./build/bin/llama-completion -m ~/llm/models/SenseNova-Loop-9B-Q4_K_M.gguf --no-host --no-op-offload \
    -ngl 99 -ot "$OT" -c 256 -b 32 -ub 32 -n 12 --temp 0 -no-cnv -t 8 -lm mmap \
    -p "The capital of France is" > $log 2>&1
  local rc=$?
  local out=$(grep -a -o "The capital of France is.*" $log | head -1 | cut -c1-120)
  local lost=$(grep -c "device lost" $log)
  echo "[$name] rc=$rc device_lost=$lost out=$out"
  grep -a -E "eval time|load time" $log | head -3
}
run nofusion   GGML_VK_DISABLE_FUSION=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1
run unfused_mhc LLAMA_SENSENOVA_MHC_FUSED=0
run fa_off     LLAMA_ARG_FLASH_ATTN=off
echo BISECT_DONE
