#!/bin/bash
# usage: run_ppl.sh <model> <label>
cd ~/llm/llama.cpp-sn
export GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456
./build/bin/llama-perplexity -m "$1" -f ~/llm/wikitext-2-raw/wiki.test.raw --no-host -fa off -ngl 99 -c 512 -b 512 -ub 128 --chunks 24 -t 8 > ~/llm/ppl_$2.log 2>&1
echo "PPL_EXIT=$?" >> ~/llm/ppl_$2.log
