#!/bin/bash
# usage: mtp_bench.sh <model> <label> <cols_limit> [server args...]
MODEL=$1; LABEL=$2; COLS=$3; shift 3
cd ~/llm/llama.cpp-sn
export GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456 GGML_VK_MUL_MAT_VEC_MAX_COLS=$COLS
./build/bin/llama-server -m "$MODEL" --no-host -ngl 99 -c 2048 -t 8 --host 127.0.0.1 --port 8099 "$@" > ~/llm/server_$LABEL.log 2>&1 &
SP=$!
for i in $(seq 1 120); do sleep 3; curl -s localhost:8099/health | grep -q '"ok"' && break; if ! kill -0 $SP 2>/dev/null; then echo "[$LABEL] server died"; tail -5 ~/llm/server_$LABEL.log; exit 1; fi; done
for P in "Write a short story about a robot who learns to paint." "日本の首都について、歴史を交えて詳しく説明してください。"; do
  BODY=$(python3 -c 'import json,sys; print(json.dumps({"prompt": "<|im_start|>user\n"+sys.argv[1]+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "n_predict": 160, "temperature": 0, "cache_prompt": False}))' "$P")
  curl -s localhost:8099/completion -d "$BODY" | python3 -c '
import json,sys
d=json.load(sys.stdin); t=d.get("timings",{})
print("[%s] gen %.2f tok/s (%.1f ms/tok) n=%d draft=%s acc=%s | %s" % (sys.argv[1], t.get("predicted_per_second",0), t.get("predicted_per_token_ms",0), t.get("predicted_n",0), t.get("draft_n","-"), t.get("draft_n_accepted","-"), d.get("content","")[:70].replace("\n"," ")))' "$LABEL"
done
kill $SP; wait $SP 2>/dev/null
