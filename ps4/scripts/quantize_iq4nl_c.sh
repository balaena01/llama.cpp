#!/bin/bash
cd ~/llm/llama.cpp-sn
time ./build/bin/llama-quantize --imatrix ~/llm/imatrix-sn-loop.gguf \
  --output-tensor-type iq4_nl --token-embedding-type q4_K --tensor-type attn_v=q6_K --tensor-type attn_qkv_gate=iq4_nl \
  ~/llm/models/SenseNova-Loop-9B-BF16.gguf ~/llm/models/SenseNova-Loop-9B-IQ4_NL-ps4c.gguf IQ4_NL 8
echo "QUANT_EXIT=$?"
