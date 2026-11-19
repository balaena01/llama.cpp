# SenseNova-Loop-9B on a PlayStation 4 (Vulkan)

This branch runs [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF) on the GPU of a base PS4 running Linux, with [OpenSenseNova/llama.cpp](https://github.com/OpenSenseNova/llama.cpp) (commit `24ea7bb`) plus the changes in this branch.

**Result: 9.2 tok/s generation (108 ms/token), all layers on the PS4 GPU**, with the same quality as the official Q4_K_M (wikitext-2 perplexity 14.84 vs 14.87).

Model: [Kujira/SenseNova-Loop-9B-IQ4_NL-PS4-GGUF](https://huggingface.co/Kujira/SenseNova-Loop-9B-IQ4_NL-PS4-GGUF)

日本語の詳しい作業記録は [WORKFLOW_ja.md](WORKFLOW_ja.md) にあります。

## Speed progression

| Step | ms/token | tok/s |
|---|---|---|
| Official Q4_K_M, unfused graph, some weights on CPU | 1142 | 0.88 |
| Fused mHC Sinkhorn into one op | 720 | 1.4 |
| GTT raised to 5.5 GiB, everything on the GPU | 612 | 1.6 |
| Weights in uncached GTT instead of spilled device-local memory | 200 | 5.0 |
| mHC router matmul back on the GPU (97 CPU/GPU splits to 2) | 145 | 6.9 |
| Re-quantized to IQ4_NL with imatrix (made on the PS4) | 118.7 | 8.4 |
| `-fa off` | 116.1 | 8.6 |
| Router fused into one kernel (rms_norm + matmul + bias + Sinkhorn) | 111 | 9.0 |
| One softplus per Sinkhorn pair, IQ4_NL output head | **108.4** | **9.2** |

## What the changes do

**ggml / Vulkan**

- `GGML_OP_MHC_ROUTE` (`ggml_mhc_route`, `ggml_mhc_route_ext`): the SenseNova two-stream mHC route as one op. The `_ext` form also folds in the router: `mul_mat(w, rms_norm(x)) + b`, then sigmoid and the 20-step log-domain Sinkhorn. CPU and Vulkan (`mhc_route.comp`, `mhc_route_fused.comp`). This replaces about 49,000 tiny graph nodes per token.
- `dsv4_hc_pre` / `dsv4_hc_post` Vulkan shaders take the stream count as a spec constant, so hc = 2 works (they were hardcoded to 4).
- Vulkan accepts `MUL_MAT` with an F32 src1 precision hint when the device has no fp16 and both inputs are F32 (nothing is converted down, so the hint holds). Before, the mHC router matmul fell back to the CPU.
- PS4 GPU detection (device name `Liverpool` / `Gladius`):
  - `mul_mat_vec` with several columns hangs the GPU when rows x columns per workgroup gets too large (for example K-quants at 5+ columns, Q5_K at 4). The column limit is set to 2 (`GGML_VK_MUL_MAT_VEC_MAX_COLS` to override).
  - `prefer_host_memory` is on: device-local buffers that spill past the 1 GiB VRAM carve-out were 4-6x slower than uncached GTT.
  - The VRAM carve-out is still filled first (`GGML_VK_VRAM_FIRST_MB`), so it isn't wasted.
  - IQ matvec rows per workgroup = 4 (`GGML_VK_DMMV_RM_IQ`).
- Tuning overrides for experiments: `GGML_VK_DMMV_RM_KQ`, `GGML_VK_DMMV_RM_STDQ`, `GGML_VK_DMMV_WG`.

**Model graph** (`src/models/sensenova-loop.cpp`)

- Both mHC streams live in one `[n_embd, 2, n_tokens]` tensor. The route is `ggml_mhc_route_ext`, and merges use `dsv4_hc_pre` / `dsv4_hc_post`.
- `LLAMA_SENSENOVA_MHC_FUSED=0` selects the original unfused graph for comparison. Outputs match (greedy decoding gave the same text, and perplexity is unchanged).

**Tests**

- `test-backend-ops`: `MHC_ROUTE`, `MHC_ROUTE_EXT`, hc = 2 cases for `DSV4_HC_PRE` / `DSV4_HC_POST`, and a cache-resident matvec perf probe.

## Hardware and software

| | |
|---|---|
| Console | PS4 (base, GPU PCI ID 1002:9923, Baikal southbridge) |
| GPU | AMD Liverpool, GCN2 / GFX7, 18 CU at 800 MHz, no fp16 math, no int8 dot |
| Memory | 8 GB GDDR5 shared. 1 GiB VRAM carve-out, about 6.9 GB visible to Linux |
| OS | CachyOS port, Strawberry kernel 7.3.3, Mesa 26.0.4 RADV |

## How to reproduce

1. **Give the GPU more memory.** The default GTT limit is 50% of RAM, which is too small. Add this to the kernel command line (for the usual loaders it's `/data/linux/boot/bootargs.txt` on the PS4, written over GoldHEN FTP):

   ```
   ttm.pages_limit=1441792
   ```

   That gives about 1 GiB VRAM + 5.5 GiB GTT = 6.5 GiB usable by Vulkan. Running headless (`systemctl set-default multi-user.target`) frees about 0.8 GB more.

2. **Build** (`scripts/build_sensenova.sh`): Vulkan on, `-DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF`. A full build takes about 20 minutes on the Jaguar CPU.

3. **Model**: the quantized file is on Hugging Face: [Kujira/SenseNova-Loop-9B-IQ4_NL-PS4-GGUF](https://huggingface.co/Kujira/SenseNova-Loop-9B-IQ4_NL-PS4-GGUF). To make it yourself, download the official BF16 GGUF and quantize on the PS4 (`scripts/run_imatrix.sh`, `scripts/quantize_iq4nl_c.sh`):

   ```sh
   llama-quantize --imatrix imatrix-sn-loop.gguf \
     --output-tensor-type iq4_nl --token-embedding-type q4_K \
     --tensor-type attn_v=q6_K --tensor-type attn_qkv_gate=iq4_nl \
     SenseNova-Loop-9B-BF16.gguf SenseNova-Loop-9B-IQ4_NL-ps4c.gguf IQ4_NL 8
   ```

   Don't use Q5_K tensors on this GPU, because they are slow and can hang. Run llama-quantize alone, since it needs about 4 GB of RAM for the output head.

4. **Run**:

   ```sh
   GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456 \
   ./build/bin/llama-server -m SenseNova-Loop-9B-IQ4_NL-ps4c.gguf \
     --no-host -ngl 99 -fa off -c 4096 -np 1 -b 512 -ub 256 \
     --cache-ram 0 --ctx-checkpoints 2 -t 8 --host 0.0.0.0 --port 8080 --path ps4/demo
   ```

   - `--no-host` keeps CPU-side weights out of pinned memory, which comes from the same GTT pool.
   - `--cache-ram 0 --ctx-checkpoints 2` stop the server from growing host memory across chats. Without them the PS4 runs out of RAM after a few requests.
   - `scripts/llama-server.service` is a systemd user unit (`loginctl enable-linger` to start at boot).
   - `demo/index.html` is a small chat page with a live tok/s readout. Open `http://<ps4>:8080/`.

## What didn't work

- **MTP speculative decoding** (`--spec-type draft-mtp`): about 50% acceptance with the current draft head, and one verify step costs about 1.9x a normal token, so it is slower (8.8 to 6.3 tok/s).
- **Hand-written IQ4_NL matvec shaders**: four versions (pair LUT, prefetch, wide strided loads, LDS staging) all lost to the generic shader (402 us for 4096x14336). It isn't bandwidth bound (a cache-resident matrix is no faster), and the real limit needs a profiler to find. The last attempt is in `experiments/`.
- 10 tok/s: the matvec is about 70% of a token, and that is where the rest would have to come from.

## Status

- Tested on one base PS4 only. PS4 Pro (Gladius) is detected by name but untested.
- This is a record of an experiment on top of a pinned, experimental fork. It is not maintained and not intended for upstream.
- Written together with an AI assistant (Claude). Every change was built and tested on the PS4, and `test-backend-ops` compares the new Vulkan ops against the CPU.

## License

Same as llama.cpp (MIT). The model is Apache-2.0.
