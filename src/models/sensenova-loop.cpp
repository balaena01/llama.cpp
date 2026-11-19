#include "models.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

void llama_model_sensenova_loop::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, true);
    llama_model_qwen35::load_arch_hparams(ml);
    ml.get_key(LLM_KV_LOOP_START, hparams.loop_start);
    ml.get_key(LLM_KV_LOOP_END,   hparams.loop_end);
    ml.get_key(LLM_KV_LOOP_COUNT, hparams.loop_count);
    ml.get_key(LLM_KV_MHC_START, hparams.mhc_start);
    ml.get_key(LLM_KV_MHC_END, hparams.mhc_end);
    ml.get_key(LLM_KV_MHC_STREAMS, hparams.mhc_streams);
    ml.get_key(LLM_KV_MHC_ITERATIONS, hparams.mhc_iterations);
    ml.get_key(LLM_KV_MHC_PRE_SCALE, hparams.mhc_pre_scale);
    ml.get_key(LLM_KV_MHC_FIXED_IDENTITY, hparams.mhc_fixed_identity);

    const uint32_t n_layer = hparams.n_layer();
    if (hparams.n_layer_nextn > 1 || hparams.loop_count == 0 || hparams.loop_start >= hparams.loop_end || hparams.loop_end > n_layer) {
        throw std::runtime_error("invalid SenseNova Loop layer range or count");
    }
    const uint64_t extra = uint64_t(hparams.loop_count - 1) * (hparams.loop_end - hparams.loop_start);
    if (extra > n_layer - hparams.loop_end) {
        throw std::runtime_error("SenseNova Loop execution plan exceeds the block count");
    }
    const uint32_t n_physical = n_layer - extra;
    if (hparams.n_layer_nextn && hparams.is_recr(n_layer)) {
        throw std::runtime_error("SenseNova Loop MTP requires a full-attention layer");
    }
    if (hparams.mhc_start >= hparams.mhc_end || hparams.mhc_end > n_physical || hparams.mhc_streams != 2 ||
            hparams.mhc_iterations == 0 || hparams.mhc_iterations > 20 || !std::isfinite(hparams.mhc_pre_scale) || hparams.mhc_pre_scale <= 0.0f) {
        throw std::runtime_error("unsupported SenseNova mHC configuration");
    }
    if (!((hparams.mhc_end <= hparams.loop_start) ||
          (hparams.mhc_start >= hparams.loop_start && hparams.mhc_end <= hparams.loop_end) ||
          (hparams.mhc_start >= hparams.loop_end))) {
        throw std::runtime_error("SenseNova mHC range crosses a Loop boundary");
    }

    layer_tensor_sources.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t physical = physical_layer(il);
        layer_tensor_sources[il] = physical < hparams.loop_end ? physical : il;
        if (hparams.is_recr(il) != hparams.is_recr(layer_tensor_sources[il])) {
            throw std::runtime_error("SenseNova Loop visits have different attention types");
        }
    }
    type = n_physical == 32 ? LLM_TYPE_9B : LLM_TYPE_UNKNOWN;
}

uint32_t llama_model_sensenova_loop::physical_layer(uint32_t il) const {
    const uint32_t width = hparams.loop_end - hparams.loop_start;
    const uint32_t extra = (hparams.loop_count - 1) * width;
    if (il < hparams.loop_start) {
        return il;
    }
    if (il < hparams.loop_end + extra) {
        return hparams.loop_start + (il - hparams.loop_start) % width;
    }
    return il - extra;
}

void llama_model_sensenova_loop::load_arch_tensors(llama_model_loader & ml) {
    // Per-tensor scales are loaded after the shared layer aliases.
    for (const auto & item : ml.weights_map) {
        const auto type = item.second.tensor->type;
        if (type == GGML_TYPE_F8_E4M3 || type == GGML_TYPE_NVFP4) {
            throw std::runtime_error("SenseNova Loop does not support preserved FP8 or NVFP4 weights");
        }
    }
    llama_model_qwen35::load_arch_tensors(ml);
    LLAMA_LOAD_LOCALS;
    const int flags = nextn_flags(ml).trunk;
    for (int il = 0; il < n_layer; ++il) {
        const uint32_t physical = physical_layer(il);
        if (physical < hparams.mhc_start || physical >= hparams.mhc_end) {
            continue;
        }
        if (layer_tensor_sources[il] != (uint32_t) il) {
            layers[il] = layers[layer_tensor_sources[il]];
            continue;
        }
        auto & layer = layers[il];
        const int64_t n_route = hparams.mhc_fixed_identity ? 4 : 8;
        layer.hc_attn_fn   = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", il), {2*n_embd, n_route}, flags);
        layer.hc_attn_fn_b = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "bias",   il), {n_route}, flags);
        layer.hc_ffn_fn    = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,  "weight", il), {2*n_embd, n_route}, flags);
        layer.hc_ffn_fn_b  = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,  "bias",   il), {n_route}, flags);
        for (const auto * router : {layer.hc_attn_fn, layer.hc_attn_fn_b, layer.hc_ffn_fn, layer.hc_ffn_fn_b}) {
            if (router && router->type != GGML_TYPE_F32) {
                throw std::runtime_error("SenseNova mHC router tensors must be F32");
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_sensenova_loop::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<llama_model_qwen35::graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_sensenova_loop::graph::routing llama_model_sensenova_loop::graph::build_route(
        ggml_tensor * a, ggml_tensor * b, ggml_tensor * weight, ggml_tensor * bias, int il) {
    ggml_tensor * flat = ggml_concat(ctx0, a, b, 0);
    flat = ggml_rms_norm(ctx0, flat, 1e-6f);
    ggml_tensor * logits = build_lora_mm(weight, flat);
    if (logits->op == GGML_OP_MUL_MAT) {
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        ggml_prec_set_src(logits, GGML_PREC_F32, 1);
    }
    logits = ggml_add(ctx0, logits, bias);
    auto row = [&](int i) {
        return ggml_view_2d(ctx0, logits, 1, a->ne[1], logits->nb[1], i*logits->nb[0]);
    };

    routing result{};
    result.input = ggml_add(ctx0, ggml_mul(ctx0, a, ggml_sigmoid(ctx0, row(0))), ggml_mul(ctx0, b, ggml_sigmoid(ctx0, row(1))));
    result.input = ggml_scale(ctx0, result.input, hparams.mhc_pre_scale);
    result.post[0] = ggml_scale(ctx0, ggml_sigmoid(ctx0, row(2)), 2.0f);
    result.post[1] = ggml_scale(ctx0, ggml_sigmoid(ctx0, row(3)), 2.0f);
    if (!hparams.mhc_fixed_identity) {
        for (int i = 0; i < 4; ++i) {
            result.mix[i] = row(4 + i);
        }
        auto normalize = [&](ggml_tensor * & x, ggml_tensor * & y) {
            // Two-element log-softmax, without exponentiating the unnormalized logits.
            ggml_tensor * delta = ggml_sub(ctx0, x, y);
            x = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, delta)));
            y = ggml_neg(ctx0, ggml_softplus(ctx0, delta));
        };
        for (uint32_t i = 0; i < hparams.mhc_iterations; ++i) {
            normalize(result.mix[0], result.mix[1]);
            normalize(result.mix[2], result.mix[3]);
            normalize(result.mix[0], result.mix[2]);
            normalize(result.mix[1], result.mix[3]);
        }
        for (auto & mix : result.mix) {
            mix = ggml_exp(ctx0, mix);
        }
    }
    cb(result.input, "mhc_input", il);
    return result;
}

void llama_model_sensenova_loop::graph::build_merge(
        ggml_tensor * & a, ggml_tensor * & b, ggml_tensor * branch, const routing & route) {
    ggml_tensor * next_a = a;
    ggml_tensor * next_b = b;
    if (!hparams.mhc_fixed_identity) {
        next_a = ggml_add(ctx0, ggml_mul(ctx0, a, route.mix[0]), ggml_mul(ctx0, b, route.mix[1]));
        next_b = ggml_add(ctx0, ggml_mul(ctx0, a, route.mix[2]), ggml_mul(ctx0, b, route.mix[3]));
    }
    a = ggml_add(ctx0, next_a, ggml_mul(ctx0, branch, route.post[0]));
    b = ggml_add(ctx0, next_b, ggml_mul(ctx0, branch, route.post[1]));
}

// LLAMA_SENSENOVA_MHC_FUSED=0 selects the unfused mHC graph
static bool sensenova_mhc_fused() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_SENSENOVA_MHC_FUSED");
        return env == nullptr || std::string(env) != "0";
    }();
    return enabled;
}

ggml_tensor * llama_model_sensenova_loop::graph::build_route_fused(
        ggml_tensor * hc, ggml_tensor * weight, ggml_tensor * bias, int il) {
    // same memory layout as concat(a, b, 0)
    ggml_tensor * flat = ggml_reshape_2d(ctx0, hc, 2*hc->ne[0], hc->ne[2]);
    if (loras->empty() && build_hadamard_input(weight, flat) == flat) {
        // rms_norm, router matmul, bias and route in one op
        ggml_tensor * route = ggml_mhc_route_ext(ctx0, flat, weight, bias, 1e-6f, 2.0f, (int32_t) hparams.mhc_iterations);
        cb(route, "mhc_route", il);
        return route;
    }
    flat = ggml_rms_norm(ctx0, flat, 1e-6f);
    ggml_tensor * logits = build_lora_mm(weight, flat);
    if (logits->op == GGML_OP_MUL_MAT) {
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        ggml_prec_set_src(logits, GGML_PREC_F32, 1);
    }
    logits = ggml_add(ctx0, logits, bias);
    ggml_tensor * route = ggml_mhc_route(ctx0, logits, 2.0f, (int32_t) hparams.mhc_iterations);
    cb(route, "mhc_route", il);
    return route;
}

ggml_tensor * llama_model_sensenova_loop::graph::build_mhc_input(ggml_tensor * hc, ggml_tensor * route, int il) {
    ggml_tensor * pre = ggml_view_2d(ctx0, route, 2, route->ne[1], route->nb[1], 0);
    ggml_tensor * input = ggml_dsv4_hc_pre(ctx0, hc, pre);
    if (hparams.mhc_pre_scale != 1.0f) {
        input = ggml_scale(ctx0, input, hparams.mhc_pre_scale);
    }
    cb(input, "mhc_input", il);
    return input;
}

ggml_tensor * llama_model_sensenova_loop::graph::build_mhc_merge(ggml_tensor * hc, ggml_tensor * branch, ggml_tensor * route) {
    ggml_tensor * post = ggml_view_2d(ctx0, route, 2, route->ne[1], route->nb[1], 2*route->nb[0]);
    ggml_tensor * comb = ggml_view_3d(ctx0, route, 2, 2, route->ne[1], 2*route->nb[0], route->nb[1], 4*route->nb[0]);
    return ggml_dsv4_hc_post(ctx0, branch, hc, post, comb);
}

llama_model_sensenova_loop::graph::graph(const llm_graph_params & params) : graph_base(params) {
    const auto & loop = static_cast<const llama_model_sensenova_loop &>(model);
    auto * inp = build_inp_mem_hybrid();
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * hc = nullptr;
    const bool fused = !hparams.mhc_fixed_identity && sensenova_mhc_fused();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const uint32_t physical = loop.physical_layer(il);
        const bool mhc = physical >= hparams.mhc_start && physical < hparams.mhc_end;
        res->t_layer_inp[il] = cur;
        if (mhc && fused) {
            if (physical == hparams.mhc_start) {
                ggml_tensor * c = ggml_reshape_3d(ctx0, cur, n_embd, 1, cur->ne[1]);
                hc = ggml_concat(ctx0, c, c, 1);
            }
            ggml_tensor * route = build_route_fused(hc, layer.hc_attn_fn, layer.hc_attn_fn_b, il);
            cur = build_norm(build_mhc_input(hc, route, il), layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            if (hparams.is_recr(il)) {
                cur = build_layer_attn_linear(inp->get_recr(), cur, il);
            } else {
                cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
            }
            if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                route = ggml_get_rows(ctx0, route, inp_out_ids);
                hc    = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, hc, 2*n_embd, hc->ne[2]), inp_out_ids);
                hc    = ggml_reshape_3d(ctx0, hc, n_embd, 2, hc->ne[1]);
            }
            hc = build_mhc_merge(hc, cur, route);
            route = build_route_fused(hc, layer.hc_ffn_fn, layer.hc_ffn_fn_b, il);
            cur = build_norm(build_mhc_input(hc, route, il), layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cur = build_layer_ffn(cur, il);
            hc = build_mhc_merge(hc, cur, route);

            ggml_tensor * s0 = ggml_view_2d(ctx0, hc, n_embd, hc->ne[2], hc->nb[2], 0);
            ggml_tensor * s1 = ggml_view_2d(ctx0, hc, n_embd, hc->ne[2], hc->nb[2], hc->nb[1]);
            ggml_tensor * v0 = build_cvec(s0, il);
            ggml_tensor * v1 = build_cvec(s1, il);
            if (v0 != s0 || v1 != s1) {
                hc = ggml_concat(ctx0,
                        ggml_reshape_3d(ctx0, ggml_cont(ctx0, v0), n_embd, 1, v0->ne[1]),
                        ggml_reshape_3d(ctx0, ggml_cont(ctx0, v1), n_embd, 1, v1->ne[1]), 1);
            }
            cur = ggml_scale(ctx0, ggml_add(ctx0, v0, v1), 0.5f);
            cb(cur, "l_out", il);
            continue;
        }
        if (mhc && physical == hparams.mhc_start) {
            a = cur;
            b = cur;
        }
        routing attn_route{};
        ggml_tensor * residual = cur;
        if (mhc) {
            attn_route = build_route(a, b, layer.hc_attn_fn, layer.hc_attn_fn_b, il);
            cur = attn_route.input;
        }
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }
        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            if (mhc) {
                a = ggml_get_rows(ctx0, a, inp_out_ids);
                b = ggml_get_rows(ctx0, b, inp_out_ids);
                for (auto & post : attn_route.post) { post = ggml_get_rows(ctx0, post, inp_out_ids); }
                if (!hparams.mhc_fixed_identity) {
                    for (auto & mix : attn_route.mix) { mix = ggml_get_rows(ctx0, mix, inp_out_ids); }
                }
            }
        }
        if (mhc) {
            build_merge(a, b, cur, attn_route);
            const routing ffn_route = build_route(a, b, layer.hc_ffn_fn, layer.hc_ffn_fn_b, il);
            cur = build_norm(ffn_route.input, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cur = build_layer_ffn(cur, il);
            build_merge(a, b, cur, ffn_route);
            a = build_cvec(a, il);
            b = build_cvec(b, il);
            cur = ggml_scale(ctx0, ggml_add(ctx0, a, b), 0.5f);
        } else {
            residual = ggml_add(ctx0, cur, residual);
            cur = build_norm(residual, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cur = ggml_add(ctx0, build_layer_ffn(cur, il), residual);
            cur = build_cvec(cur, il);
        }
        cb(cur, "l_out", il);
    }
    build_output(cur, inp_out_ids);
}
