#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <torch/library.h>
#include <limits>
#include <cmath>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace {

struct MetalRuntime {
    id<MTLComputePipelineState> pipeline_generic = nil;
    id<MTLComputePipelineState> pipeline_generic_block_written = nil;
    id<MTLComputePipelineState> pipeline_generic_bf16 = nil;
    id<MTLComputePipelineState> pipeline_dh64_bs4_single = nil;
    id<MTLComputePipelineState> pipeline_dh64_bs4_gqa1_single = nil;
    id<MTLComputePipelineState> pipeline_dh64_bs4_gqa1_block_written = nil;
    id<MTLComputePipelineState> pipeline_dh64_bs4_gqa2_single = nil;
    id<MTLComputePipelineState> pipeline_dh64_bs4_gqa2_dense = nil;
    id<MTLComputePipelineState> pipeline_dh64_bs4_gqa2_dualhead = nil;
    uint32_t thread_execution_width_generic = 32;
    uint32_t thread_execution_width_generic_block_written = 32;
    uint32_t thread_execution_width_generic_bf16 = 32;
    uint32_t thread_execution_width_dh64_bs4_single = 32;
    uint32_t thread_execution_width_dh64_bs4_gqa1_single = 32;
    uint32_t thread_execution_width_dh64_bs4_gqa1_block_written = 32;
    uint32_t thread_execution_width_dh64_bs4_gqa2_single = 32;
    uint32_t thread_execution_width_dh64_bs4_gqa2_dense = 32;
    uint32_t thread_execution_width_dh64_bs4_gqa2_dualhead = 32;
    bool init_ok = false;
};

static inline id<MTLBuffer> tensor_mtl_buffer(const at::Tensor& t) {
    return (id<MTLBuffer>)t.storage().data();
}

MetalRuntime& get_metal_runtime() {
    static MetalRuntime rt;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            NSString* this_file = [NSString stringWithUTF8String:__FILE__];
            NSString* src_dir = [this_file stringByDeletingLastPathComponent];
            NSString* kernel_path = [src_dir stringByAppendingPathComponent:@"metal_flex_attn.metal"];

            NSError* read_error = nil;
            NSString* source = [NSString stringWithContentsOfFile:kernel_path
                                                         encoding:NSUTF8StringEncoding
                                                            error:&read_error];
            if (!source) {
                return;
            }

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return;
            }

            NSError* compile_error = nil;
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = YES;
            id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&compile_error];
            if (!lib) {
                return;
            }

            id<MTLFunction> fn_generic = [lib newFunctionWithName:@"metal_flex_attn_forward"];
            id<MTLFunction> fn_generic_block_written = [lib newFunctionWithName:@"metal_flex_attn_forward_from_block_written"];
            id<MTLFunction> fn_generic_bf16 = [lib newFunctionWithName:@"metal_flex_attn_forward_bf16"];
            id<MTLFunction> fn_dh64_bs4_single = [lib newFunctionWithName:@"metal_flex_attn_forward_dh64_bs4_single"];
            id<MTLFunction> fn_dh64_bs4_gqa1_single = [lib newFunctionWithName:@"metal_flex_attn_forward_dh64_bs4_gqa1_single"];
            id<MTLFunction> fn_dh64_bs4_gqa1_block_written = [lib newFunctionWithName:@"metal_flex_attn_forward_dh64_bs4_gqa1_block_written"];
            id<MTLFunction> fn_dh64_bs4_gqa2_single = [lib newFunctionWithName:@"metal_flex_attn_forward_dh64_bs4_gqa2_single"];
            id<MTLFunction> fn_dh64_bs4_gqa2_dense = [lib newFunctionWithName:@"metal_flex_attn_forward_dh64_bs4_gqa2_dense"];
            id<MTLFunction> fn_dh64_bs4_gqa2_dualhead = [lib newFunctionWithName:@"metal_flex_attn_forward_dh64_bs4_gqa2_dualhead"];
            if (!fn_generic || !fn_dh64_bs4_single || !fn_dh64_bs4_gqa1_single || !fn_dh64_bs4_gqa1_block_written || !fn_dh64_bs4_gqa2_single || !fn_dh64_bs4_gqa2_dense || !fn_dh64_bs4_gqa2_dualhead) {
                return;
            }

            NSError* pipe_error = nil;
            rt.pipeline_generic = [device newComputePipelineStateWithFunction:fn_generic error:&pipe_error];
            if (!rt.pipeline_generic) {
                return;
            }
            rt.pipeline_generic_block_written = [device newComputePipelineStateWithFunction:fn_generic_block_written error:&pipe_error];
            if (!rt.pipeline_generic_block_written) {
                return;
            }
            if (fn_generic_bf16) {
                rt.pipeline_generic_bf16 = [device newComputePipelineStateWithFunction:fn_generic_bf16 error:&pipe_error];
                if (rt.pipeline_generic_bf16) {
                    rt.thread_execution_width_generic_bf16 = static_cast<uint32_t>(rt.pipeline_generic_bf16.threadExecutionWidth);
                }
            }
            rt.pipeline_dh64_bs4_single = [device newComputePipelineStateWithFunction:fn_dh64_bs4_single error:&pipe_error];
            if (!rt.pipeline_dh64_bs4_single) {
                return;
            }
            rt.pipeline_dh64_bs4_gqa1_single = [device newComputePipelineStateWithFunction:fn_dh64_bs4_gqa1_single error:&pipe_error];
            if (!rt.pipeline_dh64_bs4_gqa1_single) {
                return;
            }
            rt.pipeline_dh64_bs4_gqa1_block_written = [device newComputePipelineStateWithFunction:fn_dh64_bs4_gqa1_block_written error:&pipe_error];
            if (!rt.pipeline_dh64_bs4_gqa1_block_written) {
                return;
            }
            rt.pipeline_dh64_bs4_gqa2_single = [device newComputePipelineStateWithFunction:fn_dh64_bs4_gqa2_single error:&pipe_error];
            if (!rt.pipeline_dh64_bs4_gqa2_single) {
                return;
            }
            rt.pipeline_dh64_bs4_gqa2_dense = [device newComputePipelineStateWithFunction:fn_dh64_bs4_gqa2_dense error:&pipe_error];
            if (!rt.pipeline_dh64_bs4_gqa2_dense) {
                return;
            }
            rt.pipeline_dh64_bs4_gqa2_dualhead = [device newComputePipelineStateWithFunction:fn_dh64_bs4_gqa2_dualhead error:&pipe_error];
            if (!rt.pipeline_dh64_bs4_gqa2_dualhead) {
                return;
            }
            rt.thread_execution_width_generic = static_cast<uint32_t>(rt.pipeline_generic.threadExecutionWidth);
            rt.thread_execution_width_generic_block_written = static_cast<uint32_t>(rt.pipeline_generic_block_written.threadExecutionWidth);
            rt.thread_execution_width_dh64_bs4_single = static_cast<uint32_t>(rt.pipeline_dh64_bs4_single.threadExecutionWidth);
            rt.thread_execution_width_dh64_bs4_gqa1_single = static_cast<uint32_t>(rt.pipeline_dh64_bs4_gqa1_single.threadExecutionWidth);
            rt.thread_execution_width_dh64_bs4_gqa1_block_written = static_cast<uint32_t>(rt.pipeline_dh64_bs4_gqa1_block_written.threadExecutionWidth);
            rt.thread_execution_width_dh64_bs4_gqa2_single = static_cast<uint32_t>(rt.pipeline_dh64_bs4_gqa2_single.threadExecutionWidth);
            rt.thread_execution_width_dh64_bs4_gqa2_dense = static_cast<uint32_t>(rt.pipeline_dh64_bs4_gqa2_dense.threadExecutionWidth);
            rt.thread_execution_width_dh64_bs4_gqa2_dualhead = static_cast<uint32_t>(rt.pipeline_dh64_bs4_gqa2_dualhead.threadExecutionWidth);
            rt.init_ok = true;
        }
    });
    return rt;
}

static int64_t get_block_size() {
    const char* env = std::getenv("WORLD_METAL_BLOCK_SIZE");
    if (!env) {
        return 4;
    }
    const long parsed = std::strtol(env, nullptr, 10);
    return parsed > 0 ? static_cast<int64_t>(parsed) : 4;
}

static bool fast_no_fallback() {
    const char* env = std::getenv("WORLD_METAL_FAST_NO_FALLBACK");
    if (!env) {
        return false;
    }
    return std::string(env) == "1";
}

static uint32_t get_tg_size() {
    const char* env = std::getenv("WORLD_METAL_TG_SIZE");
    if (!env) {
        return 256;
    }
    const long parsed = std::strtol(env, nullptr, 10);
    return parsed > 0 ? static_cast<uint32_t>(parsed) : 256;
}

static bool enable_gqa2_dualhead_specialization() {
    const char* env = std::getenv("WORLD_METAL_ENABLE_GQA2_DUALHEAD");
    if (!env) {
        return true;
    }
    const std::string s(env);
    if (s == "1") {
        return true;
    }
    if (s == "0") {
        return false;
    }
    return true;
}

static bool enable_fp16_accum() {
    const char* env = std::getenv("WORLD_METAL_FP16_ACCUM");
    if (!env) {
        return true;
    }
    return std::string(env) == "1";
}

static bool prefer_active_dispatch_path() {
    const char* env = std::getenv("WORLD_METAL_PREFER_ACTIVE_DISPATCH");
    if (!env) {
        return true;
    }
    return std::string(env) == "1";
}

static void dispatch_fast_kernel(
    id<MTLComputePipelineState> pipeline,
    uint32_t thread_execution_width,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& active_blocks,
    at::Tensor& out,
    uint32_t B,
    uint32_t Hq,
    uint32_t T,
    uint32_t L,
    uint32_t Dh,
    uint32_t BlockSize,
    uint32_t ActiveCount,
    uint32_t Causal,
    uint32_t Hkv,
    uint32_t FP16Accum,
    const char* err_prefix,
    uint32_t tg_size_hint
) {
    auto* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, err_prefix, ": no active MPS stream");

    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)stream->commandBuffer();
    TORCH_CHECK(cb != nil, err_prefix, ": failed to acquire command buffer");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    TORCH_CHECK(enc != nil, err_prefix, ": failed to create command encoder");
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:tensor_mtl_buffer(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [enc setBuffer:tensor_mtl_buffer(k) offset:k.storage_offset() * k.element_size() atIndex:1];
    [enc setBuffer:tensor_mtl_buffer(v) offset:v.storage_offset() * v.element_size() atIndex:2];
    [enc setBuffer:tensor_mtl_buffer(active_blocks)
         offset:active_blocks.storage_offset() * active_blocks.element_size()
        atIndex:3];
    [enc setBuffer:tensor_mtl_buffer(out) offset:out.storage_offset() * out.element_size() atIndex:4];
    [enc setBytes:&B length:sizeof(B) atIndex:5];
    [enc setBytes:&Hq length:sizeof(Hq) atIndex:6];
    [enc setBytes:&T length:sizeof(T) atIndex:7];
    [enc setBytes:&L length:sizeof(L) atIndex:8];
    [enc setBytes:&Dh length:sizeof(Dh) atIndex:9];
    [enc setBytes:&BlockSize length:sizeof(BlockSize) atIndex:10];
    [enc setBytes:&ActiveCount length:sizeof(ActiveCount) atIndex:11];
    [enc setBytes:&Causal length:sizeof(Causal) atIndex:12];
    [enc setBytes:&Hkv length:sizeof(Hkv) atIndex:13];
    [enc setBytes:&FP16Accum length:sizeof(FP16Accum) atIndex:14];

    const uint32_t simd_width = std::max<uint32_t>(1u, thread_execution_width);
    const NSUInteger total = static_cast<NSUInteger>(B) * Hq * T * simd_width;
    const NSUInteger tg_req = static_cast<NSUInteger>(tg_size_hint > 0u ? tg_size_hint : get_tg_size());
    const NSUInteger tg_aligned = MAX(simd_width, (tg_req / simd_width) * simd_width);
    const NSUInteger tg = MIN(pipeline.maxTotalThreadsPerThreadgroup, tg_aligned);
    const NSUInteger tg_count = (total + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(tg_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_fast_kernel_dh64_bs4_gqa2_dualhead(
    id<MTLComputePipelineState> pipeline,
    uint32_t thread_execution_width,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& active_blocks,
    at::Tensor& out,
    uint32_t B,
    uint32_t Hq,
    uint32_t T,
    uint32_t L,
    uint32_t Dh,
    uint32_t BlockSize,
    uint32_t ActiveCount,
    uint32_t Causal,
    uint32_t Hkv,
    uint32_t FP16Accum,
    const char* err_prefix,
    uint32_t tg_size_hint
) {
    auto* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, err_prefix, ": no active MPS stream");

    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)stream->commandBuffer();
    TORCH_CHECK(cb != nil, err_prefix, ": failed to acquire command buffer");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    TORCH_CHECK(enc != nil, err_prefix, ": failed to create command encoder");
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:tensor_mtl_buffer(q) offset:q.storage_offset() * q.element_size() atIndex:0];
    [enc setBuffer:tensor_mtl_buffer(k) offset:k.storage_offset() * k.element_size() atIndex:1];
    [enc setBuffer:tensor_mtl_buffer(v) offset:v.storage_offset() * v.element_size() atIndex:2];
    [enc setBuffer:tensor_mtl_buffer(active_blocks)
         offset:active_blocks.storage_offset() * active_blocks.element_size()
        atIndex:3];
    [enc setBuffer:tensor_mtl_buffer(out) offset:out.storage_offset() * out.element_size() atIndex:4];
    [enc setBytes:&B length:sizeof(B) atIndex:5];
    [enc setBytes:&Hq length:sizeof(Hq) atIndex:6];
    [enc setBytes:&T length:sizeof(T) atIndex:7];
    [enc setBytes:&L length:sizeof(L) atIndex:8];
    [enc setBytes:&Dh length:sizeof(Dh) atIndex:9];
    [enc setBytes:&BlockSize length:sizeof(BlockSize) atIndex:10];
    [enc setBytes:&ActiveCount length:sizeof(ActiveCount) atIndex:11];
    [enc setBytes:&Causal length:sizeof(Causal) atIndex:12];
    [enc setBytes:&Hkv length:sizeof(Hkv) atIndex:13];
    [enc setBytes:&FP16Accum length:sizeof(FP16Accum) atIndex:14];

    const uint32_t simd_width = std::max<uint32_t>(1u, thread_execution_width);
    const NSUInteger total = static_cast<NSUInteger>(B) * Hkv * T * simd_width;
    const NSUInteger tg_req = static_cast<NSUInteger>(tg_size_hint > 0u ? tg_size_hint : get_tg_size());
    const NSUInteger tg_aligned = MAX(simd_width, (tg_req / simd_width) * simd_width);
    const NSUInteger tg = MIN(pipeline.maxTotalThreadsPerThreadgroup, tg_aligned);
    const NSUInteger tg_count = (total + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(tg_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

at::Tensor metal_flex_attn_ref_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    bool causal
) {
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "flex_attn_metal expects q/k/v on MPS");
    TORCH_CHECK(
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)
            && q.scalar_type() == k.scalar_type()
            && q.scalar_type() == v.scalar_type(),
        "flex_attn_metal currently supports float16 or bfloat16 (matching dtypes)"
    );
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "flex_attn_metal expects contiguous q/k/v");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must match");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
                "q/k must match on batch and head dim");
    TORCH_CHECK(q.size(1) >= k.size(1), "q heads must be >= kv heads");
    TORCH_CHECK((q.size(1) % k.size(1)) == 0, "q heads must be divisible by kv heads for GQA");

    // Phase-1 native implementation: route through known-good ATen math while
    // ensuring we execute on the current MPS stream. This validates stream
    // integration before re-introducing raw Metal buffer bindings.
    auto* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, "flex_attn_metal: no active MPS stream");
    (void)stream->commandBuffer();

    const int64_t T = q.size(2);
    const int64_t L = k.size(2);
    const int64_t Dh = q.size(3);

    at::Tensor mask_tensor;
    if (mask.has_value()) {
        mask_tensor = *mask;
        TORCH_CHECK(mask_tensor.device().is_mps(), "mask must be on MPS");
        TORCH_CHECK(mask_tensor.scalar_type() == at::kByte, "mask must be uint8");
        TORCH_CHECK(mask_tensor.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(mask_tensor.numel() == q.size(0) * q.size(1) * T * L,
                    "mask must have shape [B,H,T,L]");
    }

    auto qf = q.to(at::kFloat);
    auto kf = k.to(at::kFloat);
    auto vf = v.to(at::kFloat);

    if (q.size(1) != k.size(1)) {
        const int64_t hq = q.size(1);
        const int64_t hkv = k.size(1);
        const int64_t group_size = hq / hkv;
        std::vector<int64_t> map_vec(static_cast<size_t>(hq));
        for (int64_t i = 0; i < hq; ++i) {
            map_vec[static_cast<size_t>(i)] = i / group_size;
        }
        auto head_map = at::tensor(
            map_vec,
            q.options().device(q.device()).dtype(at::kLong)
        );
        kf = kf.index_select(/*dim=*/1, head_map);
        vf = vf.index_select(/*dim=*/1, head_map);
    }

    auto scores = at::matmul(qf, kf.transpose(-2, -1)) / std::sqrt(static_cast<double>(Dh));
    if (mask.has_value()) {
        scores = scores.masked_fill(mask_tensor.eq(0), -std::numeric_limits<float>::infinity());
    }
    if (causal) {
        // Query rows correspond to the tail window [L-T, L), so causal bounds
        // must be shifted by q_start rather than using naive row index.
        const int64_t q_start = std::max<int64_t>(0, L - T);
        auto causal_mask = at::triu(
            at::ones({T, L}, q.options().dtype(at::kBool)),
            /*diagonal=*/q_start + 1
        );
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());
    }

    auto finite_row = at::isfinite(scores).any(-1, true);
    auto safe_scores = at::where(finite_row, scores, at::zeros_like(scores));
    auto probs = at::softmax(safe_scores, -1);
    probs = at::where(finite_row, probs, at::zeros_like(probs));

    auto out = at::matmul(probs, vf);
    return out.to(q.scalar_type());
}

at::Tensor metal_flex_attn_fast_dispatch_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& block_written,
    int64_t block_size,
    bool causal,
    bool use_active_dispatch
) {
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "flex_attn_metal_fast expects q/k/v on MPS");
    TORCH_CHECK(block_written.device().is_mps(), "block_written must be on MPS");
    TORCH_CHECK(
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)
            && q.scalar_type() == k.scalar_type()
            && q.scalar_type() == v.scalar_type(),
        "flex_attn_metal_fast currently supports float16 or bfloat16 (matching dtypes)"
    );
    TORCH_CHECK(block_written.scalar_type() == at::kByte, "block_written must be uint8");
    TORCH_CHECK(block_written.is_contiguous(), "block_written must be contiguous");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must match");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
                "q/k must match on batch and head dim");
    TORCH_CHECK(q.size(1) >= k.size(1), "q heads must be >= kv heads");
    TORCH_CHECK((q.size(1) % k.size(1)) == 0, "q heads must be divisible by kv heads for GQA");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");

    auto& rt = get_metal_runtime();
    TORCH_CHECK(rt.init_ok, "flex_attn_metal_fast: metal runtime init failed");

    const auto input_dtype = q.scalar_type();
    const bool use_bf16_io = (input_dtype == at::kBFloat16);
    const at::Tensor qh = use_bf16_io ? q.to(at::kHalf).contiguous() : q.contiguous();
    const at::Tensor kh = use_bf16_io ? k.to(at::kHalf).contiguous() : k.contiguous();
    const at::Tensor vh = use_bf16_io ? v.to(at::kHalf).contiguous() : v.contiguous();

    const uint32_t B = static_cast<uint32_t>(qh.size(0));
    const uint32_t Hq = static_cast<uint32_t>(qh.size(1));
    const uint32_t Hkv = static_cast<uint32_t>(kh.size(1));
    const uint32_t T = static_cast<uint32_t>(qh.size(2));
    const uint32_t L = static_cast<uint32_t>(kh.size(2));
    const uint32_t Dh = static_cast<uint32_t>(qh.size(3));
    TORCH_CHECK(Dh <= 128, "flex_attn_metal_fast currently supports Dh <= 128");
    const uint32_t BlockSize = static_cast<uint32_t>(block_size);
    const uint32_t KVBLOCKS = (L + BlockSize - 1) / BlockSize;
    const uint32_t Causal = causal ? 1u : 0u;
    TORCH_CHECK(block_written.numel() == static_cast<int64_t>(KVBLOCKS),
                "block_written must have exactly ceil(L/block_size) elements");

    at::Tensor active_blocks;
    uint32_t ActiveCount = KVBLOCKS;
    if (use_active_dispatch) {
        active_blocks = at::nonzero(block_written.gt(0)).flatten().to(at::kInt).contiguous();
        ActiveCount = static_cast<uint32_t>(active_blocks.numel());
    }

    at::Tensor out = at::zeros_like(qh);
    if (ActiveCount == 0) {
        return use_bf16_io ? out.to(input_dtype) : out;
    }
    const uint32_t FP16Accum = enable_fp16_accum() ? 1u : 0u;
    if (use_active_dispatch && ActiveCount > 0u) {
        if (Dh == 64u && BlockSize == 4u) {
            const float density = static_cast<float>(ActiveCount) / static_cast<float>(std::max<uint32_t>(1u, KVBLOCKS));
            const bool use_gqa1_specialized = (Hq == Hkv);
            const bool use_gqa2_specialized = (Hq == (Hkv << 1));
            const bool use_gqa2_dense = use_gqa2_specialized && (ActiveCount == KVBLOCKS);
            const bool use_gqa2_dualhead = enable_gqa2_dualhead_specialization() && use_gqa2_specialized && (density <= 0.75f) && (T >= 256u);
            const uint32_t tuned_tg = get_tg_size();
            if (use_gqa1_specialized) {
                dispatch_fast_kernel(
                    rt.pipeline_dh64_bs4_gqa1_single,
                    rt.thread_execution_width_dh64_bs4_gqa1_single,
                    qh, kh, vh, active_blocks, out,
                    B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                    FP16Accum,
                    "flex_attn_metal_fast", tuned_tg
                );
            } else if (use_gqa2_dualhead) {
                dispatch_fast_kernel_dh64_bs4_gqa2_dualhead(
                    rt.pipeline_dh64_bs4_gqa2_dualhead,
                    rt.thread_execution_width_dh64_bs4_gqa2_dualhead,
                    qh, kh, vh, active_blocks, out,
                    B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                    FP16Accum,
                    "flex_attn_metal_fast", tuned_tg
                );
            } else if (use_gqa2_dense) {
                dispatch_fast_kernel(
                    rt.pipeline_dh64_bs4_gqa2_dense,
                    rt.thread_execution_width_dh64_bs4_gqa2_dense,
                    qh, kh, vh, active_blocks, out,
                    B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                    FP16Accum,
                    "flex_attn_metal_fast", tuned_tg
                );
            } else if (use_gqa2_specialized) {
                dispatch_fast_kernel(
                    rt.pipeline_dh64_bs4_gqa2_single,
                    rt.thread_execution_width_dh64_bs4_gqa2_single,
                    qh, kh, vh, active_blocks, out,
                    B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                    FP16Accum,
                    "flex_attn_metal_fast", tuned_tg
                );
            } else {
                dispatch_fast_kernel(
                    rt.pipeline_dh64_bs4_single, rt.thread_execution_width_dh64_bs4_single, qh, kh, vh, active_blocks, out,
                    B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                    FP16Accum,
                    "flex_attn_metal_fast", tuned_tg
                );
            }
        } else {
            dispatch_fast_kernel(
                rt.pipeline_generic, rt.thread_execution_width_generic, qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast", 0u
            );
        }
    } else {
        if (Dh == 64u && BlockSize == 4u && Hq == Hkv) {
            dispatch_fast_kernel(
                rt.pipeline_dh64_bs4_gqa1_block_written,
                rt.thread_execution_width_dh64_bs4_gqa1_block_written,
                qh, kh, vh, block_written, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast", get_tg_size()
            );
        } else {
            dispatch_fast_kernel(
                rt.pipeline_generic_block_written,
                rt.thread_execution_width_generic_block_written,
                qh, kh, vh, block_written, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast", 0u
            );
        }
    }
    // Do not force immediate commit/wait here; let MPS stream scheduling batch
    // this op naturally with surrounding kernels for better throughput.
    return use_bf16_io ? out.to(input_dtype) : out;
}

at::Tensor metal_flex_attn_fast_dispatch_active_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& active_blocks,
    int64_t block_size,
    bool causal
) {
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "flex_attn_metal_fast_active expects q/k/v on MPS");
    TORCH_CHECK(active_blocks.device().is_mps(), "active_blocks must be on MPS");
    TORCH_CHECK(
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)
            && q.scalar_type() == k.scalar_type()
            && q.scalar_type() == v.scalar_type(),
        "flex_attn_metal_fast_active currently supports float16 or bfloat16 (matching dtypes)"
    );
    TORCH_CHECK(active_blocks.scalar_type() == at::kInt, "active_blocks must be int32");
    TORCH_CHECK(active_blocks.is_contiguous(), "active_blocks must be contiguous");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must match");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
                "q/k must match on batch and head dim");
    TORCH_CHECK(q.size(1) >= k.size(1), "q heads must be >= kv heads");
    TORCH_CHECK((q.size(1) % k.size(1)) == 0, "q heads must be divisible by kv heads for GQA");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");

    auto& rt = get_metal_runtime();
    TORCH_CHECK(rt.init_ok, "flex_attn_metal_fast_active: metal runtime init failed");

    const auto input_dtype = q.scalar_type();
    const bool use_bf16_io = (input_dtype == at::kBFloat16);
    const bool use_native_bf16_generic = use_bf16_io && (rt.pipeline_generic_bf16 != nil);
    const at::Tensor qh = use_native_bf16_generic ? q.contiguous() : (use_bf16_io ? q.to(at::kHalf).contiguous() : q.contiguous());
    const at::Tensor kh = use_native_bf16_generic ? k.contiguous() : (use_bf16_io ? k.to(at::kHalf).contiguous() : k.contiguous());
    const at::Tensor vh = use_native_bf16_generic ? v.contiguous() : (use_bf16_io ? v.to(at::kHalf).contiguous() : v.contiguous());

    const uint32_t B = static_cast<uint32_t>(qh.size(0));
    const uint32_t Hq = static_cast<uint32_t>(qh.size(1));
    const uint32_t Hkv = static_cast<uint32_t>(kh.size(1));
    const uint32_t T = static_cast<uint32_t>(qh.size(2));
    const uint32_t L = static_cast<uint32_t>(kh.size(2));
    const uint32_t Dh = static_cast<uint32_t>(qh.size(3));
    TORCH_CHECK(Dh <= 128, "flex_attn_metal_fast_active currently supports Dh <= 128");
    const uint32_t BlockSize = static_cast<uint32_t>(block_size);
    const uint32_t KVBLOCKS = (L + BlockSize - 1) / BlockSize;
    const uint32_t Causal = causal ? 1u : 0u;
    TORCH_CHECK(
        active_blocks.numel() <= static_cast<int64_t>(KVBLOCKS),
        "active_blocks numel must be <= ceil(L/block_size)"
    );
    const uint32_t ActiveCount = static_cast<uint32_t>(active_blocks.numel());

    at::Tensor out = at::zeros_like(qh);
    if (ActiveCount == 0) {
        return use_bf16_io ? out.to(input_dtype) : out;
    }
    const uint32_t FP16Accum = enable_fp16_accum() ? 1u : 0u;
    if (use_native_bf16_generic) {
        dispatch_fast_kernel(
            rt.pipeline_generic_bf16, rt.thread_execution_width_generic_bf16, qh, kh, vh, active_blocks, out,
            B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
            FP16Accum,
            "flex_attn_metal_fast_active", 0u
        );
    } else if (Dh == 64u && BlockSize == 4u) {
        const float density = static_cast<float>(ActiveCount) / static_cast<float>(std::max<uint32_t>(1u, KVBLOCKS));
        const bool use_gqa1_specialized = (Hq == Hkv);
        const bool use_gqa2_specialized = (Hq == (Hkv << 1));
        const bool use_gqa2_dense = use_gqa2_specialized && (ActiveCount == KVBLOCKS);
        const bool use_gqa2_dualhead = enable_gqa2_dualhead_specialization() && use_gqa2_specialized && (density <= 0.75f) && (T >= 256u);
        const uint32_t tuned_tg = get_tg_size();
        if (use_gqa1_specialized) {
            dispatch_fast_kernel(
                rt.pipeline_dh64_bs4_gqa1_single,
                rt.thread_execution_width_dh64_bs4_gqa1_single,
                qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast_active", tuned_tg
            );
        } else if (use_gqa2_dualhead) {
            dispatch_fast_kernel_dh64_bs4_gqa2_dualhead(
                rt.pipeline_dh64_bs4_gqa2_dualhead,
                rt.thread_execution_width_dh64_bs4_gqa2_dualhead,
                qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast_active", tuned_tg
            );
        } else if (use_gqa2_dense) {
            dispatch_fast_kernel(
                rt.pipeline_dh64_bs4_gqa2_dense,
                rt.thread_execution_width_dh64_bs4_gqa2_dense,
                qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast_active", tuned_tg
            );
        } else if (use_gqa2_specialized) {
            dispatch_fast_kernel(
                rt.pipeline_dh64_bs4_gqa2_single,
                rt.thread_execution_width_dh64_bs4_gqa2_single,
                qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast_active", tuned_tg
            );
        } else {
            dispatch_fast_kernel(
                rt.pipeline_dh64_bs4_single, rt.thread_execution_width_dh64_bs4_single, qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast_active", tuned_tg
            );
        }
    } else {
        dispatch_fast_kernel(
            rt.pipeline_generic, rt.thread_execution_width_generic, qh, kh, vh, active_blocks, out,
            B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
            FP16Accum,
            "flex_attn_metal_fast_active", 0u
        );
    }
    return (use_bf16_io && !use_native_bf16_generic) ? out.to(input_dtype) : out;
}

at::Tensor metal_flex_attn_fast_dispatch_from_mask_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    bool causal
) {
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "flex_attn_metal_fast expects q/k/v on MPS");
    TORCH_CHECK(
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)
            && q.scalar_type() == k.scalar_type()
            && q.scalar_type() == v.scalar_type(),
        "flex_attn_metal_fast currently supports float16 or bfloat16 (matching dtypes)"
    );
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must match");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
                "q/k must match on batch and head dim");
    TORCH_CHECK(q.size(1) >= k.size(1), "q heads must be >= kv heads");
    TORCH_CHECK((q.size(1) % k.size(1)) == 0, "q heads must be divisible by kv heads for GQA");

    const uint32_t B = static_cast<uint32_t>(q.size(0));
    const uint32_t Hq = static_cast<uint32_t>(q.size(1));
    const uint32_t Hkv = static_cast<uint32_t>(k.size(1));
    const uint32_t T = static_cast<uint32_t>(q.size(2));
    const uint32_t L = static_cast<uint32_t>(k.size(2));
    const uint32_t Dh = static_cast<uint32_t>(q.size(3));
    const uint32_t BlockSize = static_cast<uint32_t>(get_block_size());
    TORCH_CHECK(BlockSize > 0, "WORLD_METAL_BLOCK_SIZE must be > 0");
    const uint32_t KVBLOCKS = (L + BlockSize - 1) / BlockSize;
    const uint32_t Causal = causal ? 1u : 0u;

    at::Tensor mask_tensor;
    at::Tensor block_written;
    if (mask.has_value()) {
        mask_tensor = *mask;
        TORCH_CHECK(mask_tensor.device().is_mps(), "mask must be on MPS");
        TORCH_CHECK(mask_tensor.scalar_type() == at::kByte, "mask must be uint8");
        TORCH_CHECK(mask_tensor.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(mask_tensor.numel() == q.size(0) * q.size(1) * q.size(2) * k.size(2),
                    "mask must have shape [B,H,T,L]");
        // Fast-kernel contract today: a single shared, block-wise mask state.
        // We enforce this explicitly to avoid silent semantic drift.
        auto row = mask_tensor.index({0, 0, 0}).contiguous(); // [L]
        auto shared_ok = mask_tensor.eq(row.view({1, 1, 1, static_cast<int64_t>(L)})).all().item<bool>();
        TORCH_CHECK(
            shared_ok,
            "flex_attn_metal_fast expects a shared mask across batch/head/query dimensions"
        );

        const int64_t full_blocks = static_cast<int64_t>(L / BlockSize);
        const int64_t rem = static_cast<int64_t>(L % BlockSize);
        at::Tensor block_vals;

        if (full_blocks > 0) {
            auto prefix = row.slice(/*dim=*/0, /*start=*/0, /*end=*/full_blocks * static_cast<int64_t>(BlockSize));
            auto blocks2d = prefix.view({full_blocks, static_cast<int64_t>(BlockSize)});
            auto first = blocks2d.index({at::indexing::Slice(), 0}).unsqueeze(1);
            auto full_ok = blocks2d.eq(first).all().item<bool>();
            TORCH_CHECK(
                full_ok,
                "flex_attn_metal_fast expects block-wise mask values (constant within each block)"
            );
            block_vals = first.squeeze(1).to(at::kByte);
        } else {
            block_vals = at::empty({0}, q.options().dtype(at::kByte));
        }

        if (rem > 0) {
            auto tail = row.slice(/*dim=*/0, /*start=*/full_blocks * static_cast<int64_t>(BlockSize), /*end=*/static_cast<int64_t>(L));
            auto tail_first = tail.index({0});
            auto tail_ok = tail.eq(tail_first).all().item<bool>();
            TORCH_CHECK(
                tail_ok,
                "flex_attn_metal_fast expects block-wise mask values (constant within each block)"
            );
            auto tail_val = tail_first.to(at::kByte).view({1});
            block_vals = (block_vals.numel() > 0) ? at::cat({block_vals, tail_val}, /*dim=*/0) : tail_val;
        }

        block_written = block_vals.contiguous();
    } else {
        block_written = at::ones({static_cast<int64_t>(KVBLOCKS)}, q.options().dtype(at::kByte)).contiguous();
    }

    return metal_flex_attn_fast_dispatch_impl(
        q, k, v, block_written, static_cast<int64_t>(BlockSize), causal, prefer_active_dispatch_path()
    );
}

at::Tensor metal_flex_attn_fast_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    bool causal
) {
    if (fast_no_fallback()) {
        return metal_flex_attn_fast_dispatch_from_mask_impl(q, k, v, mask, causal);
    }
    // Keep ref as a safety net while fast path stabilizes.
    try {
        return metal_flex_attn_fast_dispatch_from_mask_impl(q, k, v, mask, causal);
    } catch (...) {
        return metal_flex_attn_ref_impl(q, k, v, mask, causal);
    }
}

at::Tensor metal_flex_attn_fast_blocks_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& block_written,
    int64_t block_size,
    bool causal
) {
    if (fast_no_fallback()) {
        return metal_flex_attn_fast_dispatch_impl(q, k, v, block_written, block_size, causal, prefer_active_dispatch_path());
    }
    try {
        return metal_flex_attn_fast_dispatch_impl(q, k, v, block_written, block_size, causal, prefer_active_dispatch_path());
    } catch (...) {
        // Reconstruct dense mask for reference fallback.
        const int64_t B = q.size(0);
        const int64_t Hq = q.size(1);
        const int64_t T = q.size(2);
        const int64_t L = k.size(2);
        at::Tensor dense = at::zeros({L}, q.options().dtype(at::kByte));
        for (int64_t b = 0; b < block_written.numel(); ++b) {
            if (block_written.index({b}).item<int64_t>() != 0) {
                const int64_t s = b * block_size;
                const int64_t e = std::min<int64_t>(L, s + block_size);
                dense.index_put_({at::indexing::Slice(s, e)}, 1);
            }
        }
        auto dense4d = dense.view({1, 1, 1, L}).expand({B, Hq, T, L}).contiguous();
        return metal_flex_attn_ref_impl(q, k, v, dense4d, causal);
    }
}

at::Tensor metal_flex_attn_fast_blocks_direct_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& block_written,
    int64_t block_size,
    bool causal
) {
    if (fast_no_fallback()) {
        return metal_flex_attn_fast_dispatch_impl(q, k, v, block_written, block_size, causal, false);
    }
    try {
        return metal_flex_attn_fast_dispatch_impl(q, k, v, block_written, block_size, causal, false);
    } catch (...) {
        const int64_t B = q.size(0);
        const int64_t Hq = q.size(1);
        const int64_t T = q.size(2);
        const int64_t L = k.size(2);
        at::Tensor dense = at::zeros({L}, q.options().dtype(at::kByte));
        for (int64_t b = 0; b < block_written.numel(); ++b) {
            if (block_written.index({b}).item<int64_t>() != 0) {
                const int64_t s = b * block_size;
                const int64_t e = std::min<int64_t>(L, s + block_size);
                dense.index_put_({at::indexing::Slice(s, e)}, 1);
            }
        }
        auto dense4d = dense.view({1, 1, 1, L}).expand({B, Hq, T, L}).contiguous();
        return metal_flex_attn_ref_impl(q, k, v, dense4d, causal);
    }
}

at::Tensor metal_flex_attn_fast_active_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& active_blocks,
    int64_t block_size,
    bool causal
) {
    if (fast_no_fallback()) {
        return metal_flex_attn_fast_dispatch_active_impl(q, k, v, active_blocks, block_size, causal);
    }
    try {
        return metal_flex_attn_fast_dispatch_active_impl(q, k, v, active_blocks, block_size, causal);
    } catch (...) {
        // Reconstruct dense mask for reference fallback.
        const int64_t B = q.size(0);
        const int64_t Hq = q.size(1);
        const int64_t T = q.size(2);
        const int64_t L = k.size(2);
        const int64_t kv_blocks = (L + block_size - 1) / block_size;
        at::Tensor bw = at::zeros({kv_blocks}, q.options().dtype(at::kByte));
        for (int64_t i = 0; i < active_blocks.numel(); ++i) {
            const int64_t bi = active_blocks.index({i}).item<int64_t>();
            if (bi >= 0 && bi < kv_blocks) {
                bw.index_put_({bi}, 1);
            }
        }
        at::Tensor dense = at::zeros({L}, q.options().dtype(at::kByte));
        for (int64_t b = 0; b < bw.numel(); ++b) {
            if (bw.index({b}).item<int64_t>() != 0) {
                const int64_t s = b * block_size;
                const int64_t e = std::min<int64_t>(L, s + block_size);
                dense.index_put_({at::indexing::Slice(s, e)}, 1);
            }
        }
        auto dense4d = dense.view({1, 1, 1, L}).expand({B, Hq, T, L}).contiguous();
        return metal_flex_attn_ref_impl(q, k, v, dense4d, causal);
    }
}

at::Tensor metal_flex_attn_fast_active_counted_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& active_blocks,
    const at::Tensor& active_count,
    int64_t block_size,
    bool causal
) {
    TORCH_CHECK(active_count.numel() == 1, "active_count must be a scalar tensor");
    TORCH_CHECK(active_blocks.is_contiguous(), "active_blocks must be contiguous");
    TORCH_CHECK(active_blocks.scalar_type() == at::kInt, "active_blocks must be int32");
    TORCH_CHECK(active_count.scalar_type() == at::kInt, "active_count must be int32");
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "flex_attn_metal_fast_active_counted expects q/k/v on MPS");
    TORCH_CHECK(
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)
            && q.scalar_type() == k.scalar_type()
            && q.scalar_type() == v.scalar_type(),
        "flex_attn_metal_fast_active_counted currently supports float16 or bfloat16 (matching dtypes)"
    );
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must match");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
                "q/k must match on batch and head dim");
    TORCH_CHECK(q.size(1) >= k.size(1), "q heads must be >= kv heads");
    TORCH_CHECK((q.size(1) % k.size(1)) == 0, "q heads must be divisible by kv heads for GQA");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");

    auto& rt = get_metal_runtime();
    TORCH_CHECK(rt.init_ok, "flex_attn_metal_fast_active_counted: metal runtime init failed");

    const auto input_dtype = q.scalar_type();
    const bool use_bf16_io = (input_dtype == at::kBFloat16);
    const at::Tensor qh = use_bf16_io ? q.to(at::kHalf).contiguous() : q.contiguous();
    const at::Tensor kh = use_bf16_io ? k.to(at::kHalf).contiguous() : k.contiguous();
    const at::Tensor vh = use_bf16_io ? v.to(at::kHalf).contiguous() : v.contiguous();

    const uint32_t B = static_cast<uint32_t>(qh.size(0));
    const uint32_t Hq = static_cast<uint32_t>(qh.size(1));
    const uint32_t Hkv = static_cast<uint32_t>(kh.size(1));
    const uint32_t T = static_cast<uint32_t>(qh.size(2));
    const uint32_t L = static_cast<uint32_t>(kh.size(2));
    const uint32_t Dh = static_cast<uint32_t>(qh.size(3));
    TORCH_CHECK(Dh <= 128, "flex_attn_metal_fast_active_counted currently supports Dh <= 128");
    const uint32_t BlockSize = static_cast<uint32_t>(block_size);
    const uint32_t Causal = causal ? 1u : 0u;

    const uint32_t ActiveCount = static_cast<uint32_t>(active_count.item<int32_t>());
    at::Tensor out = at::zeros_like(qh);
    if (ActiveCount == 0) {
        return use_bf16_io ? out.to(input_dtype) : out;
    }
    const uint32_t FP16Accum = enable_fp16_accum() ? 1u : 0u;
    if (Dh == 64u && BlockSize == 4u) {
        const bool use_gqa1_specialized = (Hq == Hkv);
        const uint32_t tuned_tg = get_tg_size();
        if (use_gqa1_specialized) {
            dispatch_fast_kernel(
                rt.pipeline_dh64_bs4_gqa1_single,
                rt.thread_execution_width_dh64_bs4_gqa1_single,
                qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast_active_counted", tuned_tg
            );
        } else {
            dispatch_fast_kernel(
                rt.pipeline_dh64_bs4_single, rt.thread_execution_width_dh64_bs4_single,
                qh, kh, vh, active_blocks, out,
                B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
                FP16Accum,
                "flex_attn_metal_fast_active_counted", tuned_tg
            );
        }
    } else {
        dispatch_fast_kernel(
            rt.pipeline_generic, rt.thread_execution_width_generic,
            qh, kh, vh, active_blocks, out,
            B, Hq, T, L, Dh, BlockSize, ActiveCount, Causal, Hkv,
            FP16Accum,
            "flex_attn_metal_fast_active_counted", 0u
        );
    }
    return use_bf16_io ? out.to(input_dtype) : out;
}

at::Tensor metal_flex_attn_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    bool causal
) {
    // Backward-compatible alias; default to ref behavior.
    return metal_flex_attn_ref_impl(q, k, v, mask, causal);
}

} // namespace

TORCH_LIBRARY(world, m) {
    m.def("flex_attn_metal(Tensor q, Tensor k, Tensor v, Tensor? mask=None, bool causal=True) -> Tensor");
    m.def("flex_attn_metal_ref(Tensor q, Tensor k, Tensor v, Tensor? mask=None, bool causal=True) -> Tensor");
    m.def("flex_attn_metal_fast(Tensor q, Tensor k, Tensor v, Tensor? mask=None, bool causal=True) -> Tensor");
    m.def("flex_attn_metal_fast_blocks(Tensor q, Tensor k, Tensor v, Tensor block_written, int block_size, bool causal=True) -> Tensor");
    m.def("flex_attn_metal_fast_blocks_direct(Tensor q, Tensor k, Tensor v, Tensor block_written, int block_size, bool causal=True) -> Tensor");
    m.def("flex_attn_metal_fast_active(Tensor q, Tensor k, Tensor v, Tensor active_blocks, int block_size, bool causal=True) -> Tensor");
    m.def("flex_attn_metal_fast_active_counted(Tensor q, Tensor k, Tensor v, Tensor active_blocks, Tensor active_count, int block_size, bool causal=True) -> Tensor");
}

TORCH_LIBRARY_IMPL(world, MPS, m) {
    m.impl("flex_attn_metal", &metal_flex_attn_impl);
    m.impl("flex_attn_metal_ref", &metal_flex_attn_ref_impl);
    m.impl("flex_attn_metal_fast", &metal_flex_attn_fast_impl);
    m.impl("flex_attn_metal_fast_blocks", &metal_flex_attn_fast_blocks_impl);
    m.impl("flex_attn_metal_fast_blocks_direct", &metal_flex_attn_fast_blocks_direct_impl);
    m.impl("flex_attn_metal_fast_active", &metal_flex_attn_fast_active_impl);
    m.impl("flex_attn_metal_fast_active_counted", &metal_flex_attn_fast_active_counted_impl);
}

