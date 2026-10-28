#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> 
#include <stdint.h>
#include <math.h>

using namespace nvcuda; 

#define BLOCK_SIZE 32
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -1e20f;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}
//gemv_w4a16_fused
__global__ void gemv_w4a16_fused_kernel(
    const half* __restrict__ scales_base,
    const uint8_t* __restrict__ weights_base,
    const float* __restrict__ x,
    float* __restrict__ y,
    const half* __restrict__ bias,
    float* __restrict__ residual_add,
    int rows,
    int cols
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    int blocks_per_row = cols / BLOCK_SIZE;
    const half* row_scales = scales_base + warp_id * blocks_per_row;
    const uint4* row_weights_u4 = reinterpret_cast<const uint4*>(weights_base + (size_t)warp_id * blocks_per_row * 16);

    
    extern __shared__ float s_x[];
    if (cols == 2048) {
        int tid = threadIdx.x;
        float4* s_x_v = reinterpret_cast<float4*>(s_x);
        const float4* x_v = reinterpret_cast<const float4*>(x);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s_x_v[tid + i * 128] = x_v[tid + i * 128];
        }
        __syncthreads();
    }

    float thread_sum = 0.0f;

    for (int b = lane_id; b < blocks_per_row; b += WARP_SIZE) {
        float scale = __half2float(row_scales[b]);
        uint4 w_vec = row_weights_u4[b];
        const float* x_ptr = (cols == 2048) ? (s_x + b * BLOCK_SIZE) : (x + b * BLOCK_SIZE);

        float block_acc = 0.0f;
        uint32_t w_regs[4] = { w_vec.x, w_vec.y, w_vec.z, w_vec.w };

        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t w = w_regs[chunk];
            const float* xp = x_ptr + chunk * 8;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t byte = (w >> (i * 8)) & 0xFF;
                float q0 = (float)((int)(byte & 0x0F) - 8);
                float q1 = (float)((int)((byte >> 4) & 0x0F) - 8);

                block_acc += q0 * xp[i * 2];
                block_acc += q1 * xp[i * 2 + 1];
            }
        }

        thread_sum += block_acc * scale;
    }

    float row_total = warp_reduce_sum(thread_sum);

    if (lane_id == 0) {
        if (bias != nullptr) {
            row_total += __half2float(bias[warp_id]);
        }
        if (residual_add != nullptr) {
            residual_add[warp_id] += row_total;
        }
        if (y != nullptr) {
            y[warp_id] = row_total;
        }
    }
}
//WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_w4a16_wmma_kernel(
    const half* __restrict__ scales_base,
    const uint8_t* __restrict__ weights_base,
    const half* __restrict__ x_fp16,
    float* __restrict__ y_out,
    int M, 
    int K, 
    int N  
) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_id_local = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int total_n_tiles = (N + WMMA_N - 1) / WMMA_N;
    int tile_m_idx = warp_id_global / total_n_tiles;
    int tile_n_idx = warp_id_global % total_n_tiles;

    int m_start = tile_m_idx * WMMA_M;
    int n_start = tile_n_idx * WMMA_N;

    if (m_start >= M || n_start >= N) return;

    __shared__ half s_w[4][16][16]; 

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int blocks_per_row = K / BLOCK_SIZE;

    for (int k_step = 0; k_step < K; k_step += WMMA_K) {
        
        const half* a_ptr = x_fp16 + m_start * K + k_step;
        wmma::load_matrix_sync(a_frag, a_ptr, K);

        
        int row_in_tile = lane_id / 2;
        int col_half = (lane_id % 2) * 8;
        int global_row = n_start + row_in_tile;
        int global_k = k_step + col_half;

        if (global_row < N && global_k < K) {
            int b_idx = global_k / BLOCK_SIZE;
            int offset_in_block = global_k % BLOCK_SIZE;
            
            float scale = __half2float(scales_base[global_row * blocks_per_row + b_idx]);
            const uint8_t* w_ptr = weights_base + (size_t)global_row * blocks_per_row * 16 + b_idx * 16 + (offset_in_block / 2);

            uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_ptr);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t byte = (w4 >> (i * 8)) & 0xFF;
                float q0 = (float)((int)(byte & 0x0F) - 8) * scale;
                float q1 = (float)((int)((byte >> 4) & 0x0F) - 8) * scale;

                s_w[warp_id_local][row_in_tile][col_half + i * 2] = __float2half(q0);
                s_w[warp_id_local][row_in_tile][col_half + i * 2 + 1] = __float2half(q1);
            }
        }

        
        wmma::load_matrix_sync(b_frag, (half*)&s_w[warp_id_local][0][0], 16);

        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    
    if (m_start + 16 <= M && n_start + 16 <= N) {
        wmma::store_matrix_sync(y_out + m_start * N + n_start, c_frag, N, wmma::mem_row_major);
    } else {
        __shared__ float s_out[4][16][16];
        wmma::store_matrix_sync((float*)&s_out[warp_id_local][0][0], c_frag, 16, wmma::mem_row_major);
        for (int i = lane_id; i < 256; i += WARP_SIZE) {
            int r = i / 16;
            int c = i % 16;
            if (m_start + r < M && n_start + c < N) {
                y_out[(m_start + r) * N + (n_start + c)] = s_out[warp_id_local][r][c];
            }
        }
    }
}

__global__ void gemv_fused_swiglu_split_kernel(
    const half* __restrict__ gate_scales,
    const uint8_t* __restrict__ gate_weights,
    const half* __restrict__ up_scales,
    const uint8_t* __restrict__ up_weights,
    const float* __restrict__ x,
    float* __restrict__ act_out,
    int hidden_dim,
    int dim
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= hidden_dim) return;

    int blocks_per_row = dim / BLOCK_SIZE;

    const half* g_scales = gate_scales + warp_id * blocks_per_row;
    const uint4* g_weights_u4 = reinterpret_cast<const uint4*>(gate_weights + (size_t)warp_id * blocks_per_row * 16);

    const half* u_scales = up_scales + warp_id * blocks_per_row;
    const uint4* u_weights_u4 = reinterpret_cast<const uint4*>(up_weights + (size_t)warp_id * blocks_per_row * 16);

    
    extern __shared__ float s_x_swiglu[];
    if (dim == 2048) {
        int tid = threadIdx.x;
        float4* s_x_v = reinterpret_cast<float4*>(s_x_swiglu);
        const float4* x_v = reinterpret_cast<const float4*>(x);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s_x_v[tid + i * 128] = x_v[tid + i * 128];
        }
        __syncthreads();
    }

    float g_sum = 0.0f;
    float u_sum = 0.0f;

    for (int b = lane_id; b < blocks_per_row; b += WARP_SIZE) {
        const float* x_ptr = (dim == 2048) ? (s_x_swiglu + b * BLOCK_SIZE) : (x + b * BLOCK_SIZE);

        float g_scale = __half2float(g_scales[b]);
        uint4 g_vec = g_weights_u4[b];
        uint32_t gw[4] = { g_vec.x, g_vec.y, g_vec.z, g_vec.w };

        float u_scale = __half2float(u_scales[b]);
        uint4 u_vec = u_weights_u4[b];
        uint32_t uw[4] = { u_vec.x, u_vec.y, u_vec.z, u_vec.w };

        float g_acc = 0.0f;
        float u_acc = 0.0f;

        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t g_reg = gw[chunk];
            uint32_t u_reg = uw[chunk];
            const float* xp = x_ptr + chunk * 8;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float x0 = xp[i * 2];
                float x1 = xp[i * 2 + 1];

                uint8_t g_byte = (g_reg >> (i * 8)) & 0xFF;
                g_acc += (float)((int)(g_byte & 0x0F) - 8) * x0 + (float)((int)((g_byte >> 4) & 0x0F) - 8) * x1;

                uint8_t u_byte = (u_reg >> (i * 8)) & 0xFF;
                u_acc += (float)((int)(u_byte & 0x0F) - 8) * x0 + (float)((int)((u_byte >> 4) & 0x0F) - 8) * x1;
            }
        }

        g_sum += g_acc * g_scale;
        u_sum += u_acc * u_scale;
    }

    float g_total = warp_reduce_sum(g_sum);
    float u_total = warp_reduce_sum(u_sum);

    if (lane_id == 0) {
        float silu = g_total / (1.0f + expf(-g_total));
        act_out[warp_id] = silu * u_total;
    }
}

//RMSNORM
__global__ void rmsnorm_gpu_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const half* __restrict__ weight,
    int size,
    float eps
) {
    int tid = threadIdx.x;
    float sum_sq = 0.0f;

    for (int i = tid; i < size; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_std;
    if (tid == 0) {
        s_inv_std = rsqrtf(sum_sq / (float)size + eps);
    }
    __syncthreads();

    float inv_std = s_inv_std;
    for (int i = tid; i < size; i += blockDim.x) {
        out[i] = x[i] * inv_std * __half2float(weight[i]);
    }
}

//RoPE
__global__ void rope_gpu_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float rope_theta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total_pairs = (n_heads + n_kv_heads) * half;

    if (tid >= total_pairs) return;

    bool is_q = tid < (n_heads * half);
    int pair_idx = is_q ? tid : (tid - n_heads * half);
    int h = pair_idx / half;
    int i = pair_idx % half;

    float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float sin_val = sinf(val);
    float cos_val = cosf(val);

    float* target_head = is_q ? (q + h * head_dim) : (k + h * head_dim);
    float v0 = target_head[i];
    float v1 = target_head[i + half];

    target_head[i] = v0 * cos_val - v1 * sin_val;
    target_head[i + half] = v0 * sin_val + v1 * cos_val;
}

//KV-STORE
__global__ void kv_store_gpu_kernel(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    int pos
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_kv_heads * head_dim;

    if (tid >= total_elements) return;

    int kv_h = tid / head_dim;
    int d = tid % head_dim;

    int offset = kv_h * (max_seq_len * head_dim) + pos * head_dim + d;
    k_cache[offset] = __float2half(k[tid]);
    v_cache[offset] = __float2half(v[tid]);
}

//ATTENTION DECODE
__global__ void attention_decode_gpu_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float sqrt_head_dim
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int kv_mul = n_heads / n_kv_heads;
    int kv_h = h / kv_mul;
    int kv_base = kv_h * (max_seq_len * head_dim);

    extern __shared__ float s_scores[];
    __shared__ float s_max_val;
    __shared__ float s_sum_exp;

    float inv_sqrt_head_dim = 1.0f / sqrt_head_dim;
    int floats_per_thread = head_dim / WARP_SIZE;




    
    float q_reg[4];
    const float* q_ptr = q + h * head_dim + lane_id * floats_per_thread;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_reg[i] = (i < floats_per_thread) ? q_ptr[i] : 0.0f;
    }

    
    for (int p = warp_id; p <= pos; p += num_warps) {
        const half* k_ptr = k_cache + kv_base + p * head_dim + lane_id * floats_per_thread;

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i < floats_per_thread) {
                dot += q_reg[i] * __half2float(k_ptr[i]);
            }
        }
        dot = warp_reduce_sum(dot);

        if (lane_id == 0) {
            s_scores[p] = dot * inv_sqrt_head_dim;
        }
    }
    __syncthreads();

    
    float local_max = -1e20f;
    for (int p = tid; p <= pos; p += blockDim.x) {
        if (s_scores[p] > local_max) local_max = s_scores[p];
    }
    local_max = block_reduce_max(local_max);
    if (tid == 0) s_max_val = local_max;
    __syncthreads();

    
    float local_sum = 0.0f;
    float max_val = s_max_val;
    for (int p = tid; p <= pos; p += blockDim.x) {
        float exp_val = expf(s_scores[p] - max_val);
        s_scores[p] = exp_val;
        local_sum += exp_val;
    }
    local_sum = block_reduce_sum(local_sum);
    if (tid == 0) s_sum_exp = local_sum;
    __syncthreads();

    
    float inv_sum = 1.0f / s_sum_exp;
    for (int p = tid; p <= pos; p += blockDim.x) {
        s_scores[p] *= inv_sum;
    }
    __syncthreads();

    
    if (tid < head_dim) {
        float out_val = 0.0f;
        #pragma unroll 4
        for (int p = 0; p <= pos; ++p) {
            out_val += s_scores[p] * __half2float(v_cache[kv_base + p * head_dim + tid]);
        }
        out[h * head_dim + tid] = out_val;
    }
}
__global__ void f32_to_f16_kernel(half* __restrict__ dst, const float* __restrict__ src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}
__device__ __forceinline__ float warp_reduce_max_abs_xor(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

__global__ void quantize_x_q8_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ xq,
    float2* __restrict__ xmeta,
    int n_blocks
) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_blocks) return; 

    float v = x[warp * 32 + lane];
    float amax = warp_reduce_max_abs_xor(v);
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));

    
    xq[warp * 32 + (lane & 1) * 16 + (lane >> 1)] = (int8_t)q;

    int s = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, off);
    }
    if (lane == 0) {
        xmeta[warp] = make_float2(amax / 127.0f, (float)s);
    }
}


__global__ void gemv_q4_dp4a_kernel(
    const half* __restrict__ scales_base,
    const uint8_t* __restrict__ weights_base,
    const int8_t* __restrict__ xq,
    const float2* __restrict__ xmeta,
    float* __restrict__ y,
    const half* __restrict__ bias,
    float* __restrict__ residual_add,
    int rows,
    int cols
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return; 

    int nb = cols >> 5;
    const half* row_scales = scales_base + (size_t)warp_id * nb;
    const uint4* row_w = reinterpret_cast<const uint4*>(weights_base + (size_t)warp_id * nb * 16);
    const int4* xq4 = reinterpret_cast<const int4*>(xq);

    float sum = 0.0f;
    #pragma unroll 2
    for (int b = lane; b < nb; b += 32) {
        uint4 w = row_w[b];       
        int4 xe = xq4[b * 2];     
        int4 xo = xq4[b * 2 + 1]; 
        float2 m = xmeta[b];

        int acc = 0;
        acc = __dp4a((int)(w.x & 0x0F0F0F0Fu), xe.x, acc);
        acc = __dp4a((int)((w.x >> 4) & 0x0F0F0F0Fu), xo.x, acc);
        acc = __dp4a((int)(w.y & 0x0F0F0F0Fu), xe.y, acc);
        acc = __dp4a((int)((w.y >> 4) & 0x0F0F0F0Fu), xo.y, acc);
        acc = __dp4a((int)(w.z & 0x0F0F0F0Fu), xe.z, acc);
        acc = __dp4a((int)((w.z >> 4) & 0x0F0F0F0Fu), xo.z, acc);
        acc = __dp4a((int)(w.w & 0x0F0F0F0Fu), xe.w, acc);
        acc = __dp4a((int)((w.w >> 4) & 0x0F0F0F0Fu), xo.w, acc);

        sum += __half2float(row_scales[b]) * m.x * ((float)acc - 8.0f * m.y);
    }

    float row_total = warp_reduce_sum(sum);

    if (lane == 0) {
        if (bias != nullptr) row_total += __half2float(bias[warp_id]);
        if (residual_add != nullptr) residual_add[warp_id] += row_total;
        if (y != nullptr) y[warp_id] = row_total;
    }
}


__global__ void swiglu_dp4a_kernel(
    const half* __restrict__ gate_scales,
    const uint8_t* __restrict__ gate_weights,
    const half* __restrict__ up_scales,
    const uint8_t* __restrict__ up_weights,
    const int8_t* __restrict__ xq,
    const float2* __restrict__ xmeta,
    float* __restrict__ act_out,
    int hidden_dim,
    int dim
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= hidden_dim) return;

    int nb = dim >> 5;
    const half* g_sc = gate_scales + (size_t)warp_id * nb;
    const half* u_sc = up_scales + (size_t)warp_id * nb;
    const uint4* g_w = reinterpret_cast<const uint4*>(gate_weights + (size_t)warp_id * nb * 16);
    const uint4* u_w = reinterpret_cast<const uint4*>(up_weights + (size_t)warp_id * nb * 16);
    const int4* xq4 = reinterpret_cast<const int4*>(xq);

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    #pragma unroll 2
    for (int b = lane; b < nb; b += 32) {
        uint4 gw = g_w[b];
        uint4 uw = u_w[b];
        int4 xe = xq4[b * 2];
        int4 xo = xq4[b * 2 + 1];
        float2 m = xmeta[b];

        int ga = 0;
        int ua = 0;
        ga = __dp4a((int)(gw.x & 0x0F0F0F0Fu), xe.x, ga);  ua = __dp4a((int)(uw.x & 0x0F0F0F0Fu), xe.x, ua);
        ga = __dp4a((int)((gw.x >> 4) & 0x0F0F0F0Fu), xo.x, ga);  ua = __dp4a((int)((uw.x >> 4) & 0x0F0F0F0Fu), xo.x, ua);
        ga = __dp4a((int)(gw.y & 0x0F0F0F0Fu), xe.y, ga);  ua = __dp4a((int)(uw.y & 0x0F0F0F0Fu), xe.y, ua);
        ga = __dp4a((int)((gw.y >> 4) & 0x0F0F0F0Fu), xo.y, ga);  ua = __dp4a((int)((uw.y >> 4) & 0x0F0F0F0Fu), xo.y, ua);
        ga = __dp4a((int)(gw.z & 0x0F0F0F0Fu), xe.z, ga);  ua = __dp4a((int)(uw.z & 0x0F0F0F0Fu), xe.z, ua);
        ga = __dp4a((int)((gw.z >> 4) & 0x0F0F0F0Fu), xo.z, ga);  ua = __dp4a((int)((uw.z >> 4) & 0x0F0F0F0Fu), xo.z, ua);
        ga = __dp4a((int)(gw.w & 0x0F0F0F0Fu), xe.w, ga);  ua = __dp4a((int)(uw.w & 0x0F0F0F0Fu), xe.w, ua);
        ga = __dp4a((int)((gw.w >> 4) & 0x0F0F0F0Fu), xo.w, ga);  ua = __dp4a((int)((uw.w >> 4) & 0x0F0F0F0Fu), xo.w, ua);

        g_sum += __half2float(g_sc[b]) * m.x * ((float)ga - 8.0f * m.y);
        u_sum += __half2float(u_sc[b]) * m.x * ((float)ua - 8.0f * m.y);
    }

    float g_total = warp_reduce_sum(g_sum);
    float u_total = warp_reduce_sum(u_sum);

    if (lane == 0) {
        float silu = g_total / (1.0f + expf(-g_total));
        act_out[warp_id] = silu * u_total;
    }
}
extern "C" {

bool cuda_check_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

void cuda_get_gpu_name(char* name_buf, int max_len) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        strncpy(name_buf, prop.name, max_len - 1);
        name_buf[max_len - 1] = '\0';
    } else {
        name_buf[0] = '\0';
    }
}

void cuda_warmup() {
    cudaFree(0);
}

void cuda_matvec_q4_fused(
    const void* d_scales,
    const void* d_weights,
    const float* d_x,
    float* d_y,
    const void* d_bias_fp16,
    float* d_residual_add,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int total_blocks = (rows + warps_per_block - 1) / warps_per_block;

    size_t shmem = (cols == 2048) ? (2048 * sizeof(float)) : 0;
    gemv_w4a16_fused_kernel<<<total_blocks, threads_per_block, shmem, stream>>>(
        (const half*)d_scales,
        (const uint8_t*)d_weights,
        d_x,
        d_y,
        (const half*)d_bias_fp16,
        d_residual_add,
        rows,
        cols
    );
}

void cuda_swiglu_gemv(
    const void* d_gate_scales,
    const void* d_gate_weights,
    const void* d_up_scales,
    const void* d_up_weights,
    const float* d_x,
    float* d_act_out,
    int hidden_dim,
    int dim,
    cudaStream_t stream
) {
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int total_blocks = (hidden_dim + warps_per_block - 1) / warps_per_block;

    size_t shmem_swiglu = (dim == 2048) ? (2048 * sizeof(float)) : 0;
    gemv_fused_swiglu_split_kernel<<<total_blocks, threads_per_block, shmem_swiglu, stream>>>(
        (const half*)d_gate_scales,
        (const uint8_t*)d_gate_weights,
        (const half*)d_up_scales,
        (const uint8_t*)d_up_weights,
        d_x,
        d_act_out,
        hidden_dim,
        dim
    );
}

void cuda_rmsnorm(
    float* d_out,
    const float* d_x,
    const void* d_weight_fp16,
    int size,
    float eps,
    cudaStream_t stream
) {
    rmsnorm_gpu_kernel<<<1, 256, 0, stream>>>(d_out, d_x, (const half*)d_weight_fp16, size, eps);
}

void cuda_rope(
    float* d_q,
    float* d_k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float rope_theta,
    cudaStream_t stream
) {
    int half = head_dim / 2;
    int total_pairs = (n_heads + n_kv_heads) * half;
    int threads = 128;
    int blocks = (total_pairs + threads - 1) / threads;
    rope_gpu_kernel<<<blocks, threads, 0, stream>>>(d_q, d_k, n_heads, n_kv_heads, head_dim, pos, rope_theta);
}

void cuda_kv_store(
    void* d_k_cache,           
    void* d_v_cache,           
    const float* d_k,
    const float* d_v,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    cudaStream_t stream
) {
    int total_elements = n_kv_heads * head_dim;
    int threads = 128;
    int blocks = (total_elements + threads - 1) / threads;
    kv_store_gpu_kernel<<<blocks, threads, 0, stream>>>(
        (half*)d_k_cache,      
        (half*)d_v_cache,      
        d_k, 
        d_v, 
        n_kv_heads, 
        head_dim, 
        max_seq_len, 
        pos
    );
}

void cuda_attention_decode(
    float* d_out,
    const float* d_q,
    const void* d_k_cache,     
    const void* d_v_cache,     
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float sqrt_head_dim,
    cudaStream_t stream
) {
    size_t shared_mem = (pos + 1) * sizeof(float);
    attention_decode_gpu_kernel<<<n_heads, head_dim, shared_mem, stream>>>(
        d_out,
        d_q,
        (const half*)d_k_cache,  
        (const half*)d_v_cache,  
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        pos,
        sqrt_head_dim
    );
}

void* cuda_malloc_device(size_t bytes) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void cuda_free_device(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void cuda_device_sync() {
    cudaDeviceSynchronize();
}
void cuda_gemm_w4a16_wmma(
        const void* d_scales,
        const void* d_weights,
        const void* d_x_fp16,
        float* d_y_out,
        int M,
        int K,
        int N,
        cudaStream_t stream
    ) {
        int warps_m = (M + WMMA_M - 1) / WMMA_M;
        int warps_n = (N + WMMA_N - 1) / WMMA_N;
        int total_warps = warps_m * warps_n;

        int threads_per_block = 128;
        int warps_per_block = 4;
        int total_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

        gemm_w4a16_wmma_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
            (const half*)d_scales,
            (const uint8_t*)d_weights,
            (const half*)d_x_fp16,
            d_y_out,
            M,
            K,
            N
        );
    }
void cuda_quantize_x_q8(
    const float* d_x,
    void* d_xq,
    void* d_xmeta,
    int n,
    cudaStream_t stream
) {
    int n_blocks = n / 32;
    int threads = 128;               
    int blocks = (n_blocks + 3) / 4;
    quantize_x_q8_kernel<<<blocks, threads, 0, stream>>>(d_x, (int8_t*)d_xq, (float2*)d_xmeta, n_blocks);
}

void cuda_matvec_q4_dp4a(
    const void* d_scales,
    const void* d_weights,
    const void* d_xq,
    const void* d_xmeta,
    float* d_y,
    const void* d_bias_fp16,
    float* d_residual_add,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int threads = 128;               
    int blocks = (rows + 3) / 4;
    gemv_q4_dp4a_kernel<<<blocks, threads, 0, stream>>>(
        (const half*)d_scales, (const uint8_t*)d_weights,
        (const int8_t*)d_xq, (const float2*)d_xmeta,
        d_y, (const half*)d_bias_fp16, d_residual_add, rows, cols
    );
}

void cuda_swiglu_dp4a(
    const void* d_gate_scales,
    const void* d_gate_weights,
    const void* d_up_scales,
    const void* d_up_weights,
    const void* d_xq,
    const void* d_xmeta,
    float* d_act_out,
    int hidden_dim,
    int dim,
    cudaStream_t stream
) {
    int threads = 128;
    int blocks = (hidden_dim + 3) / 4;
    swiglu_dp4a_kernel<<<blocks, threads, 0, stream>>>(
        (const half*)d_gate_scales, (const uint8_t*)d_gate_weights,
        (const half*)d_up_scales, (const uint8_t*)d_up_weights,
        (const int8_t*)d_xq, (const float2*)d_xmeta,
        d_act_out, hidden_dim, dim
    );
}

void cuda_f32_to_f16(void* d_dst, const float* d_src, int size, cudaStream_t stream) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        f32_to_f16_kernel<<<blocks, threads, 0, stream>>>((half*)d_dst, d_src, size);
    }
}
