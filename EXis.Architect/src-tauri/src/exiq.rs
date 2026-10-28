use memmap2::MmapOptions;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use half::f16;
use rayon::prelude::*;
use tokenizers::Tokenizer;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cuda::{
    is_cuda_available, get_gpu_device_name, CudaBuffer,
    gpu_matvec_q4_fused, gpu_swiglu_gemv, gpu_rmsnorm,
    gpu_rope, gpu_kv_store_layer, gpu_attention_decode_layer,
    gpu_gemm_w4a16_wmma, gpu_f32_to_f16, sync_gpu,
    gpu_quantize_x_q8, gpu_matvec_q4_dp4a, gpu_swiglu_dp4a
};

const EXIQ_MAGIC: u32 = 0x51495845;
const BLOCK_SIZE: usize = 32;
const EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceSelection {
    Auto,
    Cuda,
    Cpu,
}

impl DeviceSelection {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cuda" | "gpu" => Self::Cuda,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub quant_type: u32,
    pub data_offset: usize,
}

pub struct GpuWorkspace {
    pub d_x: CudaBuffer,
    pub d_x_norm: CudaBuffer,
    pub d_q: CudaBuffer,
    pub d_k: CudaBuffer,
    pub d_v: CudaBuffer,
    pub d_k_cache: CudaBuffer,
    pub d_v_cache: CudaBuffer,
    pub d_attn_out: CudaBuffer,
    pub d_act: CudaBuffer,
    pub d_x_fp16: CudaBuffer, 
    pub d_logits: CudaBuffer,
    pub d_xq: CudaBuffer,
    pub d_xmeta: CudaBuffer,
}

impl GpuWorkspace {
    pub fn new(cfg: &ModelConfig, max_seq_len: usize) -> Option<Self> {
        let total_kv_elements = cfg.n_layers * cfg.n_kv_heads * max_seq_len * cfg.head_dim;
        let max_cols = cfg.dim.max(cfg.hidden_dim);
        Some(Self {
            d_x: CudaBuffer::new(cfg.dim * 4)?,
            d_x_norm: CudaBuffer::new(cfg.dim * 4)?,
            d_q: CudaBuffer::new(cfg.n_heads * cfg.head_dim * 4)?,
            d_k: CudaBuffer::new(cfg.n_kv_heads * cfg.head_dim * 4)?,
            d_v: CudaBuffer::new(cfg.n_kv_heads * cfg.head_dim * 4)?,
            d_k_cache: CudaBuffer::new(total_kv_elements * 2)?, 
            d_v_cache: CudaBuffer::new(total_kv_elements * 2)?,
            d_x_fp16: CudaBuffer::new(16 * cfg.dim * 2)?, 
            d_attn_out: CudaBuffer::new(cfg.dim * 4)?,
            d_act: CudaBuffer::new(cfg.hidden_dim * 4)?,
            d_logits: CudaBuffer::new(cfg.vocab_size * 4)?,
            d_xq: CudaBuffer::new(max_cols)?,
            d_xmeta: CudaBuffer::new((max_cols / 32) * 8)?,
        })
    }
}

pub struct ExiqModel {
    pub mmap: memmap2::Mmap,
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub tensors: Vec<TensorDescriptor>,
    pub gpu_weights: Option<CudaBuffer>,
    pub gpu_workspace: Option<GpuWorkspace>,
    pub cpu_kv_cache: Option<KVCacheInt8Cpu>,
    pub cached_tokens: Vec<u32>,
}

#[inline(always)]
fn read_u32(buf: &[u8], offset: usize, name: &str) -> Result<u32, String> {
    if offset + 4 > buf.len() {
        return Err(format!("Файл поврежден: смещение {} для '{}' выходит за границы файла", offset, name));
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    Ok(u32::from_le_bytes(b))
}

#[inline(always)]
fn read_u64(buf: &[u8], offset: usize, name: &str) -> Result<u64, String> {
    if offset + 8 > buf.len() {
        return Err(format!("Файл поврежден: смещение {} для '{}' выходит за границы файла", offset, name));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[offset..offset + 8]);
    Ok(u64::from_le_bytes(b))
}

#[inline(always)]
fn read_f32(buf: &[u8], offset: usize, name: &str) -> Result<f32, String> {
    if offset + 4 > buf.len() {
        return Err(format!("Файл поврежден: смещение {} для '{}' выходит за границы файла", offset, name));
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    Ok(f32::from_le_bytes(b))
}
pub struct Q8Act {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}
 
impl Q8Act {
    pub fn new(max_len: usize) -> Self {
        Self { q: vec![0i8; max_len], s: vec![0.0f32; max_len / BLOCK_SIZE] }
    }
 
    
    pub fn quantize(&mut self, src: &[f32]) {
        debug_assert!(src.len() % BLOCK_SIZE == 0 && src.len() <= self.q.len());
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                quantize_q8_avx2(src, &mut self.q, &mut self.s);
                return;
            }
        }
        quantize_q8_scalar(src, &mut self.q, &mut self.s);
    }
}
 
fn quantize_q8_scalar(src: &[f32], q: &mut [i8], s: &mut [f32]) {
    for (b, blk) in src.chunks_exact(BLOCK_SIZE).enumerate() {
        let max_abs = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let id = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
        s[b] = max_abs / 127.0;
        for (i, &v) in blk.iter().enumerate() {
            q[b * BLOCK_SIZE + i] = (v * id).round() as i8; 
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_q8_avx2(src: &[f32], q: &mut [i8], s: &mut [f32]) {
    let nb = src.len() / BLOCK_SIZE;
    let sign_mask = _mm256_set1_ps(-0.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for b in 0..nb {
        let p = src.as_ptr().add(b * BLOCK_SIZE);
        let v0 = _mm256_loadu_ps(p);
        let v1 = _mm256_loadu_ps(p.add(8));
        let v2 = _mm256_loadu_ps(p.add(16));
        let v3 = _mm256_loadu_ps(p.add(24));
 
        
        let mut m = _mm256_andnot_ps(sign_mask, v0);
        m = _mm256_max_ps(m, _mm256_andnot_ps(sign_mask, v1));
        m = _mm256_max_ps(m, _mm256_andnot_ps(sign_mask, v2));
        m = _mm256_max_ps(m, _mm256_andnot_ps(sign_mask, v3));
        let m4 = _mm_max_ps(_mm256_extractf128_ps::<1>(m), _mm256_castps256_ps128(m));
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let m1 = _mm_max_ss(m2, _mm_movehdup_ps(m2));
        let max_abs = _mm_cvtss_f32(m1);
 
        s[b] = max_abs / 127.0;
        let id = if max_abs != 0.0 { 127.0 / max_abs } else { 0.0 };
        let idv = _mm256_set1_ps(id);
 
        let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, idv));
        let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, idv));
        let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, idv));
        let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, idv));
 
        let p01 = _mm256_packs_epi32(i0, i1);
        let p23 = _mm256_packs_epi32(i2, i3);
        let pk = _mm256_packs_epi16(p01, p23);
        let pk = _mm256_permutevar8x32_epi32(pk, perm); 
        _mm256_storeu_si256(q.as_mut_ptr().add(b * BLOCK_SIZE) as *mut __m256i, pk);
    }
}
 

fn dot_row_q4_q8_scalar(scales: &[u16], w: &[u8], xq: &[i8], xs: &[f32], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let wp = &w[b * 16..(b + 1) * 16];
        let xp = &xq[b * 32..(b + 1) * 32];
        let mut isum = 0i32;
        for i in 0..16 {
            let lo = (wp[i] & 0x0F) as i32 - 8;
            let hi = (wp[i] >> 4) as i32 - 8;
            isum += lo * xp[2 * i] as i32 + hi * xp[2 * i + 1] as i32;
        }
        sum += isum as f32 * f16::from_bits(scales[b]).to_f32() * xs[b];
    }
    sum
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_row_q4_q8_avx2(
    scales: *const u16, 
    wq: *const u8,      
    xq: *const i8,      
    xs: *const f32,     
    nb: usize,
) -> f32 {
    let mask = _mm_set1_epi8(0x0F);
    let eight = _mm256_set1_epi8(8);
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps(); 
 
    macro_rules! block {
        ($b:expr, $acc:ident) => {{
            let bi: usize = $b;
            
            let raw = _mm_loadu_si128(wq.add(bi * 16) as *const __m128i);
            let lo = _mm_and_si128(raw, mask);                          
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask);     
            
            let w = _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm_unpacklo_epi8(lo, hi)),
                _mm_unpackhi_epi8(lo, hi),
            );
            let w = _mm256_sub_epi8(w, eight); 
            let x = _mm256_loadu_si256(xq.add(bi * 32) as *const __m256i);
            
            let ax = _mm256_sign_epi8(w, w);
            let sx = _mm256_sign_epi8(x, w);
            let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sx), ones);
            
            let ws = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(*scales.add(bi) as i32)));
            let d = _mm256_mul_ps(ws, _mm256_broadcast_ss(&*xs.add(bi)));
            $acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), $acc);
        }};
    }
 
    let mut b = 0usize;
    while b + 2 <= nb {
        block!(b, acc0);
        block!(b + 1, acc1);
        b += 2;
    }
    if b < nb {
        block!(b, acc0);
    }
 
    let total = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps::<1>(total);
    let lo = _mm256_castps256_ps128(total);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}
impl ExiqModel {
    pub fn load(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path).map_err(|e| format!("Не удалось открыть файл {:?}: {}", path, e))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        println!("[Exis] 📂 Загрузка модели: {:?} (Размер: {:.2} MB)", path, mmap.len() as f64 / 1024.0 / 1024.0);

        if mmap.len() < 24 {
            return Err(format!("Файл {:?} слишком короткий ({} байт)!", path, mmap.len()).into());
        }

        let magic = read_u32(&mmap, 0, "magic").map_err(|e| e.to_string())?;
        if magic != EXIQ_MAGIC {
            return Err(format!("Файл {:?} не является EXIQ (неверный magic: 0x{:X})!", path, magic).into());
        }

        let version = read_u32(&mmap, 4, "version").map_err(|e| e.to_string())?;
        let tensor_count = read_u64(&mmap, 8, "tensor_count").map_err(|e| e.to_string())?;
        let meta_size = read_u64(&mmap, 16, "meta_size").map_err(|e| e.to_string())? as usize;

        let mut cursor = 24;
        let tok_size = if version >= 3 && mmap.len() >= 32 {
            let s = read_u64(&mmap, 24, "tok_size").map_err(|e| e.to_string())? as usize;
            cursor = 32;
            s
        } else {
            0
        };

        println!("[Exis] ℹ️ Формат EXIQ v{}, Тензоров: {}, Метаданные: {} Б, Токенизатор: {} Б", version, tensor_count, meta_size, tok_size);

        let config = if meta_size >= 32 && cursor + 32 <= mmap.len() {
            let dim = read_u32(&mmap, cursor, "dim").map_err(|e| e.to_string())? as usize;
            let hidden_dim = read_u32(&mmap, cursor + 4, "hidden_dim").map_err(|e| e.to_string())? as usize;
            let n_layers = read_u32(&mmap, cursor + 8, "n_layers").map_err(|e| e.to_string())? as usize;
            let n_heads = read_u32(&mmap, cursor + 12, "n_heads").map_err(|e| e.to_string())? as usize;
            let n_kv_heads = read_u32(&mmap, cursor + 16, "n_kv_heads").map_err(|e| e.to_string())? as usize;
            let head_dim = read_u32(&mmap, cursor + 20, "head_dim").map_err(|e| e.to_string())? as usize;
            let vocab_size = read_u32(&mmap, cursor + 24, "vocab_size").map_err(|e| e.to_string())? as usize;
            let rope_theta = read_f32(&mmap, cursor + 28, "rope_theta").map_err(|e| e.to_string())?;
            cursor += meta_size;

            ModelConfig { dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, rope_theta }
        } else {
            cursor += meta_size;
            ModelConfig {
                dim: 2048, hidden_dim: 5632, n_layers: 22,
                n_heads: 32, n_kv_heads: 4, head_dim: 64,
                vocab_size: 32000, rope_theta: 10000.0
            }
        };

        let tokenizer = if tok_size > 0 && cursor + tok_size <= mmap.len() {
            let tok_bytes = &mmap[cursor..cursor + tok_size];
            cursor += tok_size;
            Tokenizer::from_bytes(tok_bytes).map_err(|e| format!("Ошибка парсинга токенизатора: {}", e))?
        } else {
            let tok_path = if PathBuf::from("tokenizer.json").exists() {
                PathBuf::from("tokenizer.json")
            } else {
                path.with_file_name("tokenizer.json")
            };
            Tokenizer::from_file(&tok_path).map_err(|e| format!("Не найден tokenizer.json: {}", e))?
        };

        let mut tensors = Vec::new();
        for i in 0..tensor_count {
            let name_len = read_u32(&mmap, cursor, &format!("tensor_{}_name_len", i)).map_err(|e| e.to_string())? as usize;
            cursor += 4;

            let name = String::from_utf8(mmap[cursor..cursor + name_len].to_vec())
                .map_err(|e| format!("Неверный UTF-8 в имени тензора {}: {}", i, e))?;
            cursor += name_len;

            let n_dims = read_u32(&mmap, cursor, "n_dims").map_err(|e| e.to_string())? as usize;
            cursor += 4;

            let mut shape = Vec::new();
            for d in 0..n_dims {
                shape.push(read_u64(&mmap, cursor, &format!("dim_{}", d)).map_err(|e| e.to_string())? as usize);
                cursor += 8;
            }

            let quant_type = read_u32(&mmap, cursor, "quant_type").map_err(|e| e.to_string())?;
            cursor += 4;

            let data_offset = read_u64(&mmap, cursor, "data_offset").map_err(|e| e.to_string())? as usize;
            cursor += 8;

            tensors.push(TensorDescriptor { name, shape, quant_type, data_offset });
        }

        let (gpu_weights, mut gpu_workspace) = if is_cuda_available() {
            let weights = CudaBuffer::from_slice(&mmap);
            let mut ws = GpuWorkspace::new(&config, 2048);
            
            
            if let (Some(ref w), Some(ref mut workspace)) = (&weights, &mut ws) {
                if let Some(t_q) = tensors.iter().find(|t| t.name.ends_with("self_attn.q_proj.weight")) {
                    let start = std::time::Instant::now();
                    crate::cuda::gpu_gemm_w4a16_wmma(
                        w.ptr,
                        t_q.data_offset,
                        &workspace.d_x_fp16,
                        &mut workspace.d_q,
                        16,          
                        config.dim,  
                        t_q.shape[0],
                    );
                    crate::cuda::sync_gpu();
                    println!(
                        "[Tensor Cores] ⚡ Боевой прогон на реальном слое Q_proj (16 токенов x {}): {:?}",
                        config.dim, start.elapsed()
                    );
                }
            }

            (weights, ws)
        } else {
            (None, None)
        };

        Ok(Self {
            mmap, 
            config, 
            tokenizer, 
            tensors, 
            gpu_weights, 
            gpu_workspace, 
            cpu_kv_cache: None,
            cached_tokens: Vec::new() 
        })
    }

    #[inline(always)]
    pub fn get_tensor(&self, name: &str) -> &TensorDescriptor {
        self.tensors.iter().find(|t| t.name == name)
            .unwrap_or_else(|| panic!("Тензор {} не найден!", name))
    }

    #[inline(always)]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    #[inline(always)]
    pub fn get_fp16_data(&self, tensor: &TensorDescriptor) -> &[f16] {
        let num_elements = tensor.shape.iter().product();
        unsafe {
            let ptr = self.mmap.as_ptr().add(tensor.data_offset) as *const f16;
            std::slice::from_raw_parts(ptr, num_elements)
        }
    }

    pub fn matvec_q4_cpu(&self, tensor: &TensorDescriptor, xa: &Q8Act, y: &mut [f32]) {
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        let nb = cols / BLOCK_SIZE;
        let total_blocks = rows * nb;
 
        let scales: &[u16] = unsafe {
            let ptr = self.mmap.as_ptr().add(tensor.data_offset) as *const u16;
            std::slice::from_raw_parts(ptr, total_blocks)
        };
        let weights: &[u8] = unsafe {
            let ptr = self.mmap.as_ptr().add(tensor.data_offset + total_blocks * 2);
            std::slice::from_raw_parts(ptr, total_blocks * 16)
        };
        let xq = &xa.q[..cols];
        let xs = &xa.s[..nb];
 
        
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
 
        y.par_iter_mut().enumerate().for_each(|(row, out)| {
            let s = &scales[row * nb..(row + 1) * nb];
            let w = &weights[row * nb * 16..(row + 1) * nb * 16];
 
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_avx2 {
                    *out = dot_row_q4_q8_avx2(s.as_ptr(), w.as_ptr(), xq.as_ptr(), xs.as_ptr(), nb);
                    return;
                }
            }
 
            *out = dot_row_q4_q8_scalar(s, w, xq, xs, nb);
        });
    }
    
}

#[inline(always)]
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f16]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv_std = 1.0 / (ss + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_std * weight[i].to_f32();
    }
}

fn rope_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect()
}
 

#[inline(always)]
fn apply_rope(vec: &mut [f32], n_heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        let (a, b) = head.split_at_mut(half);
        for i in 0..half {
            let (v0, v1) = (a[i], b[i]);
            a[i] = v0 * cos[i] - v1 * sin[i];
            b[i] = v0 * sin[i] + v1 * cos[i];
        }
    }
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn quantize_to_i8(src: &[f32], dst_bytes: &mut [i8]) -> f32 {
    let mut max_val = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > max_val { max_val = a; }
    }
    let scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;
    for (i, &v) in src.iter().enumerate() {
        dst_bytes[i] = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
    }
    scale
}

fn sample_token(logits: &mut [f32], generated_tokens: &[u32], temperature: f32, top_p: f32, repeat_penalty: f32) -> u32 {
    if repeat_penalty > 1.0 {
        for &tok in generated_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= repeat_penalty;
                } else {
                    logits[idx] *= repeat_penalty;
                }
            }
        }
    }

    if temperature <= 0.01 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap();
    }

    for l in logits.iter_mut() { *l /= temperature; }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max_val).exp();
        sum += *l;
    }
    let inv_sum = 1.0 / sum;
    for l in logits.iter_mut() { *l *= inv_sum; }

    let k = 64.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(k);
    
    
    let mut all_indices: Vec<usize> = (0..logits.len()).collect();
    all_indices.select_nth_unstable_by(k, |&a, &b| logits[b].total_cmp(&logits[a]));
    
    for &idx in &all_indices[..k] {
        indexed.push((idx, logits[idx]));
    }
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = (i + 1).min(indexed.len());
            break;
        }
    }

    let rng_val = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as f32 % 10000.0) / 10000.0 * cumsum;

    let mut acc = 0.0f32;
    for &(token_id, p) in &indexed[..cutoff_idx] {
        acc += p;
        if acc >= rng_val {
            return token_id as u32;
        }
    }
    indexed[0].0 as u32
}

pub struct KVCacheInt8Cpu {
    k_bytes: Vec<Vec<i8>>,
    k_scales: Vec<Vec<f32>>,
    v_bytes: Vec<Vec<i8>>,
    v_scales: Vec<Vec<f32>>,
}

pub fn generate_stream<F>(
    model: &mut ExiqModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    num_threads: usize,
    device_selection: DeviceSelection,
    is_new_chat: bool,
    context_size: usize, 
    is_aborted: Arc<AtomicBool>,
    on_token: F,
) -> Result<(), String>
where
    F: Fn(String) + Send + Sync + 'static,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;

    pool.install(|| {
        let cfg = model.config;
        let encoding = model.tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
        let prompt_tokens = encoding.get_ids().to_vec();

        let max_seq_len = context_size; 
        let layer_stride = cfg.n_kv_heads * max_seq_len * cfg.head_dim;

        
        let needed_kv_bytes = cfg.n_layers * cfg.n_kv_heads * max_seq_len * cfg.head_dim * 2;
        if let Some(ref ws) = model.gpu_workspace {
            if ws.d_k_cache.size != needed_kv_bytes {
                model.gpu_workspace = GpuWorkspace::new(&cfg, max_seq_len);
                model.cached_tokens.clear();
            }
        }

        
        let mut prefix_len = 0;
        if !is_new_chat {
            while prefix_len < prompt_tokens.len()
                && prefix_len < model.cached_tokens.len()
                && prompt_tokens[prefix_len] == model.cached_tokens[prefix_len]
            {
                prefix_len += 1;
            }
        }
        
        if prefix_len + (prompt_tokens.len() - prefix_len) + max_tokens >= max_seq_len {
            prefix_len = 0;
            model.cached_tokens.clear();
        } else {
            model.cached_tokens.truncate(prefix_len);
        }

        let mut ws = model.gpu_workspace.take();
        let use_cuda = match device_selection {
            DeviceSelection::Cuda => ws.is_some(),
            DeviceSelection::Cpu => false,
            DeviceSelection::Auto => ws.is_some(),
        };

        let d_model_ptr = model.gpu_weights.as_ref().map(|w| w.ptr).unwrap_or(std::ptr::null_mut());

        let mut x = vec![0.0f32; cfg.dim];
        let mut x_norm = vec![0.0f32; cfg.dim];
        let mut q = vec![0.0f32; cfg.n_heads * cfg.head_dim];
        let mut k = vec![0.0f32; cfg.n_kv_heads * cfg.head_dim];
        let mut v = vec![0.0f32; cfg.n_kv_heads * cfg.head_dim];
        let mut attn_out = vec![0.0f32; cfg.dim];
        let mut o_proj = vec![0.0f32; cfg.dim];
        let mut gate = vec![0.0f32; cfg.hidden_dim];
        let mut up = vec![0.0f32; cfg.hidden_dim];
        let mut act = vec![0.0f32; cfg.hidden_dim];
        let mut mlp_out = vec![0.0f32; cfg.dim];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        let mut scores = vec![0.0f32; 2048];

        if is_new_chat {
            model.cpu_kv_cache = None;
        }

        let mut cpu_kv_cache = if !use_cuda {
            Some(model.cpu_kv_cache.take().unwrap_or_else(|| {
                KVCacheInt8Cpu {
                    k_bytes: vec![vec![0i8; cfg.n_kv_heads * max_seq_len * cfg.head_dim]; cfg.n_layers],
                    k_scales: vec![vec![1.0f32; cfg.n_kv_heads * max_seq_len]; cfg.n_layers],
                    v_bytes: vec![vec![0i8; cfg.n_kv_heads * max_seq_len * cfg.head_dim]; cfg.n_layers],
                    v_scales: vec![vec![1.0f32; cfg.n_kv_heads * max_seq_len]; cfg.n_layers],
                }
            }))
        } else {
            None
        };

    
        let embed_tensor = model.get_tensor("model.embed_tokens.weight");
        let embed_data = model.get_fp16_data(embed_tensor);
        let final_norm_tensor = model.get_tensor("model.norm.weight");
        let final_norm = model.get_fp16_data(final_norm_tensor);
        let lm_head = model.get_tensor("lm_head.weight");

        
        let prompt_start_pos = prefix_len;
        let prompt_end_pos = prompt_tokens.len();
        let target_max_pos = (prompt_end_pos + max_tokens).min(max_seq_len);

        let mut pos = prompt_start_pos;
        let mut current_token = if prompt_start_pos < prompt_tokens.len() {
            prompt_tokens[prompt_start_pos]
        } else {
            prompt_tokens[0]
        };

        let sqrt_head_dim = (cfg.head_dim as f32).sqrt();
        let mut generated_history = Vec::new();
        let mut prev_text = String::new(); 
        let mut xa = Q8Act::new(cfg.dim.max(cfg.hidden_dim));
        let inv_freq = rope_inv_freq(cfg.head_dim, cfg.rope_theta);
        let mut rope_cos = vec![0.0f32; cfg.head_dim / 2];
        let mut rope_sin = vec![0.0f32; cfg.head_dim / 2];
        while pos < target_max_pos {
            if is_aborted.load(Ordering::Relaxed) {
                break;
            }

            let is_prefill = pos + 1 < prompt_end_pos;

            let emb_offset = (current_token as usize) * cfg.dim;
            for i in 0..cfg.dim {
                x[i] = embed_data[emb_offset + i].to_f32();
            }

            if use_cuda {
                let gpu_ws = ws.as_mut().unwrap();
                gpu_ws.d_x.copy_from_host(&x);

                for l in 0..cfg.n_layers {
                    let norm1_tensor = model.get_tensor(&format!("model.layers.{}.input_layernorm.weight", l));
                    let t_q = model.get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", l));
                    let t_k = model.get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", l));
                    let t_v = model.get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", l));
                    let t_o = model.get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", l));
                    let norm2_tensor = model.get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", l));
                    let t_gate = model.get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", l));
                    let t_up = model.get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", l));
                    let t_down = model.get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", l));

                    
                    let norm1_ptr = unsafe { (d_model_ptr as *const u8).add(norm1_tensor.data_offset) as *const std::ffi::c_void };
                    gpu_rmsnorm(&mut gpu_ws.d_x_norm, &gpu_ws.d_x, norm1_ptr, cfg.dim, EPS);
                    
                    gpu_quantize_x_q8(&gpu_ws.d_x_norm, &mut gpu_ws.d_xq, &mut gpu_ws.d_xmeta, cfg.dim);

                    
                    let b_q_ptr = model.find_tensor(&format!("model.layers.{}.self_attn.q_proj.bias", l))
                        .map(|t| unsafe { (d_model_ptr as *const u8).add(t.data_offset) as *const std::ffi::c_void });
                    gpu_matvec_q4_dp4a(d_model_ptr, t_q.data_offset, &gpu_ws.d_xq, &gpu_ws.d_xmeta, Some(&mut gpu_ws.d_q), b_q_ptr, None, t_q.shape[0], t_q.shape[1]);

                    let b_k_ptr = model.find_tensor(&format!("model.layers.{}.self_attn.k_proj.bias", l))
                        .map(|t| unsafe { (d_model_ptr as *const u8).add(t.data_offset) as *const std::ffi::c_void });
                    gpu_matvec_q4_dp4a(d_model_ptr, t_k.data_offset, &gpu_ws.d_xq, &gpu_ws.d_xmeta, Some(&mut gpu_ws.d_k), b_k_ptr, None, t_k.shape[0], t_k.shape[1]);

                    let b_v_ptr = model.find_tensor(&format!("model.layers.{}.self_attn.v_proj.bias", l))
                        .map(|t| unsafe { (d_model_ptr as *const u8).add(t.data_offset) as *const std::ffi::c_void });
                    gpu_matvec_q4_dp4a(d_model_ptr, t_v.data_offset, &gpu_ws.d_xq, &gpu_ws.d_xmeta, Some(&mut gpu_ws.d_v), b_v_ptr, None, t_v.shape[0], t_v.shape[1]);

                    
                    gpu_rope(&mut gpu_ws.d_q, &mut gpu_ws.d_k, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, pos, cfg.rope_theta);

                    
                    gpu_kv_store_layer(&mut gpu_ws.d_k_cache, &mut gpu_ws.d_v_cache, l, layer_stride, &gpu_ws.d_k, &gpu_ws.d_v, cfg.n_kv_heads, cfg.head_dim, max_seq_len, pos);

                    
                    gpu_attention_decode_layer(&mut gpu_ws.d_attn_out, &gpu_ws.d_q, &gpu_ws.d_k_cache, &gpu_ws.d_v_cache, l, layer_stride, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, max_seq_len, pos, sqrt_head_dim);

                    
                    gpu_quantize_x_q8(&gpu_ws.d_attn_out, &mut gpu_ws.d_xq, &mut gpu_ws.d_xmeta, t_o.shape[1]);
                    gpu_matvec_q4_dp4a(d_model_ptr, t_o.data_offset, &gpu_ws.d_xq, &gpu_ws.d_xmeta, None, None, Some(&mut gpu_ws.d_x), t_o.shape[0], t_o.shape[1]);

                    
                    let norm2_ptr = unsafe { (d_model_ptr as *const u8).add(norm2_tensor.data_offset) as *const std::ffi::c_void };
                    gpu_rmsnorm(&mut gpu_ws.d_x_norm, &gpu_ws.d_x, norm2_ptr, cfg.dim, EPS);
                    gpu_quantize_x_q8(&gpu_ws.d_x_norm, &mut gpu_ws.d_xq, &mut gpu_ws.d_xmeta, cfg.dim);

                    
                    gpu_swiglu_dp4a(d_model_ptr, t_gate.data_offset, t_up.data_offset, &gpu_ws.d_xq, &gpu_ws.d_xmeta, &mut gpu_ws.d_act, cfg.hidden_dim, cfg.dim);

                    
                    gpu_quantize_x_q8(&gpu_ws.d_act, &mut gpu_ws.d_xq, &mut gpu_ws.d_xmeta, cfg.hidden_dim);
                    gpu_matvec_q4_dp4a(d_model_ptr, t_down.data_offset, &gpu_ws.d_xq, &gpu_ws.d_xmeta, None, None, Some(&mut gpu_ws.d_x), t_down.shape[0], t_down.shape[1]);
                }

                if is_prefill {
                    pos += 1;
                    if pos < prompt_tokens.len() {
                        current_token = prompt_tokens[pos];
                    }
                    continue;
                }

                let final_norm_ptr = unsafe { (d_model_ptr as *const u8).add(final_norm_tensor.data_offset) as *const std::ffi::c_void };
                gpu_rmsnorm(&mut gpu_ws.d_x_norm, &gpu_ws.d_x, final_norm_ptr, cfg.dim, EPS);
                gpu_quantize_x_q8(&gpu_ws.d_x_norm, &mut gpu_ws.d_xq, &mut gpu_ws.d_xmeta, cfg.dim);
                
                gpu_matvec_q4_dp4a(
                    d_model_ptr, lm_head.data_offset, &gpu_ws.d_xq, &gpu_ws.d_xmeta,
                    Some(&mut gpu_ws.d_logits), None, None,
                    lm_head.shape[0], lm_head.shape[1],
                );
                gpu_ws.d_logits.copy_to_host(&mut logits);

            } else {
                let kv_cache = cpu_kv_cache.as_mut().unwrap();
                for i in 0..cfg.head_dim / 2 {
                    let (s, c) = (pos as f32 * inv_freq[i]).sin_cos();
                    rope_sin[i] = s;
                    rope_cos[i] = c;
                }
                for l in 0..cfg.n_layers {
                    let norm1_tensor = model.get_tensor(&format!("model.layers.{}.input_layernorm.weight", l));
                    let input_norm_w = model.get_fp16_data(norm1_tensor);
                    rmsnorm(&mut x_norm, &x, input_norm_w);
                    xa.quantize(&x_norm); 
                    let t_q = model.get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", l));
                    let t_k = model.get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", l));
                    let t_v = model.get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", l));

                    model.matvec_q4_cpu(t_q, &xa, &mut q);
                    model.matvec_q4_cpu(t_k, &xa, &mut k);
                    model.matvec_q4_cpu(t_v, &xa, &mut v);

                    
                    if let Some(b_q) = model.find_tensor(&format!("model.layers.{}.self_attn.q_proj.bias", l)) {
                        let b_data = model.get_fp16_data(b_q);
                        for i in 0..q.len() { q[i] += b_data[i].to_f32(); }
                    }
                    if let Some(b_k) = model.find_tensor(&format!("model.layers.{}.self_attn.k_proj.bias", l)) {
                        let b_data = model.get_fp16_data(b_k);
                        for i in 0..k.len() { k[i] += b_data[i].to_f32(); }
                    }
                    if let Some(b_v) = model.find_tensor(&format!("model.layers.{}.self_attn.v_proj.bias", l)) {
                        let b_data = model.get_fp16_data(b_v);
                        for i in 0..v.len() { v[i] += b_data[i].to_f32(); }
                    }

                    apply_rope(&mut q, cfg.n_heads, cfg.head_dim, &rope_cos, &rope_sin);
                    apply_rope(&mut k, cfg.n_kv_heads, cfg.head_dim, &rope_cos, &rope_sin);


                    for kv_h in 0..cfg.n_kv_heads {
                        let cache_byte_offset = kv_h * (max_seq_len * cfg.head_dim) + pos * cfg.head_dim;
                        let scale_offset = kv_h * max_seq_len + pos;
                        let k_src = &k[kv_h * cfg.head_dim .. (kv_h + 1) * cfg.head_dim];
                        let v_src = &v[kv_h * cfg.head_dim .. (kv_h + 1) * cfg.head_dim];
                        let k_dst = &mut kv_cache.k_bytes[l][cache_byte_offset .. cache_byte_offset + cfg.head_dim];
                        let v_dst = &mut kv_cache.v_bytes[l][cache_byte_offset .. cache_byte_offset + cfg.head_dim];
                        kv_cache.k_scales[l][scale_offset] = quantize_to_i8(k_src, k_dst);
                        kv_cache.v_scales[l][scale_offset] = quantize_to_i8(v_src, v_dst);
                    }

                    let kv_mul = cfg.n_heads / cfg.n_kv_heads;
                    attn_out.par_chunks_mut(cfg.head_dim).enumerate().for_each(|(h, head_out)| {
                        let kv_h = h / kv_mul;
                        let q_head = &q[h * cfg.head_dim .. (h + 1) * cfg.head_dim];
                        let head_kv_byte_base = kv_h * (max_seq_len * cfg.head_dim);
                        let head_kv_scale_base = kv_h * max_seq_len;
                        let mut local_scores = [0.0f32; 2048]; 
                        
                        let mut max_s = f32::NEG_INFINITY;
                        for p in 0..=pos {
                            let p_byte_start = head_kv_byte_base + p * cfg.head_dim;
                            let p_byte_end = p_byte_start + cfg.head_dim;
                            let k_ptr = &kv_cache.k_bytes[l][p_byte_start..p_byte_end];
                            let k_scale = kv_cache.k_scales[l][head_kv_scale_base + p];
                            let mut dot = 0.0f32;
                            for d in 0..cfg.head_dim {
                                dot += q_head[d] * (k_ptr[d] as f32);
                            }
                            let s = (dot * k_scale) / sqrt_head_dim;
                            local_scores[p] = s;
                            if s > max_s { max_s = s; }
                        }

                        let mut sum_s = 0.0f32;
                        for p in 0..=pos {
                            local_scores[p] = (local_scores[p] - max_s).exp();
                            sum_s += local_scores[p];
                        }
                        let inv_sum = 1.0 / sum_s;

                        head_out.fill(0.0);
                        for p in 0..=pos {
                            let p_byte_start = head_kv_byte_base + p * cfg.head_dim;
                            let p_byte_end = p_byte_start + cfg.head_dim;
                            let v_ptr = &kv_cache.v_bytes[l][p_byte_start..p_byte_end];
                            let v_scale = kv_cache.v_scales[l][head_kv_scale_base + p];
                            let eff_weight = local_scores[p] * inv_sum * v_scale;
                            for d in 0..cfg.head_dim {
                                head_out[d] += eff_weight * (v_ptr[d] as f32);
                            }
                        }
                    });

                    let t_o = model.get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", l));
                    xa.quantize(&attn_out);
                    model.matvec_q4_cpu(t_o, &xa, &mut o_proj);
                    for i in 0..cfg.dim { x[i] += o_proj[i]; }

                     let norm2_tensor = model.get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", l));
                    let post_norm_w = model.get_fp16_data(norm2_tensor);
                    rmsnorm(&mut x_norm, &x, post_norm_w);

                    
                    let t_gate = model.get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", l));
                    let t_up = model.get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", l));

                    
                    xa.quantize(&x_norm);

                    
                    model.matvec_q4_cpu(t_gate, &xa, &mut gate);
                    model.matvec_q4_cpu(t_up, &xa, &mut up);

                    
                    for i in 0..cfg.hidden_dim {
                        act[i] = silu(gate[i]) * up[i];
                    }

                    
                    let t_down = model.get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", l));
                    xa.quantize(&act);
                    model.matvec_q4_cpu(t_down, &xa, &mut mlp_out);
                    for i in 0..cfg.dim { x[i] += mlp_out[i]; }
                }

                if is_prefill {
                    pos += 1;
                    if pos < prompt_tokens.len() {
                        current_token = prompt_tokens[pos];
                    }
                    continue;
                }

                rmsnorm(&mut x_norm, &x, final_norm);
                xa.quantize(&x_norm);
                model.matvec_q4_cpu(lm_head, &xa, &mut logits);
            }

            let next_token = sample_token(&mut logits, &generated_history, temperature, top_p, repeat_penalty);
            pos += 1;
            current_token = next_token;
            generated_history.push(next_token);

            if next_token == 2 || next_token == 151645 || next_token == 151643 {
                break;
            }

            if let Ok(full_text) = model.tokenizer.decode(&generated_history, false) {
                if full_text.len() > prev_text.len() {
                    
                    
                    let delta = if full_text.starts_with(&prev_text) {
                        full_text[prev_text.len()..].to_string()
                    } else {
                        let prev_chars = prev_text.chars().count();
                        full_text.chars().skip(prev_chars).collect::<String>()
                    };

                    if !delta.is_empty() && !delta.contains('\u{FFFD}') {
                        prev_text = full_text;
                        if delta.contains("<|im_end|>") || delta.contains("<|endoftext|>") || delta.contains("</s>") {
                            break;
                        }
                        on_token(delta);
                    }
                }
            }

        }

        model.cached_tokens = prompt_tokens;

        let n = generated_history.len().saturating_sub(1);
        model.cached_tokens.extend_from_slice(&generated_history[..n]);
        model.gpu_workspace = ws;
        model.cpu_kv_cache = cpu_kv_cache;

        Ok(())
    })
}