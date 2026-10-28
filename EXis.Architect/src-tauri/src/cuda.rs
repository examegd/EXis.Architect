use std::ffi::{c_char, c_void, CStr};
use half::f16;
extern "C" {
    fn cuda_check_available() -> bool;
    fn cuda_get_gpu_name(name_buf: *mut c_char, max_len: i32);
    fn cuda_warmup();
    fn cuda_matvec_q4_fused(
        d_scales: *const c_void, 
        d_weights: *const c_void, 
        d_x: *const f32, 
        d_y: *mut f32, 
        d_bias_fp16: *const c_void,
        d_residual_add: *mut f32,
        rows: i32, 
        cols: i32, 
        stream: *mut c_void
    );
    fn cuda_swiglu_gemv(
        d_gate_scales: *const c_void,
        d_gate_weights: *const c_void,
        d_up_scales: *const c_void,
        d_up_weights: *const c_void,
        d_x: *const f32,
        d_act_out: *mut f32,
        hidden_dim: i32,
        dim: i32,
        stream: *mut c_void
    );
        fn cuda_quantize_x_q8(
        d_x: *const f32,
        d_xq: *mut c_void,
        d_xmeta: *mut c_void,
        n: i32,
        stream: *mut c_void,
    );
    fn cuda_matvec_q4_dp4a(
        d_scales: *const c_void,
        d_weights: *const c_void,
        d_xq: *const c_void,
        d_xmeta: *const c_void,
        d_y: *mut f32,
        d_bias_fp16: *const c_void,
        d_residual_add: *mut f32,
        rows: i32,
        cols: i32,
        stream: *mut c_void,
    );
    fn cuda_swiglu_dp4a(
        d_gate_scales: *const c_void,
        d_gate_weights: *const c_void,
        d_up_scales: *const c_void,
        d_up_weights: *const c_void,
        d_xq: *const c_void,
        d_xmeta: *const c_void,
        d_act_out: *mut f32,
        hidden_dim: i32,
        dim: i32,
        stream: *mut c_void,
    );
    fn cuda_rmsnorm(d_out: *mut f32, d_x: *const f32, d_weight_fp16: *const c_void, size: i32, eps: f32, stream: *mut c_void);
    fn cuda_rope(d_q: *mut f32, d_k: *mut f32, n_heads: i32, n_kv_heads: i32, head_dim: i32, pos: i32, rope_theta: f32, stream: *mut c_void);
    fn cuda_kv_store(d_k_cache: *mut c_void, d_v_cache: *mut c_void, d_k: *const f32, d_v: *const f32, n_kv_heads: i32, head_dim: i32, max_seq_len: i32, pos: i32, stream: *mut c_void);
    fn cuda_attention_decode(d_out: *mut f32, d_q: *const f32, d_k_cache: *const c_void, d_v_cache: *const c_void, n_heads: i32, n_kv_heads: i32, head_dim: i32, max_seq_len: i32, pos: i32, sqrt_head_dim: f32, stream: *mut c_void);
    fn cuda_gemm_w4a16_wmma(
        d_scales: *const c_void,
        d_weights: *const c_void,
        d_x_fp16: *const c_void,
        d_y_out: *mut f32,
        m: i32,
        k: i32,
        n: i32,
        stream: *mut c_void,
    );
    fn cuda_f32_to_f16(d_dst: *mut c_void, d_src: *const f32, size: i32, stream: *mut c_void);
    fn cuda_malloc_device(bytes: usize) -> *mut c_void;
    fn cuda_free_device(ptr: *mut c_void);
    fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize);
    fn cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize);
    fn cuda_device_sync();
}
pub fn gpu_f32_to_f16(d_dst: &mut CudaBuffer, d_src: &CudaBuffer, size: usize) {
    unsafe {
        cuda_f32_to_f16(d_dst.ptr, d_src.ptr as *const f32, size as i32, std::ptr::null_mut());
    }
}
pub fn is_cuda_available() -> bool {
    unsafe { cuda_check_available() }
}

pub fn get_gpu_device_name() -> String {
    let mut buf = [0u8; 256];
    unsafe {
        cuda_get_gpu_name(buf.as_mut_ptr() as *mut c_char, 256);
        CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned()
    }
}

pub fn warmup_cuda() {
    if is_cuda_available() {
        unsafe { cuda_warmup(); }
    }
}

pub struct CudaBuffer {
    pub ptr: *mut c_void,
    pub size: usize,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    pub fn new(size: usize) -> Option<Self> {
        let ptr = unsafe { cuda_malloc_device(size) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, size })
        }
    }

    pub fn from_slice<T>(data: &[T]) -> Option<Self> {
        let size = std::mem::size_of_val(data);
        let buf = Self::new(size)?;
        unsafe {
            cuda_memcpy_h2d(buf.ptr, data.as_ptr() as *const c_void, size);
        }
        Some(buf)
    }

    pub fn copy_to_host<T>(&self, dst: &mut [T]) {
        let bytes = std::mem::size_of_val(dst);
        assert!(bytes <= self.size);
        unsafe {
            cuda_memcpy_d2h(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes);
        }
    }

    pub fn copy_from_host<T>(&self, src: &[T]) {
        let bytes = std::mem::size_of_val(src);
        assert!(bytes <= self.size);
        unsafe {
            cuda_memcpy_h2d(self.ptr, src.as_ptr() as *const c_void, bytes);
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_free_device(self.ptr); }
        }
    }
}
pub fn gpu_quantize_x_q8(d_x: &CudaBuffer, d_xq: &mut CudaBuffer, d_xmeta: &mut CudaBuffer, n: usize) {
    unsafe {
        cuda_quantize_x_q8(
            d_x.ptr as *const f32,
            d_xq.ptr,
            d_xmeta.ptr,
            n as i32,
            std::ptr::null_mut(),
        );
    }
}
 
pub fn gpu_matvec_q4_dp4a(
    d_model_ptr: *const c_void,
    offset: usize,
    d_xq: &CudaBuffer,
    d_xmeta: &CudaBuffer,
    d_y: Option<&mut CudaBuffer>,
    d_bias_ptr: Option<*const c_void>,
    d_residual_add: Option<&mut CudaBuffer>,
    rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / 32;
    let total_blocks = rows * blocks_per_row;
    let scales_size = total_blocks * 2;
 
    unsafe {
        let scales_ptr = (d_model_ptr as *const u8).add(offset) as *const c_void;
        let weights_ptr = (d_model_ptr as *const u8).add(offset + scales_size) as *const c_void;
        let bias_ptr = d_bias_ptr.unwrap_or(std::ptr::null());
        let res_ptr = d_residual_add.map(|b| b.ptr as *mut f32).unwrap_or(std::ptr::null_mut());
        let y_ptr = d_y.map(|b| b.ptr as *mut f32).unwrap_or(std::ptr::null_mut());
 
        cuda_matvec_q4_dp4a(
            scales_ptr,
            weights_ptr,
            d_xq.ptr as *const c_void,
            d_xmeta.ptr as *const c_void,
            y_ptr,
            bias_ptr,
            res_ptr,
            rows as i32,
            cols as i32,
            std::ptr::null_mut(),
        );
    }
}
 
pub fn gpu_swiglu_dp4a(
    d_model_ptr: *const c_void,
    gate_offset: usize,
    up_offset: usize,
    d_xq: &CudaBuffer,
    d_xmeta: &CudaBuffer,
    d_act_out: &mut CudaBuffer,
    hidden_dim: usize,
    dim: usize,
) {
    let blocks_per_row = dim / 32;
    let total_blocks = hidden_dim * blocks_per_row;
    let scales_size = total_blocks * 2;
 
    unsafe {
        let g_scales = (d_model_ptr as *const u8).add(gate_offset) as *const c_void;
        let g_weights = (d_model_ptr as *const u8).add(gate_offset + scales_size) as *const c_void;
        let u_scales = (d_model_ptr as *const u8).add(up_offset) as *const c_void;
        let u_weights = (d_model_ptr as *const u8).add(up_offset + scales_size) as *const c_void;
 
        cuda_swiglu_dp4a(
            g_scales,
            g_weights,
            u_scales,
            u_weights,
            d_xq.ptr as *const c_void,
            d_xmeta.ptr as *const c_void,
            d_act_out.ptr as *mut f32,
            hidden_dim as i32,
            dim as i32,
            std::ptr::null_mut(),
        );
    }
}
pub fn gpu_matvec_q4_fused(
    d_model_ptr: *const c_void,
    offset: usize,
    d_x: &CudaBuffer,
    d_y: Option<&mut CudaBuffer>,
    d_bias_ptr: Option<*const c_void>,
    d_residual_add: Option<&mut CudaBuffer>,
    rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / 32;
    let total_blocks = rows * blocks_per_row;
    let scales_size = total_blocks * 2;

    unsafe {
        let scales_ptr = (d_model_ptr as *const u8).add(offset) as *const c_void;
        let weights_ptr = (d_model_ptr as *const u8).add(offset + scales_size) as *const c_void;
        let bias_ptr = d_bias_ptr.unwrap_or(std::ptr::null());
        let res_ptr = d_residual_add.map(|b| b.ptr as *mut f32).unwrap_or(std::ptr::null_mut());
        let y_ptr = d_y.map(|b| b.ptr as *mut f32).unwrap_or(std::ptr::null_mut());

        cuda_matvec_q4_fused(
            scales_ptr,
            weights_ptr,
            d_x.ptr as *const f32,
            y_ptr,
            bias_ptr,
            res_ptr,
            rows as i32,
            cols as i32,
            std::ptr::null_mut(),
        );
    }
}

pub fn gpu_swiglu_gemv(
    d_model_ptr: *const c_void,
    gate_offset: usize,
    up_offset: usize,
    d_x: &CudaBuffer,
    d_act_out: &mut CudaBuffer,
    hidden_dim: usize,
    dim: usize,
) {
    let blocks_per_row = dim / 32;
    let total_blocks = hidden_dim * blocks_per_row;
    let scales_size = total_blocks * 2;

    unsafe {
        let g_scales = (d_model_ptr as *const u8).add(gate_offset) as *const c_void;
        let g_weights = (d_model_ptr as *const u8).add(gate_offset + scales_size) as *const c_void;
        let u_scales = (d_model_ptr as *const u8).add(up_offset) as *const c_void;
        let u_weights = (d_model_ptr as *const u8).add(up_offset + scales_size) as *const c_void;

        cuda_swiglu_gemv(
            g_scales,
            g_weights,
            u_scales,
            u_weights,
            d_x.ptr as *const f32,
            d_act_out.ptr as *mut f32,
            hidden_dim as i32,
            dim as i32,
            std::ptr::null_mut(),
        );
    }
    
}

pub fn gpu_rmsnorm(d_out: &mut CudaBuffer, d_x: &CudaBuffer, d_weight_fp16: *const c_void, size: usize, eps: f32) {
    unsafe {
        cuda_rmsnorm(d_out.ptr as *mut f32, d_x.ptr as *const f32, d_weight_fp16, size as i32, eps, std::ptr::null_mut());
    }
}

pub fn gpu_rope(d_q: &mut CudaBuffer, d_k: &mut CudaBuffer, n_heads: usize, n_kv_heads: usize, head_dim: usize, pos: usize, rope_theta: f32) {
    unsafe {
        cuda_rope(d_q.ptr as *mut f32, d_k.ptr as *mut f32, n_heads as i32, n_kv_heads as i32, head_dim as i32, pos as i32, rope_theta, std::ptr::null_mut());
    }
}

pub fn sync_gpu() {
    unsafe { cuda_device_sync(); }
}

pub fn gpu_kv_store_layer(
    d_k_cache: &mut CudaBuffer,
    d_v_cache: &mut CudaBuffer,
    layer: usize,
    layer_stride: usize,
    d_k: &CudaBuffer,
    d_v: &CudaBuffer,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    pos: usize,
) {
    unsafe {
        
        let k_ptr = (d_k_cache.ptr as *mut u8).add(layer * layer_stride * 2) as *mut c_void;
        let v_ptr = (d_v_cache.ptr as *mut u8).add(layer * layer_stride * 2) as *mut c_void;
        cuda_kv_store(k_ptr, v_ptr, d_k.ptr as *const f32, d_v.ptr as *const f32, n_kv_heads as i32, head_dim as i32, max_seq_len as i32, pos as i32, std::ptr::null_mut());
    }
}

pub fn gpu_attention_decode_layer(
    d_out: &mut CudaBuffer,
    d_q: &CudaBuffer,
    d_k_cache: &CudaBuffer,
    d_v_cache: &CudaBuffer,
    layer: usize,
    layer_stride: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    pos: usize,
    sqrt_head_dim: f32,
) {
    unsafe {
        
        let k_ptr = (d_k_cache.ptr as *const u8).add(layer * layer_stride * 2) as *const c_void;
        let v_ptr = (d_v_cache.ptr as *const u8).add(layer * layer_stride * 2) as *const c_void;
        cuda_attention_decode(d_out.ptr as *mut f32, d_q.ptr as *const f32, k_ptr, v_ptr, n_heads as i32, n_kv_heads as i32, head_dim as i32, max_seq_len as i32, pos as i32, sqrt_head_dim, std::ptr::null_mut());
    }
}
pub fn gpu_gemm_w4a16_wmma(
    d_model_ptr: *const c_void,
    offset: usize,
    d_x_fp16: &CudaBuffer,
    d_y_out: &mut CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) {
    let blocks_per_row = k / 32;
    let total_blocks = n * blocks_per_row;
    let scales_size = total_blocks * 2;

    unsafe {
        let scales_ptr = (d_model_ptr as *const u8).add(offset) as *const c_void;
        let weights_ptr = (d_model_ptr as *const u8).add(offset + scales_size) as *const c_void;

        cuda_gemm_w4a16_wmma(
            scales_ptr,
            weights_ptr,
            d_x_fp16.ptr as *const c_void,
            d_y_out.ptr as *mut f32,
            m as i32,
            k as i32,
            n as i32,
            std::ptr::null_mut(),
        );
    }
}
pub fn test_wmma_tensor_cores() -> Result<(), String> {
    let m = 16;
    let k = 32;
    let n = 16;

    
    let scales = vec![f16::from_f32(1.0); n];
    
    let weights = vec![0x99u8; n * 16];
    
    let x_fp16 = vec![f16::from_f32(1.0); m * k];
    let mut y_out = vec![0.0f32; m * n];

    let d_scales = CudaBuffer::from_slice(&scales).ok_or("Alloc d_scales failed")?;
    let d_weights = CudaBuffer::from_slice(&weights).ok_or("Alloc d_weights failed")?;
    let d_x = CudaBuffer::from_slice(&x_fp16).ok_or("Alloc d_x failed")?;
    let d_y = CudaBuffer::new(m * n * 4).ok_or("Alloc d_y failed")?;

    sync_gpu();
    let start = std::time::Instant::now();

    unsafe {
        cuda_gemm_w4a16_wmma(
            d_scales.ptr,
            d_weights.ptr,
            d_x.ptr,
            d_y.ptr as *mut f32,
            m as i32,
            k as i32,
            n as i32,
            std::ptr::null_mut(),
        );
    }
    sync_gpu();
    let elapsed = start.elapsed();

    d_y.copy_to_host(&mut y_out);

    let actual = y_out[0];
    let expected = 32.0f32;

    if (actual - expected).abs() < 0.5 {
        println!(
            "[Tensor Cores] 🚀 Аппаратные тензорные ядра активны! Тест пройден: Y[0] = {:.1} (ожидалось {:.1}). Время: {:?}",
            actual, expected, elapsed
        );
        Ok(())
    } else {
        Err(format!("Сбой математики Tensor Cores: Y[0] = {}, ожидалось {}", actual, expected))
    }
}