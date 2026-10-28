use std::process::Command;
use std::env;
use std::path::{Path, PathBuf};

fn find_cl_exe() -> Option<PathBuf> {
    let direct_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\HostX64\x64\cl.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe",
    ];

    for candidate in &direct_candidates {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }

    let search_roots = [
        r"C:\Program Files (x86)\Microsoft Visual Studio",
        r"C:\Program Files\Microsoft Visual Studio",
    ];

    for base in &search_roots {
        let base_path = Path::new(base);
        if !base_path.exists() { continue; }

        if let Ok(entries) = std::fs::read_dir(base_path) {
            for entry in entries.flatten() {
                let vs_dir = entry.path();
                let msvc_candidates = [
                    vs_dir.join("VC").join("Tools").join("MSVC"),
                    vs_dir.join("BuildTools").join("VC").join("Tools").join("MSVC"),
                    vs_dir.join("Community").join("VC").join("Tools").join("MSVC"),
                ];

                for m_dir in &msvc_candidates {
                    if m_dir.exists() {
                        if let Ok(ver_entries) = std::fs::read_dir(m_dir) {
                            for ver_entry in ver_entries.flatten() {
                                for host in &["HostX64", "Hostx64"] {
                                    let cl_path = ver_entry.path().join("bin").join(host).join("x64").join("cl.exe");
                                    if cl_path.exists() {
                                        return Some(cl_path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

fn main() {
    tauri_build::build();

    println!("cargo:rerun-if-changed=src/cuda/kernel.cu");

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3".to_string()
    });
    let nvcc = format!("{}\\bin\\nvcc.exe", cuda_path);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_file = out_dir.join("cuda_kernel.obj");

    let mut cmd = Command::new(&nvcc);

    if let Some(cl_path) = find_cl_exe() {
        let cl_dir = cl_path.parent().unwrap();
        println!("cargo:warning=Нашли компилятор MSVC: {}", cl_path.display());
        cmd.arg("-ccbin").arg(&cl_path);

        if let Ok(current_path) = env::var("PATH") {
            cmd.env("PATH", format!("{};{}", cl_dir.display(), current_path));
        }
    } else {
        println!("cargo:warning=ВНИМАНИЕ: cl.exe не найден автоматически!");
    }

    cmd.args([
        "-c",
        "src/cuda/kernel.cu",
        "-o",
        obj_file.to_str().unwrap(),
        "-O3",
        "--use_fast_math",
        "-arch=sm_75",
        "-Xcompiler",
        "/MT",
    ]);

    let status = cmd.status().expect("Не удалось запустить nvcc!");

    if !status.success() {
        panic!("Ошибка компиляции CUDA ядра через nvcc!");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
    println!("cargo:rustc-link-lib=cudart_static");
    println!("cargo:rustc-link-arg={}", obj_file.display());
}