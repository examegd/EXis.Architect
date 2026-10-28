use std::sync::Mutex;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Child};
use tauri::Manager;
use sysinfo::System;
use tauri::tray::{TrayIconBuilder, TrayIconEvent, MouseButton};
use tauri::menu::{Menu, MenuItem};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crate::cuda::{is_cuda_available, get_gpu_device_name};
mod cuda;
mod server;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

struct SystemState(Mutex<System>);

mod exiq;
use exiq::{ExiqModel, generate_stream};
use tokenizers::Tokenizer;
use tauri::ipc::Channel;


pub struct ExiqState {
    pub model: Mutex<Option<ExiqModel>>,
    pub active_model_name: Mutex<String>,
    pub is_aborted: Arc<AtomicBool>,
}
struct ProcessState(Mutex<Option<Child>>);
fn kill_zombie_processes() {
    #[cfg(target_os = "windows")]
    {
        
        let _ = Command::new("taskkill")
            .args(["/F", "/IM", "llama-server.exe"])
            .creation_flags(0x08000000) 
            .status();

        
        let _ = Command::new("taskkill")
            .args(["/F", "/IM", "llama-quantize.exe"])
            .creation_flags(0x08000000)
            .status();
    }
}


fn get_db_path(app: &tauri::AppHandle) -> PathBuf {
    let mut path = app.path().document_dir().unwrap_or_else(|_| PathBuf::from("."));
    path.push("exis");
    if !path.exists() {
        fs::create_dir_all(&path).unwrap();
    }
    path.push("exisdb.json");
    path
}

#[tauri::command]
fn read_exis_db(app: tauri::AppHandle) -> String {
    let path = get_db_path(&app);
    if !path.exists() {
        return r#"{"memories": []}"#.to_string();
    }
    fs::read_to_string(path).unwrap_or_else(|_| r#"{"memories": []}"#.to_string())
}

#[tauri::command]
fn write_exis_db(app: tauri::AppHandle, data: String) -> Result<(), String> {
    let path = get_db_path(&app);
    fs::write(path, data).map_err(|e| e.to_string())
}


#[tauri::command]
fn get_system_stats(state: tauri::State<'_, SystemState>) -> (f32, f32) {
    let mut sys = state.0.lock().unwrap();
    sys.refresh_all(); 

    let cpu_usage = sys.global_cpu_info().cpu_usage(); 
    let used_mem = sys.used_memory() as f32;
    let ram_gb = used_mem / 1024.0 / 1024.0 / 1024.0;

    (cpu_usage, ram_gb)
}
#[tauri::command]
async fn pick_model_file() -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let script = r#"
            [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null;
            $f = New-Object System.Windows.Forms.OpenFileDialog;
            $f.Filter = 'Model Files (*.exiq;*.gguf)|*.exiq;*.gguf|ExisQuant (*.exiq)|*.exiq|GGUF (*.gguf)|*.gguf|All Files (*.*)|*.*';
            $f.Title = 'Select Model File (.exiq or .gguf)';
            
            $topForm = New-Object System.Windows.Forms.Form;
            $topForm.TopMost = $true;
            
            if ($f.ShowDialog($topForm) -eq [System.Windows.Forms.DialogResult]::OK) { 
                Write-Output $f.FileName 
            }
        "#;

        let mut cmd = Command::new("powershell");
        cmd.args(["-NoProfile", "-Sta", "-Command", script]);

        #[cfg(target_os = "windows")]
        cmd.creation_flags(0x08000000);

        let output = cmd.output().map_err(|e| e.to_string())?;
        let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();

        if path_str.is_empty() {
            Ok(None)
        } else {
            Ok(Some(path_str))
        }
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn abort_exiq_generation(state: tauri::State<'_, ExiqState>) {
    state.is_aborted.store(true, Ordering::Relaxed);
}
#[tauri::command]
fn import_exiq_model(app: tauri::AppHandle, name: String, file_path: String) -> Result<String, String> {
    let src_path = PathBuf::from(&file_path);
    if !src_path.exists() {
        return Err("Указанный файл .exiq не существует!".to_string());
    }

    let doc_dir = app.path().document_dir().map_err(|e| e.to_string())?;
    let models_dir = doc_dir.join("exis").join("models");
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;
    }

    let clean_name = if name.ends_with(".exiq") { name } else { format!("{}.exiq", name) };
    let dest_model_path = models_dir.join(&clean_name);

    
    if src_path != dest_model_path {
        fs::copy(&src_path, &dest_model_path).map_err(|e| e.to_string())?;
    }

    
    let src_tok = src_path.with_file_name("tokenizer.json");
    if src_tok.exists() {
        let dest_tok = models_dir.join("tokenizer.json");
        let _ = fs::copy(&src_tok, &dest_tok);
    }

    Ok(format!("exiq:{}", clean_name))
}
#[tauri::command]
async fn run_exiq_chat(
    app: tauri::AppHandle,
    state: tauri::State<'_, ExiqState>,
    model_name: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    num_threads: usize,
    device: Option<String>,
    is_new_chat: Option<bool>,
    context_size: Option<usize>, 
    on_token: Channel<String>,
) -> Result<(), String> {
    let clean_name = model_name.replace("exiq:", "");
    let dev_str = device.unwrap_or_else(|| "auto".to_string());
    let device_sel = exiq::DeviceSelection::from_str(&dev_str);
    let is_new = is_new_chat.unwrap_or(false); 
    let ctx_size = context_size.unwrap_or(4096); 
    state.is_aborted.store(false, Ordering::Relaxed);
    let abort_flag = state.is_aborted.clone();

    
    {
        let mut model_lock = state.model.lock().unwrap_or_else(|e| e.into_inner());
        let mut name_lock = state.active_model_name.lock().unwrap_or_else(|e| e.into_inner());

        if model_lock.is_none() || *name_lock != clean_name {
            let path = PathBuf::from(&clean_name);
            let model_path = if path.exists() {
                path
            } else if let Ok(doc_dir) = app.path().document_dir() {
                doc_dir.join("exis").join("models").join(&clean_name)
            } else {
                return Err(format!("Файл модели {} не найден!", clean_name));
            };

            let m = ExiqModel::load(&model_path).map_err(|e| e.to_string())?;

            *model_lock = Some(m);
            *name_lock = clean_name;
        }
    }

    let app_handle = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let state = app_handle.state::<ExiqState>();
        let mut model_lock = state.model.lock().unwrap_or_else(|e| e.into_inner());
        let model = model_lock.as_mut().unwrap();

        let on_token_cb = move |tok: String| {
            let _ = on_token.send(tok);
        };

        generate_stream(
            model,
            &prompt,
            max_tokens,
            temperature,
            top_p,
            repeat_penalty,
            num_threads,
            device_sel,
            is_new, 
            ctx_size,
            abort_flag,
            on_token_cb,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let sys = System::new_all();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init()) 
        .manage(SystemState(Mutex::new(sys)))
        .manage(ProcessState(Mutex::new(None))) 
        .setup(|app| {
                let app_handle = app.handle().clone();
                let server_state = Arc::new(ExiqState {
                model: Mutex::new(None),
                active_model_name: Mutex::new(String::new()),
                is_aborted: Arc::new(AtomicBool::new(false)),
            });
            tauri::async_runtime::spawn_blocking(|| {
                crate::cuda::warmup_cuda();
                println!("[ExisQuant] 🔥 Контекст CUDA прогрет в фоне!");
                if let Err(e) = crate::cuda::test_wmma_tensor_cores() {
                    eprintln!("[ExisQuant] ❌ Ошибка проверки Tensor Cores: {}", e);
                }
            });
            tauri::async_runtime::spawn(async move {
                server::start_api_server(server_state, 11600).await;
            });

            
            let mut resource_path = app_handle.path().resource_dir().unwrap();
            resource_path.push("binaries");
            resource_path.push("exis.exe");

            if resource_path.exists() {
                let mut cmd = Command::new(resource_path);
                cmd.arg("serve");

                
                #[cfg(target_os = "windows")]
                cmd.creation_flags(0x08000000); 

                let child = cmd.spawn();
                
                if let Ok(c) = child {
                    let state = app_handle.state::<ProcessState>();
                    *state.0.lock().unwrap() = Some(c);
                }
            } else {
                eprintln!("Critical: exis.exe not found!");
            }
            
            
            let quit_item = MenuItem::with_id(app, "quit", "Exit", true, None::<&str>).unwrap();
            let show_item = MenuItem::with_id(app, "show", "Show exis.Architect", true, None::<&str>).unwrap();
            let tray_menu = Menu::with_items(app, &[&show_item, &quit_item]).unwrap();

            
            let tray_icon = app.default_window_icon().cloned().unwrap();

            let _tray = TrayIconBuilder::new()
                .icon(tray_icon)
                .menu(&tray_menu) 
                .on_menu_event(|app_handle, event| {
                    match event.id.as_ref() {
                        "quit" => {
                            let state = app_handle.state::<ProcessState>();
                            let mut lock = state.0.lock().unwrap();
                            if let Some(mut child) = lock.take() {
                                let _ = child.kill();
                            }
                            app_handle.exit(0);
                        }
                        "show" => {
                         if let Some(window) = app_handle.get_webview_window("main") {
                             window.show().unwrap();
                             window.set_focus().unwrap();
                             
                             if let Some(overlay) = app_handle.get_webview_window("overlay") {
                                 let _ = overlay.hide();
                             }
                         }
                     }
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, event| {
                 if let TrayIconEvent::Click { button: MouseButton::Left, .. } = event {
                     let app_handle = tray.app_handle();
                     if let Some(window) = app_handle.get_webview_window("main") {
                         window.show().unwrap();
                         window.set_focus().unwrap();
                         
                         if let Some(overlay) = app_handle.get_webview_window("overlay") {
                             let _ = overlay.hide();
                         }
                     }
                 }
             })
             
                .build(app)?;
                if let Some(overlay_window) = app_handle.get_webview_window("overlay") {
                if let Ok(Some(monitor)) = overlay_window.primary_monitor() {
                    let monitor_size = monitor.size();
                    let scale_factor = monitor.scale_factor();
                    
                    let hud_width = 380;
                    let hud_height = 450;
                    
                    
                    
                    let x = (monitor_size.width as f64 / scale_factor) as i32 - hud_width - 20;
                    let y = (monitor_size.height as f64 / scale_factor) as i32 - hud_height - 60;
                    
                    
                    let _ = overlay_window.set_position(tauri::Position::Physical(tauri::PhysicalPosition::new(
                        (x as f64 * scale_factor) as i32,
                        (y as f64 * scale_factor) as i32
                    )));
                }
            }

            Ok(())
        })
        
        .on_window_event(|window, event| {
            match event {
                
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    
                    if window.label() == "main" {
                        api.prevent_close();
                        window.hide().unwrap();
                        
                        if let Some(overlay) = window.app_handle().get_webview_window("overlay") {
                            let _ = overlay.show();
                            let _ = overlay.set_focus();
                        }
                    }
                }
                tauri::WindowEvent::Destroyed => {
                    let app_handle = window.app_handle();
                    let state = app_handle.state::<ProcessState>();
                    let mut lock = state.0.lock().unwrap();
                    if let Some(mut child) = lock.take() {
                        let _ = child.kill();
                    }
                    kill_zombie_processes();

                }
                _ => {}
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_system_stats,
            read_exis_db,
            write_exis_db,
            restore_main_window,
            exit_application,
            create_local_model,
            capture_screen
        ])
        .manage(ExiqState {
            model: Mutex::new(None),
            active_model_name: Mutex::new(String::new()),
            is_aborted: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            get_system_stats,
            read_exis_db,
            write_exis_db,
            restore_main_window,
            exit_application,
            create_local_model,
            capture_screen,
            list_exiq_models,
            run_exiq_chat,
            abort_exiq_generation, 
            pick_model_file,
            import_exiq_model,
            get_hardware_info,
            search_tavily
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
#[tauri::command]
async fn search_tavily(api_key: String, query: String) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        use std::io::Write;
        use std::process::Stdio;

        let clean_key = api_key.trim().trim_matches('"').trim_matches('\'').trim();
        let auth_header = format!("Authorization: Bearer {}", clean_key);

        let payload = serde_json::json!({
            "api_key": clean_key,
            "query": query.trim(),
            "search_depth": "basic",
            "max_results": 4
        });
        let payload_bytes = payload.to_string().into_bytes();

        let mut cmd = Command::new("curl.exe");
        cmd.args([
            "-s",
            "-X", "POST",
            "https://api.tavily.com/search",
            "-H", "Content-Type: application/json",
            "-H", "Accept: application/json",
            "-H", &auth_header, 
            "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "-d", "@-"
        ]);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());

        #[cfg(target_os = "windows")]
        cmd.creation_flags(0x08000000);

        let mut child = cmd.spawn().map_err(|e| format!("Ошибка запуска curl: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&payload_bytes);
        }

        let output = child.wait_with_output().map_err(|e| format!("Ошибка curl: {}", e))?;
        let resp_str = String::from_utf8_lossy(&output.stdout).to_string();
        
        let json: serde_json::Value = serde_json::from_str(&resp_str)
            .map_err(|e| format!("Ошибка ответа Tavily: {} (сырой ответ: {})", e, resp_str))?;

        if let Some(err_detail) = json.get("detail") {
            return Err(format!("Tavily Error: {}", err_detail));
        }

        Ok(json)
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn get_hardware_info() -> serde_json::Value {
    let has_cuda = is_cuda_available();
    let gpu_name = if has_cuda { get_gpu_device_name() } else { "None".to_string() };
    serde_json::json!({
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
    })
}
#[tauri::command]
fn restore_main_window(app: tauri::AppHandle) {
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.hide();
    }
}


#[tauri::command]
fn exit_application(state: tauri::State<'_, ProcessState>, app: tauri::AppHandle) {
    
    let mut lock = state.0.lock().unwrap();
    if let Some(mut child) = lock.take() {
        let _ = child.kill();
    }
    kill_zombie_processes();

    app.exit(0); 
}

#[tauri::command]
fn create_local_model(app: tauri::AppHandle, name: String, gguf_path: String) -> Result<(), String> {
   
    let gguf_path_buf = PathBuf::from(&gguf_path);
    if !gguf_path_buf.exists() {
        return Err(format!(
            "GGUF model file not found at the specified path:\n\"{}\"\nPlease verify the file location.", 
            gguf_path
        ));
    }


    let mut app_data_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    if !app_data_dir.exists() {
        
        fs::create_dir_all(&app_data_dir).map_err(|e| e.to_string())?;
    }
    
    let mut modelfile_path = app_data_dir;
    modelfile_path.push("TempModelfile");
    
    
    let safe_gguf_path = gguf_path.replace("\\", "/");
    let content = format!("FROM {}\n", safe_gguf_path);
    fs::write(&modelfile_path, content).map_err(|e| e.to_string())?;
    
    
    let mut resource_path = app.path().resource_dir().unwrap_or_else(|_| PathBuf::from("."));
    resource_path.push("binaries");
    resource_path.push("exis.exe");
    
    if !resource_path.exists() {
        resource_path = PathBuf::from("binaries");
        resource_path.push("exis.exe");
    }
    
    if !resource_path.exists() {
        let _ = fs::remove_file(&modelfile_path);
        return Err("Critical: exis.exe binary not found in resources or local binaries folder!".to_string());
    }
    
    let mut cmd = Command::new(resource_path);
    cmd.arg("create")
       .arg(&name)
       .arg("-f")
       .arg(&modelfile_path);

    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); 
    
    let output = cmd.output().map_err(|e| e.to_string())?;
                       
    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr).to_string();
        let _ = fs::remove_file(&modelfile_path);
        return Err(format!("Failed to create model: {}", err_msg));
    }
    
    let _ = fs::remove_file(modelfile_path);
    Ok(())
}


#[tauri::command]
fn capture_screen(app: tauri::AppHandle, window: tauri::WebviewWindow) -> Result<String, String> {
    
    let _ = window.hide();
    
    
    std::thread::sleep(std::time::Duration::from_millis(150));

    let mut temp_path = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    if !temp_path.exists() {
        fs::create_dir_all(&temp_path).unwrap();
    }
    temp_path.push("temp_screenshot.png");
    
    let safe_path = temp_path.to_string_lossy().replace("\\", "/");

    let script = format!(
        "[void][Reflection.Assembly]::LoadWithPartialName('System.Drawing'); \
         [void][Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); \
         $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; \
         $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; \
         $graphics = [System.Drawing.Graphics]::FromImage($bmp); \
         $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bmp.Size); \
         $bmp.Save('{}', [System.Drawing.Imaging.ImageFormat]::Png); \
         $graphics.Dispose(); \
         $bmp.Dispose();",
        safe_path
    );
    
    let mut cmd = Command::new("powershell");
    cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &script]);

    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000);
    
    let output = cmd.output().map_err(|e| e.to_string())?;
                
    
    let _ = window.show();
    let _ = window.set_focus();

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("PowerShell Capture Error: {}", err_msg));
    }
    
    Ok(temp_path.to_string_lossy().to_string())
}
#[tauri::command]
fn list_exiq_models(app: tauri::AppHandle) -> Vec<String> {
    let mut list = Vec::new();
    
    
    if let Ok(doc_dir) = app.path().document_dir() {
        let models_dir = doc_dir.join("exis").join("models");
        if let Ok(entries) = fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".exiq") {
                        list.push(format!("exiq:{}", name));
                    }
                }
            }
        }
    }
    
    
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".exiq") {
                    list.push(format!("exiq:{}", name));
                }
            }
        }
    }
    
    list
}
