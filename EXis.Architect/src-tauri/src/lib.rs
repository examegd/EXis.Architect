use std::sync::Mutex;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Child};
use tauri::Manager;
use sysinfo::System;


use tauri::tray::{TrayIconBuilder, TrayIconEvent, MouseButton};
use tauri::menu::{Menu, MenuItem};


#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

struct SystemState(Mutex<System>);


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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let sys = System::new_all();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init()) 
        .manage(SystemState(Mutex::new(sys)))
        .manage(ProcessState(Mutex::new(None))) 
        .setup(|app| {
            let app_handle = app.handle().clone();
            
            
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
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
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
fn capture_screen(app: tauri::AppHandle) -> Result<String, String> {
    let mut temp_path = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    if !temp_path.exists() {
        fs::create_dir_all(&temp_path).unwrap();
    }
    temp_path.push("temp_screenshot.png");
    
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms, System.Drawing; \
         $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; \
         $bmp = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height); \
         $graphics = [System.Drawing.Graphics]::FromImage($bmp); \
         $graphics.CopyFromScreen(0, 0, 0, 0, $bmp.Size); \
         $bmp.Save('{}', [System.Drawing.Imaging.ImageFormat]::Png)",
        temp_path.to_string_lossy().replace("\\", "\\\\")
    );
    
    
    let mut cmd = Command::new("powershell");
    cmd.args(["-Command", &script]);

    
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000);
    
    
    let output = cmd.output().map_err(|e| e.to_string())?;
                
    if !output.status.success() {
        return Err("Failed to capture screenshot".to_string());
    }
    
    Ok(temp_path.to_string_lossy().to_string())
}