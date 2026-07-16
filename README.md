# exis.Architect (v1.0.0)

`exis.Architect` is a customized, lightweight, local/cloud-hybrid AI assistant suite designed with a dark-matte, glowing-green sci-fi aesthetic inspired by the **Precursor/Architect technology**. 

It consists of two parts:
1. **`exis.exe` (Backend):** A custom Go/C++ local inference engine derived from Ollama.
2. **`exis.Architect` (Frontend):** A hardware-accelerated desktop interface built using Tauri v2 (Rust) and React (TypeScript).
<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716_175830_410.png" width="400" alt="exis.Architect">

---

## 🚀 Key Features

### 1. Unified Local & Cloud Multimodal Support
- **Local Inference:** Runs open-weight GGUF models locally on your PC on a dedicated, non-conflicting port (`11500`) using custom environment variables (`EXIS_MODELS`, `EXIS_DEBUG`, etc.).
- **Cloud Models Integration:** Real-time loading of Google's **Gemini** models (e.g., `gemini-2.5-flash`) directly from Google AI Studio using an API key (I intend to add support for other cloud models, but I chose Google first because their API is free) 
- **Multimodal Vision:** Drag-and-drop or select images to send to both local Vision models (like `qwen2-vl` or `llava`), with image rendering inside chat bubbles.




### 2. Symmetrical Sentry & Tactical HUD
- **Main Workspace:** A two-column dashboard with smooth scrolling lists for chat history, settings, and model compilation.
- **Tactical HUD (Overlay):** A frameless, transparent, always-on-top floating widget sitting in the system tray. Pressing `?` opens a minimalist input prompt. Clicking the collapse button `⤢` restores the main window and hides the HUD.
- **System Tray Integration:** Intercepts standard Windows close requests to minimize/hide the app to the tray instead of terminating, with background process lifecycle management.
<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716_174650_688.png" width="400" alt="Example tray mode">

### 3. Computational Throttle (Background CPU Limiter)
- **Hardware Protection:** In settings, you can adjust the "Background Throttle" from **10% to 90%** via a slider. 
- When the main app is minimized to the system tray, `exis` automatically cuts the allocated CPU threads (`num_thread`) to your background model in half (or by your custom percentage) to keep your PC responsive for gaming or coding.
- **In addition, the settings cover a wide range of options**, such as the *system prompt*, *cores* (as I mentioned earlier), and penalties. There are also API keys for *Gemini* and *Tavily*. And most importantly: **the ExisDB input field.**

<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716_175828_359.png" width="400" alt="setting">


### 4. Local Memory Database (`exisdb.json`)
- **Conversational Memory:** A private, human-readable JSON database (`exisdb.json`) created automatically in your `Documents/exis/` folder.
- **Instant Keyword RAG:** Built-in local search engine optimized for Russian grammar and verb stemming. It retrieves relevant user facts and injects them directly into your current prompt as context.
<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716-181129.png" width="400" alt="Описание картинки">

- **Active Memory Update Protocol:** After every response, a background AI model analyzes the dialogue and displays a green/red diff panel (`MEMORIAL_UPDATE_PROTOCOL`) asking to `COMMIT` or `ABORT` memory additions, updates, or deletions.
<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716-181730.png" width="400" alt="Example BD">



### 5. Double Search & Sources Drawer (`DATA_RETRIEVAL_LOG`)
- **Web Search Integration:** Toggle live web searches via Tavily API using the globe icon.
- **Real-Time Sources:** If the model accesses local memories, Tavily results, or Google Grounding, a `DATA_RETRIEVAL_LOG` box displays the exact references, snippets, and clickable URLs that open in your default OS browser.
<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716-181129.png" width="400" alt="Example search">

### 6. Additional Quality of Life Enhancements
- **Model Compiler (1-Click Creator):** Choose `[ CREATE_MODEL... ]` from the models dropdown to open a modal where you can select a local `.gguf` file and compile a new model as a background process.
- **Abort Stream (`STOP`):** While generating, the send button turns into a red `STOP` button. Clicking it triggers an `AbortController` signal to stop the local/cloud stream and release CPU resources.
- **Clean Monospace Analytics:** Displays real-time CPU/RAM hardware stats in the header (pulled directly from Rust `sysinfo`), along with real-time generation speed (T/S) calculations.
---
### 7. API Compatibility & External Integrations
- **Standard API Support:** The backend maintains compatibility with the standard Ollama REST API. You can easily interact with your local models using custom scripts (e.g., Python `requests`, Node.js, or `curl`) by simply routing your requests to port `11500`.
- **Ecosystem Compatibility:** Integrates with third-party tools, frameworks, and orchestration libraries that natively support Ollama (such as LangChain, LlamaIndex, or AutoGen) by pointing them to the custom port.
- **Remote Deployment Capable:** Because the engine is built on a Go/C++ core, it can be compiled and run as a standalone backend service on remote servers (VPS/VDS). This allows you to offload heavy computation and use the backend as a private, centralized API endpoint.

## 🛠️ Technical Architecture

- **Backend:** Go, C++ (`llama.cpp` integration), CMake, MSVC/MSYS2 compilers.
- **Frontend / Window Wrapper:** Rust (Tauri v2), leveraging native Windows WebView2 transparency and file system management.
- **UI Code:** React, Svelte, or TypeScript, utilizing CSS Glassmorphism and hardware-accelerated CSS transition layouts.

---

## 🔧 Installation & Build

### Prerequisites
To build both components from source, you will need:
- **Go** (1.22+)
- **CMake** (3.24+) and C++ compilers (Visual Studio Build Tools / MSYS2)
- **Node.js** (LTS)
- **Rust / Cargo** (via rustup)

### Compiling the Backend (`exis.exe`)
1. Clone the backend repository and navigate to its folder.
2. Build the C++ llama-server dependencies:
   ```powershell
   cmake -B build .
   cmake --build build --parallel 8
   ```
3. Build the Go binary:
   ```powershell
   go build -o exis.exe .
   ```

### Compiling the UI & Packaging the Installer
1. Navigate to the `EXis.Architect` project folder.
2. Copy your compiled `exis.exe` and the `lib/ollama/` folder from `exis-backend` into `src-tauri/binaries/` (so that the structure is `src-tauri/binaries/exis.exe` and `src-tauri/binaries/lib/ollama/llama-server.exe`).
3. Install Node packages and build the final NSIS `.exe` installer:
   ```powershell
   npm install
   npm run tauri build
   ```
4. Find your ready-to-use installer inside `src-tauri/target/release/bundle/nsis/exis-architect_1.0.0_x64-setup.exe`.

---

## 📜 Credits & Attribution

- **Backend Engine:** The core `exis.exe` local server and C++ model runners are based on a modified, custom fork of the open-source **[Ollama](https://github.com/ollama/ollama)** project (MIT License). Thanks to the Ollama team and community for building such a robust foundation.
- **Frontend & Integrations:** The custom Tauri/React desktop frame, tactical HUD overlays, Active Memory DB, system monitors, stream parsing, and custom styling were designed and developed by **examegd**.

---

## 🛡️ License

`exis.Architect` is licensed under the **GNU GPLv3 (General Public License v3)**. 
- You are free to read, modify, and redistribute this software.
- Any modified versions or derivative works of this application **must** remain open-source and be distributed under the same GPLv3 license for free.
- Original copyright notices and author attributions must be preserved.
```
