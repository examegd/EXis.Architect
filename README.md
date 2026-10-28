# exis.Architect (v2.0.0)

`exis.Architect` is a lightweight, local/cloud-hybrid AI desktop client and inference suite featuring a dark-matte, glowing-green sci-fi interface.

The project is built on three core components:
1. **`exis.exe` (Backend):** A local inference engine based on Ollama / `llama.cpp` (Go/C++).
2. **`ExisQuant` (Native Engine):** A standalone CUDA C++ / Rust inference engine utilizing the proprietary `.exiq` quantization format.
3. **`exis.Architect` (Frontend):** A hardware-accelerated desktop interface built with Tauri v2 (Rust) and React / TypeScript.

<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716_175830_410.png" width="400" alt="exis.Architect">

---

## Key Features

### 1. Unified Local & Cloud Multimodal Support
- **Local GGUF Inference:** Runs open-weight GGUF models locally on a dedicated port (`11500`) configured via environment variables (`EXIS_MODELS`, `EXIS_DEBUG`).
- **Cloud Models Integration:** Direct API integration with Google Gemini models via Google AI Studio keys.
- **Multimodal Vision:** Image attachments supported for local vision models (such as `qwen2-vl` and `llava`) with in-chat image rendering.

### 2. ExisQuant (`.exiq`) Inference Engine
In addition to GGUF, the project includes an independent runtime and quantization pipeline written in Rust and CUDA C++ without external runtime dependencies.


- **Q4 Block Quantization:** FP16 weights are packed into 4-bit values (two weights per byte) with a block group size of 32 and independent scale factors. Reduces a 3B model from ~7 GB down to 2.2 GB (and 1.5B models to ~1.1 GB).
- **Single-File Container:** The `.exiq` container packages both quantized weights and tokenizer metadata into a single binary mapped directly to host memory via `mmap`.
- **Split Memory Layout:** Scales and 4-bit weights are partitioned into contiguous blocks. NVIDIA GPUs load weights via 128-bit vector reads (`uint4` / `LDG.128`), reading 32 weights per instruction to maximize bus throughput.
- **Shared Memory Cache:** Hidden state vectors are cached in GPU shared memory / L1 cache to minimize global VRAM traffic.
- **Tensor Core (WMMA) Execution:** Leverages NVIDIA Tensor Cores for $16 \times 16 \times 16$ half-precision matrix multiplication tiles. 
- **Prefix Caching:** KV-cache state is preserved between turns. Matching prompt prefixes are skipped without recomputation, minimizing time-to-first-token (TTFT) during long multi-turn sessions.
- **Native OpenAI-Compatible API (Port 11600):** Runs an independent HTTP server on port `11600` exposing a standard `/v1/chat/completions` endpoint with SSE streaming.

<img src="https://github.com/examegd/EXis.Architect/blob/media/speed.png" width="400" alt="ExisQuant Speed">

### 3. Symmetrical Sentry & Tactical HUD
- **Main Workspace:** Two-column layout with smooth scrolling for chat history, settings, and model compilation.
- **Tactical HUD (Overlay):** Frameless, transparent floating widget docked in the system tray. Pressing `?` opens a minimal prompt bar; the collapse icon `⤢` switches back to the main interface.
- **System Tray Management:** Intercepts close events to keep background services active without terminating execution.

<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716_174650_688.png" width="400" alt="Example tray mode">

### 4. Background CPU Limiter (Computational Throttle)
- **Dynamic Thread Scaling:** Configurable background throttle slider (10% to 90%).
- Minimizing the application to the tray automatically reduces the active thread count (`num_thread`) to reduce system load during gaming or development workflows.
- Configurable settings cover system prompts, CPU core limits, repetition penalties, API keys (Gemini, Tavily), and ExisDB configuration.

<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716_175828_359.png" width="400" alt="setting">

### 5. Local Memory Database (`exisdb.json`)
- **Conversational Memory:** Stores long-term user facts in plain JSON at `Documents/exis/exisdb.json`.
- **Keyword RAG:** Local retrieval engine with Russian stemming, automatically injecting relevant user context into incoming prompts.
- **Diff Confirmation Protocol:** Background model suggests updates, additions, or deletions via a visual diff panel (`MEMORIAL_UPDATE_PROTOCOL`), requiring explicit `COMMIT` or `ABORT` from the user.

<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716-181129.png" width="400" alt="Local Memory Search">
<img src="https://github.com/examegd/EXis.Architect/blob/media/IMG_20260716-181730.png" width="400" alt="Memory Diff">

### 6. Dual Search & Retrieval Drawer
- **Web Retrieval:** Optional real-time web search powered by the Tavily API.
- **Transparent Sources:** Accessed local memory blocks, search snippets, and Google Grounding references are displayed in an expandable `DATA_RETRIEVAL_LOG` with clickable URLs.

### 7. Core Ergonomics
- **Model Compiler:** Create models from arbitrary local `.gguf` files via `[ CREATE_MODEL... ]`.
- **Generation Control:** Dedicated abort mechanism issuing an `AbortController` signal to immediately halt model generation and release allocated threads.
- **Real-Time Diagnostics:** Monospace header monitoring CPU and RAM utilization (via Rust `sysinfo`) and live tokens-per-second (T/S) generation speed.

---

### 8. API Compatibility
- **Ollama REST API (Port 11500):** Compatible with standard scripts, LangChain, LlamaIndex, or AutoGen configured to target port `11500`.
- **OpenAI REST API (Port 11600):** ExisQuant server accepts requests formatted for OpenAI endpoints (`/v1/chat/completions`).
- **Headless Server Deployment:** Backend components can be compiled independently to run as a remote compute service on dedicated Linux/Windows servers.

---

## Technical Architecture

- **Backend (exis):** Go, C++ (`llama.cpp`), CMake, MSVC/MSYS2.
- **Inference Engine (ExisQuant):** Rust, CUDA C++, Tensor Core WMMA.
- **Desktop Host:** Rust (Tauri v2), Windows WebView2.
- **Frontend UI:** React, TypeScript, CSS Glassmorphism.

---

## Installation & Build

### Prerequisites
- **Go** (1.22+)
- **CMake** (3.24+) and C++ compilers (Visual Studio Build Tools / MSYS2)
- **CUDA Toolkit** (12.0+ for ExisQuant CUDA compilation)
- **Rust & Cargo**
- **Node.js** (LTS)

### Building the Backend (`exis.exe`)
```powershell
# 1. Build C++ server dependencies
cmake -B build .
cmake --build build --parallel 8

# 2. Build Go executable
go build -o exis.exe .
```

### Packaging the Desktop App
1. Move `exis.exe` and the `lib/ollama/` directory into `src-tauri/binaries/` (resulting in `src-tauri/binaries/exis.exe` and `src-tauri/binaries/lib/ollama/llama-server.exe`).
2. Install frontend dependencies and run the Tauri bundler:
```powershell
npm install
npm run tauri build
```
3. The generated NSIS installer will be located at:
   `src-tauri/target/release/bundle/nsis/exis-architect_1.0.0_x64-setup.exe`

---

## Credits & Attribution

- **Ollama Engine:** `exis.exe` is derived from the open-source **[Ollama](https://github.com/ollama/ollama)** project (MIT License).
- **Frontend & Runtime Integration:** Desktop UI, ExisQuant engine, Tactical HUD overlay, Active Memory subsystem, and diagnostic integrations developed by **examegd**.

---

## License

Distributed under the **GNU General Public License v3 (GPLv3)**. Derived works and modifications must remain open-source under GPLv3 with original copyright notices retained.
