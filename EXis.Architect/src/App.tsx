import { useState, useEffect, useRef } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";
import "./App.css";
import { getCurrentWindow } from "@tauri-apps/api/window";
const handleOpenLink = async (url: string) => {
  try {
    await open(url);
  } catch (e) {
    console.error("Failed to open external link:", e);
  }
};
interface Source {
  type: "db" | "web";
  title: string;
  url?: string;
  snippet?: string;
}
interface Memory {
  id: string;
  text: string;
  tags: string[];
  timestamp: string;
}
interface ProposedChange {
  action: "add" | "update" | "delete" | "none";
  addText?: string;
  deleteId?: string;
  deleteText?: string;
  tags?: string[];
}
interface Message {
  id: number;
  sender: "user" | "assistant";
  text: string;
  thinking?: string;
  speed?: number; 
  sources?: Source[];
  image?: string;
}
interface ExisDB {
  memories: Memory[];
}
interface Chat {
  id: string;
  title: string;
  messages: Message[];
}
function PrecursorGlyph() {
  return (
    <svg className="precursor-glyph" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path 
        d="M12 2L2 12L12 22L22 12L12 2ZM12 6L18 12L12 18L6 12L12 6Z" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      />
      <rect x="10" y="10" width="4" height="4" rx="0.5" fill="currentColor"/>
    </svg>
  );
}
function GlobeIcon() {
  return (
    <svg className="search-icon-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path 
        d="M12 21C16.9706 21 21 16.9706 21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21Z" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      />
      <path 
        d="M3.6 9H20.4M3.6 15H20.4M12 3C14.5 5.5 15.5 8.5 15.5 12C15.5 15.5 14.5 18.5 12 21C9.5 18.5 8.5 15.5 8.5 12C8.5 8.5 9.5 5.5 12 3Z" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      />
    </svg>
  );
}
function PaperclipIcon() {
  return (
    <svg className="file-icon-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path 
        d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      />
    </svg>
  );
}
function ThinkingBlock({ content }: { content: string }) {
  const [isOpen, setIsOpen] = useState(true);
  return (
    <div className="thinking-box">
      <div className="thinking-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="thinking-header-left">
          <PrecursorGlyph />
          <span>ARCHITECT_REASONING_PROTOCOL</span>
        </div>
        <span>{isOpen ? "COLLAPSE" : "EXPAND"}</span>
      </div>
      {isOpen && <div className="thinking-content">{content}</div>}
    </div>
  );
}
function SourcesGlyph() {
  return (
    <svg className="sources-glyph" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path 
        d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      />
    </svg>
  );
}
function SourcesBlock({ sources }: { sources: Source[] }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="sources-box">
      <div className="sources-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="sources-header-left">
          <SourcesGlyph />
          <span>DATA_RETRIEVAL_LOG</span>
        </div>
        <span>{isOpen ? "COLLAPSE" : "EXPAND"}</span>
      </div>
      {isOpen && (
        <div className="sources-content">
          {sources.map((src, index) => (
            <div key={index} className="source-item">
              <span className="source-item-title">
                {src.type === "db" ? "[DATABASE_MATCH]" : "[WEB_REFERENCE]"} {src.title}
              </span>
              {}
              {src.url && (
                <span 
                  className="source-item-link" 
                  onClick={() => handleOpenLink(src.url!)}
                >
                  {src.url}
                </span>
              )}
              {}
              {src.snippet && (
                <div className="source-item-snippet">
                  {src.snippet}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
function OverlayHUD({
  models,
  selectedModel,
  setSelectedModel,
  systemPrompt,
  numThreads,
  temperature,
  repeatPenalty,
  geminiApiKey,
  handleCaptureScreen,
  throttleInBg,
  throttlePercent,
}: {
  models: string[];
  selectedModel: string;
  setSelectedModel: (m: string) => void;
  systemPrompt: string;
  numThreads: number;
  temperature: number;
  repeatPenalty: number;
  geminiApiKey: string;
  setIsCreateModelOpen: (b: boolean) => void;
  handleCaptureScreen: () => void;
  throttleInBg: boolean;
  throttlePercent: number;
  presencePenalty: number;
  frequencyPenalty: number;
  topP: number;
  topK: number;
  numCtx: number;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [hudInput, setHudInput] = useState("");
  const [hudResponse, setHudResponse] = useState("");
  const [hudThinking, setHudThinking] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
 const handleRestoreMain = async () => {
   try {
     await invoke("restore_main_window");
   } catch (e) {
     console.error("Failed to restore main window:", e);
   }
 };
 const handleExitApp = async () => {
   try {
     await invoke("exit_application");
   } catch (e) {
     console.error("Failed to exit application:", e);
   }
 };
  const handleHudSend = async () => {
    if (!hudInput.trim() || isGenerating) return;
    const userText = hudInput;
    setHudInput("");
    setIsGenerating(true);
    setHudResponse("");
    setHudThinking("");
    const isGeminiModel = selectedModel.startsWith("gemini-");
    try {
      let response: Response;
      if (isGeminiModel) {
        if (!geminiApiKey.trim()) throw new Error("Missing Gemini API Key");
        response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:streamGenerateContent?alt=sse&key=${geminiApiKey}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              contents: [{ role: "user", parts: [{ text: userText }] }],
              systemInstruction: { parts: [{ text: systemPrompt }] },
              generationConfig: { temperature: temperature }
            })
          }
        );
      } else {
        response = await fetch("http://127.0.0.1:11500/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedModel,
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userText }
            ],
            stream: true,
            options: {
              num_thread: throttleInBg 
                ? Math.max(1, Math.round(numThreads * (1 - throttlePercent / 100))) 
                : numThreads,
              temperature: temperature,
              repeat_penalty: repeatPenalty
            }
          })
        });
      }
      if (!response.body) throw new Error("Empty body from API stream");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let partialLine = "";
      let fullTextAccumulator = "";
      let thinkingAccumulator = "";
      let responseTextAccumulator = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + chunk).split("\n");
        partialLine = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            let token = "";
            if (isGeminiModel) {
              if (line.startsWith("data: ")) {
                const jsonStr = line.slice(6).trim();
                const parsed = JSON.parse(jsonStr);
                token = parsed.candidates?.[0]?.content?.parts?.[0]?.text || "";
              }
            } else {
              const parsed = JSON.parse(line);
              token = parsed.message?.content || "";
            }
            fullTextAccumulator += token;
            if (fullTextAccumulator.includes("<thinking>")) {
              const parts = fullTextAccumulator.split("<thinking>");
              const textBeforeThinking = parts[0];
              const rest = parts[1] || "";
              if (rest.includes("</thinking>")) {
                const subParts = rest.split("</thinking>");
                thinkingAccumulator = subParts[0];
                responseTextAccumulator = textBeforeThinking + (subParts[1] || "");
              } else {
                thinkingAccumulator = rest;
                responseTextAccumulator = textBeforeThinking;
              }
            } else {
              responseTextAccumulator = fullTextAccumulator;
            }
            setHudThinking(thinkingAccumulator);
            setHudResponse(responseTextAccumulator);
          } catch (err) {
            console.error("Stream parse error:", err);
          }
        }
      }
    } catch (e) {
      console.error("HUD execution error:", e);
      setHudResponse("SYSTEM_ERROR: Failed to connect to core.");
    } finally {
      setIsGenerating(false);
    }
  };
  return (
    <div className="hud-window">
      {}
      {isOpen && (
        <div className="hud-glowing-frame">
          <div className="change-header">
            <div className="change-header-left">
              <PrecursorGlyph />
              <span>TACTICAL_HUD_DIRECTIVE</span>
            </div>
            {}
             <select
              className="model-select"
              style={{ fontSize: "10px", padding: "2px 6px" }}
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isGenerating}
            >
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          {}
          <textarea
            className="hud-input"
            placeholder={isGenerating ? "COMPUTING..." : "Enter quick directive..."}
            value={hudInput}
            onChange={(e) => setHudInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleHudSend();
              }
            }}
            disabled={isGenerating}
          />
          <div className="hud-actions">
            <button 
              className="hud-btn" 
              style={{ borderColor: "#00bcff", color: "#00bcff" }}
              onClick={handleCaptureScreen}
            >
              CAPTURE
            </button>
            <button className="hud-btn" onClick={handleHudSend} disabled={isGenerating}>
              {isGenerating ? "..." : "EXECUTE"}
            </button>
          </div>
          {}
          {(hudThinking || hudResponse || isGenerating) && (
            <div className="hud-response-box">
              {hudThinking && <ThinkingBlock content={hudThinking} />}
              {isGenerating && !hudResponse && !hudThinking && (
                <div style={{ color: "var(--glow-color)", animation: "pulse 1s infinite" }}>
                  COMPUTING_RESPONSE...
                </div>
              )}
              <div>{hudResponse}</div>
            </div>
          )}
        </div>
      )}
      {}
      <div className="hud-trigger-buttons">
       {}
       <button 
         className="hud-trigger-circle" 
         onClick={handleExitApp}
         title="Exit exis.Architect"
         style={{ borderColor: "#ff4a4a", color: "#ff4a4a" }}
       >
         ✕
       </button>
       {}
       <button 
         className="hud-trigger-circle" 
         onClick={handleRestoreMain}
         title="Maximize exis.Architect"
         style={{ borderColor: "#00bcff", color: "#00bcff" }}
       >
         ⤢
       </button>
       {}
       <button 
         className="hud-trigger-circle" 
         onClick={() => setIsOpen(!isOpen)}
         title="Toggle Tactical HUD"
       >
         ?
       </button>
     </div>
    </div>
  );
}
function renderMessageContent(text: string) {
  if (!text) return null;
  const combinedRegex = /(!?\[.*?\]\(.*?\))/g;
  const elements: React.ReactNode[] = [];
  let lastIndex = 0;
  let match;
  while ((match = combinedRegex.exec(text)) !== null) {
    const matchIndex = match.index;
    if (matchIndex > lastIndex) {
      elements.push(<span key={`text-${lastIndex}`}>{text.substring(lastIndex, matchIndex)}</span>);
    }
    const token = match[0];
    if (token.startsWith("![")) {
      const imgMatch = /!\[(.*?)\]\((.*?)\)/.exec(token);
      if (imgMatch) {
        const alt = imgMatch[1];
        const url = imgMatch[2];
        elements.push(
          <img 
            key={`img-${matchIndex}`} 
            src={url} 
            alt={alt} 
            className="message-image-inline" 
          />
        );
      }
    } else if (token.startsWith("[")) {
      const linkMatch = /\[(.*?)\]\((.*?)\)/.exec(token);
      if (linkMatch) {
        const label = linkMatch[1];
        const url = linkMatch[2];
        const isVideo = label.toLowerCase() === "video" || 
                        url.endsWith(".mp4") || 
                        url.endsWith(".webm") || 
                        url.endsWith(".ogg");
        if (isVideo) {
          elements.push(
            <video 
              key={`video-${matchIndex}`} 
              src={url} 
              controls 
              className="message-video" 
            />
          );
        } else {
          elements.push(
            <span 
              key={`link-${matchIndex}`} 
              className="source-item-link" 
              style={{ textDecoration: "underline", cursor: "pointer" }}
              onClick={() => handleOpenLink(url)}
            >
              {label}
            </span>
          );
        }
      }
    }
    lastIndex = combinedRegex.lastIndex;
  }
  if (lastIndex < text.length) {
    elements.push(<span key={`text-${lastIndex}`}>{text.substring(lastIndex)}</span>);
  }
  return <div style={{ whiteSpace: "pre-wrap" }}>{elements.length > 0 ? elements : text}</div>;
}
export default function App() {
  const [computeDevice, setComputeDevice] = useState<string>(() => {
    return localStorage.getItem("exis_compute_device") || "auto";
  });
  const [gpuName, setGpuName] = useState<string>("");
  useEffect(() => {
    localStorage.setItem("exis_compute_device", computeDevice);
  }, [computeDevice]);
  useEffect(() => {
    invoke<any>("get_hardware_info").then((info) => {
      if (info && info.has_cuda) {
        setGpuName(info.gpu_name);
      }
    }).catch(console.error);
  }, []);
  const [chats, setChats] = useState<Chat[]>(() => {
    const saved = localStorage.getItem("exis_chats");
    return saved ? JSON.parse(saved) : [
      {
        id: "chat-1",
        title: "SCHEMATIC_ALPHA",
        messages: [
          {
          id: 1,
          sender: "assistant",
          text: "exis.Architect link active. Core operational on port 11500. Select model and initialize instruction set.",
          thinking: "SYSTEM_SELF_TEST: OK\nPORT_11500: ACTIVE\nSYNCHRONIZING_MODEL_REGISTRY...",
          sources: [
            { 
              type: "db", 
              title: "exisdb_core_protocol", 
              snippet: "System specifications for Port 11500 and localized memory storage paths." 
            },
            {
              type: "web",
              title: "AnarchyBox",
              url: "https://thunderstore.io/c/repo/p/examegd/AnarchyBox/",
              snippet: "Information regarding the Architects' high-tech energy structures and ion crystal matrix cores."
            }
          ]
        }
        ]
      }
    ];
  });
  const [systemPrompt, setSystemPrompt] = useState<string>(() => {
    const saved = localStorage.getItem("exis_system_prompt");
    return saved || "You are exis.Architect.";
  });
  const [numThreads, setNumThreads] = useState<number>(() => {
    const saved = localStorage.getItem("exis_num_threads");
    return saved ? Math.max(1, parseInt(saved, 10)) : 4; 
  });
  const [temperature, setTemperature] = useState<number>(() => {
    const saved = localStorage.getItem("exis_temperature");
    return saved ? Math.min(2, Math.max(0, parseFloat(saved))) : 0.7;
  });
  const [repeatPenalty, setRepeatPenalty] = useState<number>(() => {
    const saved = localStorage.getItem("exis_repeat_penalty");
    return saved ? Math.max(0, parseFloat(saved)) : 1.1;
  });
  const [numCtx, setNumCtx] = useState<number>(() => {
    const saved = localStorage.getItem("exis_num_ctx");
    return saved ? Math.max(512, parseInt(saved, 10)) : 4096;
  });
  const [topP, setTopP] = useState<number>(() => {
    const saved = localStorage.getItem("exis_top_p");
    return saved ? Math.min(1, Math.max(0, parseFloat(saved))) : 0.9;
  });
  const [topK, setTopK] = useState<number>(() => {
    const saved = localStorage.getItem("exis_top_k");
    return saved ? Math.max(1, parseInt(saved, 10)) : 40;
  });
  const [presencePenalty, setPresencePenalty] = useState<number>(() => {
    const saved = localStorage.getItem("exis_presence_penalty");
    return saved ? Math.max(0, parseFloat(saved)) : 0.0;
  });
  const [frequencyPenalty, setFrequencyPenalty] = useState<number>(() => {
    const saved = localStorage.getItem("exis_frequency_penalty");
    return saved ? Math.max(0, parseFloat(saved)) : 0.0;
  });
  const [newMemoryText, setNewMemoryText] = useState("");
  const [newMemoryTags, setNewMemoryTags] = useState("");
  const [isWebSearchEnabled, setIsWebSearchEnabled] = useState<boolean>(() => {
    const saved = localStorage.getItem("exis_web_search_enabled");
    return saved === "true";
  });
    const [throttleInBg, setThrottleInBg] = useState<boolean>(() => {
    const saved = localStorage.getItem("exis_throttle_in_bg");
    return saved ? saved === "true" : true;
  });
  useEffect(() => {
    localStorage.setItem("exis_throttle_in_bg", throttleInBg.toString());
  }, [throttleInBg]);
  const [throttlePercent, setThrottlePercent] = useState<number>(() => {
    const saved = localStorage.getItem("exis_throttle_percent");
    return saved ? parseInt(saved, 10) : 50;
  });
  useEffect(() => {
    localStorage.setItem("exis_throttle_percent", throttlePercent.toString());
  }, [throttlePercent]);
  const [tavilyApiKey, setTavilyApiKey] = useState<string>(() => {
    return localStorage.getItem("exis_tavily_api_key") || "";
  });
    const [useSearchGrounding, setUseSearchGrounding] = useState<boolean>(() => {
    const saved = localStorage.getItem("exis_use_search_grounding");
    return saved === "true";
  });
  useEffect(() => {
    localStorage.setItem("exis_use_search_grounding", useSearchGrounding.toString());
  }, [useSearchGrounding]);
  const [geminiApiKey, setGeminiApiKey] = useState<string>(() => {
    return localStorage.getItem("exis_gemini_api_key") || "";
  });
  const [windowLabel, setWindowLabel] = useState<string>("main");
  useEffect(() => {
    const label = getCurrentWindow().label;
    setWindowLabel(label);
    if (label === "overlay") {
      const targets = [
        document.documentElement,
        document.body,
        document.getElementById("root")
      ];
      targets.forEach((el) => {
        if (el) {
          el.style.setProperty("background", "transparent", "important");
          el.style.setProperty("background-color", "transparent", "important");
        }
      });
    }
  }, []);
  useEffect(() => {
    localStorage.setItem("exis_tavily_api_key", tavilyApiKey);
  }, [tavilyApiKey]);
  useEffect(() => {
    localStorage.setItem("exis_web_search_enabled", isWebSearchEnabled.toString());
  }, [isWebSearchEnabled]);
  useEffect(() => {
    localStorage.setItem("exis_tavily_api_key", tavilyApiKey);
  }, [tavilyApiKey]);
  useEffect(() => {
    localStorage.setItem("exis_gemini_api_key", geminiApiKey);
  }, [geminiApiKey]);
const handleAddMemory = async () => {
    if (!newMemoryText.trim()) return;
    const tagsArray = newMemoryTags
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);
    const newMemory: Memory = {
      id: `mem-${Date.now()}`,
      text: newMemoryText,
      tags: tagsArray,
      timestamp: new Date().toISOString()
    };
    const updatedDB: ExisDB = {
      memories: [...exisDB.memories, newMemory]
    };
    try {
      await invoke("write_exis_db", { data: JSON.stringify(updatedDB) });
      setExisDB(updatedDB);
      setNewMemoryText("");
      setNewMemoryTags("");
    } catch (e) {
      console.error("Failed to write exisdb.json through Rust:", e);
    }
  };
  const [cpuUsage, setCpuUsage] = useState<number>(0);
  const [exisDB, setExisDB] = useState<ExisDB>({ memories: [] });
  const [ramUsage, setRamUsage] = useState<number>(0);
  const [inputValue, setInputValue] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [pendingChange, setPendingChange] = useState<ProposedChange | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [attachedImage, setAttachedImage] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [isButtonHovered, setIsButtonHovered] = useState(false);
    const [isCreateModelOpen, setIsCreateModelOpen] = useState(false);
  const [newModelName, setNewModelName] = useState("");
  const [newModelGgufPath, setNewModelGgufPath] = useState("");
  const [isCreatingModel, setIsCreatingModel] = useState(false);
  const [createModelError, setCreateModelError] = useState("");
  const [scannerImage, setScannerImage] = useState<string | null>(null);
  const [cropStart, setCropStart] = useState<{ x: number; y: number } | null>(null);
  const [cropCurrent, setCropCurrent] = useState<{ x: number; y: number } | null>(null);
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("exis.Architect currently only supports image processing (PNG/JPG). Audio and video analysis is not supported by the active model.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setImagePreview(result);
      const base64Data = result.split(",")[1];
      setAttachedImage(base64Data);
    };
    reader.readAsDataURL(file);
  };
  const handleRemoveImage = () => {
    setAttachedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };
  const [activeChatId, setActiveChatId] = useState<string>(() => {
    return chats[0]?.id || "chat-1";
  });
  const activeChat = chats.find((c) => c.id === activeChatId) || chats[0];
  const chatEndRef = useRef<HTMLDivElement>(null);
  const lastScrollTimeRef = useRef<number>(0);
  useEffect(() => {
    localStorage.setItem("exis_chats", JSON.stringify(chats));
  }, [chats]);
  useEffect(() => {
    invoke<string>("read_exis_db")
      .then((dataStr) => {
        try {
          setExisDB(JSON.parse(dataStr));
        } catch (e) {
          console.error("Failed to parse exisdb.json:", e);
        }
      })
      .catch((err) => {
        console.error("Failed to read exisdb from Rust:", err);
      });
  }, []);
  useEffect(() => {
    localStorage.setItem("exis_system_prompt", systemPrompt);
  }, [systemPrompt]);
  useEffect(() => {
    localStorage.setItem("exis_num_ctx", numCtx.toString());
  }, [numCtx]);
  useEffect(() => {
    localStorage.setItem("exis_top_p", topP.toString());
  }, [topP]);
  useEffect(() => {
    localStorage.setItem("exis_top_k", topK.toString());
  }, [topK]);
  useEffect(() => {
    localStorage.setItem("exis_presence_penalty", presencePenalty.toString());
  }, [presencePenalty]);
  useEffect(() => {
    localStorage.setItem("exis_frequency_penalty", frequencyPenalty.toString());
  }, [frequencyPenalty]);
  useEffect(() => {
    const interval = setInterval(() => {
      invoke<[number, number]>("get_system_stats")
        .then(([cpu, ram]) => {
          setCpuUsage(cpu);
          setRamUsage(ram);
        })
        .catch((err) => {
          console.error("Failed to fetch system stats from Rust:", err);
        });
    }, 1000);
    return () => clearInterval(interval);
  }, []);
useEffect(() => {
    invoke<string[]>("list_exiq_models").then((exiqList) => {
      if (exiqList && exiqList.length > 0) {
        setModels((prev) => Array.from(new Set([...prev, ...exiqList])));
        if (!selectedModel) setSelectedModel(exiqList[0]);
      }
    }).catch(console.error);
    fetch("http://127.0.0.1:11500/api/tags")
      .then((res) => res.json())
      .then(async (localData: any) => {
        let modelNames: string[] = [];
        if (localData.models && localData.models.length > 0) {
          modelNames = localData.models.map((m: any) => m.name);
        }
        if (geminiApiKey.trim().length > 0) {
          try {
            const googleRes = await fetch(
              `https://generativelanguage.googleapis.com/v1beta/models?key=${geminiApiKey}`
            );
            const googleData = await googleRes.json();
            if (googleData.models) {
              const geminiModels = googleData.models
                .filter((m: any) => m.name.includes("gemini") && m.supportedGenerationMethods.includes("generateContent"))
                .map((m: any) => m.name.replace("models/", ""));
              modelNames = [...modelNames, ...geminiModels];
            }
          } catch (e) {
            console.error("Gemini load error:", e);
          }
        }
        setModels((prev) => Array.from(new Set([...prev, ...modelNames])));
      })
      .catch((err) => {
        console.warn("exis.exe server offline", err);
      });
  }, [geminiApiKey]);
  useEffect(() => {
    const now = Date.now();
    if (isGenerating) {
      if (now - lastScrollTimeRef.current > 150) {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
        lastScrollTimeRef.current = now;
      }
    } else {
      chatEndRef.current?.scrollIntoView({ behavior: "auto" });
    }
  }, [activeChat.messages, isGenerating]);
  const handleCreateNewChat = () => {
    const newId = `chat-${Date.now()}`;
    const newChat: Chat = {
      id: newId,
      title: `SCHEMATIC_${chats.length + 1}`,
      messages: [
        {
          id: Date.now(),
          sender: "assistant",
          text: "New schematic session created. Ready for initialization.",
          thinking: "CLEARING_CONTEXT_CACHE...\nINITIALIZING_NEW_SESSION..."
        }
      ]
    };
    setChats((prev) => [...prev, newChat]);
    setActiveChatId(newId);
  };
    const handleDeleteChat = (id: string) => {
    if (chats.length <= 1) return;
    setChats((prev) => {
      const filtered = prev.filter((c) => c.id !== id);
      if (activeChatId === id) {
        setActiveChatId(filtered[0].id);
      }
      return filtered;
    });
  };
const handleAbortGeneration = () => {
    invoke("abort_exiq_generation").catch(console.error);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
  };
  const retrieveMemories = (query: string, memories: Memory[]): Memory[] => {
    const words = query
      .toLowerCase()
      .replace(/[^a-zA-Zа-яА-Я0-9\s]/g, "")
      .split(/\s+/)
      .filter((w) => w.length > 2)
      .map((w) => (w.length > 4 ? w.slice(0, 4) : w));
    if (words.length === 0) return [];
    return memories.filter((mem) => {
      const memTextLower = mem.text.toLowerCase();
      const textMatch = words.some((word) => memTextLower.includes(word));
      const tagMatch = mem.tags.some((tag) => 
        words.some((word) => tag.toLowerCase().includes(word))
      );
      return textMatch || tagMatch;
    });
  };
const handleSend = async () => {
    if (!inputValue.trim() || isGenerating) return;
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const userText = inputValue;
    setInputValue("");
    setIsGenerating(true);
    const userMessageId = Date.now();
    const newUserMessage: Message = {
      id: userMessageId,
      sender: "user",
      text: userText,
      image: imagePreview || undefined 
    };
    const imageToSend = attachedImage; 
    handleRemoveImage();
    setChats((prevChats) =>
      prevChats.map((c) => {
        if (c.id === activeChatId) {
          return { ...c, messages: [...c.messages, newUserMessage] };
        }
        return c;
      })
    );
    const matchedMemories = retrieveMemories(userText, exisDB.memories);
    const messageSources: Source[] = [];
    let activeMessageSources: Source[] = [];
    if (matchedMemories.length > 0) {
      matchedMemories.forEach((m) => {
        messageSources.push({
          type: "db",
          title: `exisdb: ${m.tags.join(", ") || "memory"}`,
          snippet: m.text
        });
      });
    } 
    let webSearchContext = "";
    if (isWebSearchEnabled && tavilyApiKey.trim()) {
      try {
        const searchData = await invoke<any>("search_tavily", {
          apiKey: tavilyApiKey.trim(),
          query: userText
        });
        if (searchData.results && searchData.results.length > 0) {
          const snippets: string[] = [];
          searchData.results.forEach((res: any) => {
            messageSources.push({
              type: "web",
              title: res.title || "Web Source",
              url: res.url,
              snippet: res.content
            });
            snippets.push(`- Источник [${res.title}](${res.url}): ${res.content}`);
          });
          webSearchContext = snippets.join("\n");
        }
      } catch (e: any) {
        console.error("Web search execution failed:", e);
        alert(`Ошибка поиска Tavily:\n${e}`);
      }
    }
    activeMessageSources = [...messageSources];
    const formattedHistory = [
      { role: "system", content: systemPrompt }
    ];
    activeChat.messages.forEach((m) => {
      formattedHistory.push({
        role: m.sender === "user" ? "user" : "assistant",
        content: m.text,
        images: m.image ? [m.image.split(",")[1]] : undefined
      } as any);
    });
    let finalUserMessageContent = userText;
    let injectedContext = "";
    if (matchedMemories.length > 0) {
      const memoryContext = matchedMemories.map((m) => `- ${m.text}`).join("\n");
      injectedContext += `### LOCAL_DATABASE_RECORDS:\n${memoryContext}\n\n`;
    }
    if (webSearchContext) {
      injectedContext += `### LIVE_WEB_SEARCH_RESULTS:\n${webSearchContext}\n\n`;
    }
    if (injectedContext) {
      finalUserMessageContent = `[SYSTEM_RETRIEVED_CONTEXT: You MUST use the following verified local database records and live web search results to construct an accurate, up-to-date response in Russian. If the search results do not contain the answer, ignore this block. Context:\n\n${injectedContext}]\n\nMy Question: ${userText}`;
    }
    const lastUserMsgPayload: any = {
      role: "user",
      content: finalUserMessageContent
    };
    if (imageToSend) {
      lastUserMsgPayload.images = [imageToSend];
    }
    formattedHistory.push(lastUserMsgPayload);
    const assistantMessageId = Date.now() + 1;
    const initialAssistantMessage: Message = {
      id: assistantMessageId,
      sender: "assistant",
      text: "",
      thinking: "",
      sources: activeMessageSources.length > 0 ? activeMessageSources : undefined
    };
    setChats((prevChats) =>
      prevChats.map((c) => {
        if (c.id === activeChatId) {
          return { ...c, messages: [...c.messages, initialAssistantMessage] };
        }
        return c;
      })
    );
    try {
      const isGeminiModel = selectedModel.startsWith("gemini-");
      const isExiqModel = selectedModel.startsWith("exiq:");
      //exiq
      if (isExiqModel) {
        const tokenChannel = new Channel<string>();
        let fullText = "";
        let genStartTime = 0;
        let tokenCount = 0;
        let finalSpeed = 0;
        tokenChannel.onmessage = (token: string) => {
          if (tokenCount === 0) {
            genStartTime = Date.now();
          }
          fullText += token;
          tokenCount++;
          const elapsed = (Date.now() - genStartTime) / 1000;
          if (elapsed > 0.3) {
            finalSpeed = tokenCount / elapsed;
          }
          setChats((prevChats) =>
            prevChats.map((c) => {
              if (c.id === activeChatId) {
                return {
                  ...c,
                  messages: c.messages.map((m) => {
                    if (m.id === assistantMessageId) {
                      return {
                        ...m,
                        text: fullText,
                        speed: finalSpeed > 0 ? finalSpeed : undefined
                      };
                    }
                    return m;
                  })
                };
              }
              return c;
            })
          );
        };
        let chatPrompt = `<|im_start|>system\n${systemPrompt} Отвечай строго на русском языке.<|im_end|>\n`;
        activeChat.messages.forEach((m) => {
          if (m.id !== 1 && m.text.trim()) { 
            chatPrompt += `<|im_start|>${m.sender}\n${m.text}<|im_end|>\n`;
          }
        });
        chatPrompt += `<|im_start|>user\n${finalUserMessageContent}<|im_end|>\n<|im_start|>assistant\n`;
        await invoke("run_exiq_chat", {
          modelName: selectedModel,
          prompt: chatPrompt,
          maxTokens: 1024,
          contextSize: numCtx || 4096,
          temperature: temperature,
          topP: topP,
          repeatPenalty: repeatPenalty,
          numThreads: numThreads,
          device: computeDevice || "auto",
          isNewChat: false,
          onToken: tokenChannel
        });
        setIsGenerating(false); 
        return;
      }
      //cloud
      let response: Response;
      if (isGeminiModel) {
        if (!geminiApiKey.trim()) {
          throw new Error("Gemini API Key is missing. Please add it in Core Settings.");
        }
        const geminiContents = activeChat.messages.map((m) => {
          const parts: any[] = [{ text: m.text }];
          if (m.image) {
            const rawBase64 = m.image.split(",")[1];
            parts.push({
              inlineData: {
                mimeType: "image/jpeg",
                data: rawBase64
              }
            });
          }
          return {
            role: m.sender === "user" ? "user" : "model",
            parts: parts
          };
        });
        const lastMsgParts: any[] = [{ text: finalUserMessageContent }];
        if (imageToSend) {
          lastMsgParts.push({
            inlineData: {
              mimeType: "image/jpeg",
              data: imageToSend
            }
          });
        }
        geminiContents.push({
          role: "user",
          parts: lastMsgParts
        });
        const geminiRequestBody: any = {
          contents: geminiContents,
          systemInstruction: {
            parts: [{ text: systemPrompt }]
          },
          generationConfig: {
            temperature: temperature
          }
        };
        if (useSearchGrounding) {
          geminiRequestBody.tools = [{ googleSearch: {} }];
        }
        response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:streamGenerateContent?alt=sse&key=${geminiApiKey}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify(geminiRequestBody)
          }
        );
      }
      //exis.exe(gguf)
      else {
        response = await fetch("http://127.0.0.1:11500/api/chat", {
          method: "POST",
          signal: controller.signal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedModel,
            messages: formattedHistory,
            stream: true,
            options: {
              num_thread: numThreads,
              temperature: temperature,
              repeat_penalty: repeatPenalty,
              presence_penalty: presencePenalty,
              frequency_penalty: frequencyPenalty,
              top_p: topP,
              top_k: topK,
              num_ctx: numCtx
            }
          })
        });
      }
      if (!response.body) {
        throw new Error("Empty body from API stream");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let partialLine = "";
      let fullTextAccumulator = "";
      let thinkingAccumulator = "";
      let responseTextAccumulator = "";
      let calculatedSpeed = 0;
      const startTime = Date.now();
      let tokenCount = 0;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + chunk).split("\n");
        partialLine = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            let token = "";
            if (isGeminiModel) {
              if (line.startsWith("data: ")) {
                const jsonStr = line.slice(6).trim();
                const parsed = JSON.parse(jsonStr);
                token = parsed.candidates?.[0]?.content?.parts?.[0]?.text || "";
                tokenCount++;
                const elapsedSeconds = (Date.now() - startTime) / 1000;
                calculatedSpeed = tokenCount / elapsedSeconds;
                const grounding = parsed.candidates?.[0]?.groundingMetadata;
                if (grounding && grounding.groundingChunks && grounding.groundingChunks.length > 0) {
                  const googleSources: Source[] = [];
                  grounding.groundingChunks.forEach((chunk: any) => {
                    if (chunk.web && chunk.web.uri) {
                      googleSources.push({
                        type: "web",
                        title: chunk.web.title || "Google Search Reference",
                        url: chunk.web.uri,
                        snippet: "Verified real-time Google search source"
                      });
                    }
                  });
                  if (googleSources.length > 0) {
                    activeMessageSources = [...messageSources, ...googleSources];
                  }
                }
              }
            } else {
              const parsed = JSON.parse(line);
              if (parsed.done && parsed.eval_count && parsed.eval_duration) {
                const count = parsed.eval_count;
                const durationNs = parsed.eval_duration;
                calculatedSpeed = count / (durationNs / 1e9);
              }
              token = parsed.message?.content || "";
            }
            fullTextAccumulator += token;
            if (fullTextAccumulator.includes("<thinking>")) {
              const parts = fullTextAccumulator.split("<thinking>");
              const textBeforeThinking = parts[0];
              const rest = parts[1] || "";
              if (rest.includes("</thinking>")) {
                const subParts = rest.split("</thinking>");
                thinkingAccumulator = subParts[0];
                responseTextAccumulator = textBeforeThinking + (subParts[1] || "");
              } else {
                thinkingAccumulator = rest;
                responseTextAccumulator = textBeforeThinking;
              }
            } else if (fullTextAccumulator.includes("<thought>")) {
              const parts = fullTextAccumulator.split("<thought>");
              const textBeforeThinking = parts[0];
              const rest = parts[1] || "";
              if (rest.includes("</thought>")) {
                const subParts = rest.split("</thought>");
                thinkingAccumulator = subParts[0];
                responseTextAccumulator = textBeforeThinking + (subParts[1] || "");
              } else {
                thinkingAccumulator = rest;
                responseTextAccumulator = textBeforeThinking;
              }
            } else {
              responseTextAccumulator = fullTextAccumulator;
            }
            setChats((prevChats) =>
              prevChats.map((c) => {
                if (c.id === activeChatId) {
                  return {
                    ...c,
                    messages: c.messages.map((m) => {
                      if (m.id === assistantMessageId) {
                        return {
                          ...m,
                          text: responseTextAccumulator,
                          thinking: thinkingAccumulator || undefined,
                          speed: calculatedSpeed > 0 ? calculatedSpeed : undefined
                        };
                      }
                      return m;
                    })
                  };
                }
                return c;
              })
            );
          } catch (err) {
            console.error("Failed to parse stream line:", err);
          }
        }
      }
    } catch (error) {
      console.error("Generation error:", error);
      setChats((prevChats) =>
        prevChats.map((c) => {
          if (c.id === activeChatId) {
            return {
              ...c,
              messages: c.messages.map((m) => {
                if (m.id === assistantMessageId) {
                  return {
                    ...m,
                    text: `SYSTEM_ERROR: ${error}`
                  };
                }
                return m;
              })
            };
          }
          return c;
        })
      );
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
      triggerMemoryExtractor(userText, matchedMemories);
    }
  };
const triggerMemoryExtractor = async (userInput: string, activeMemories: Memory[]) => {
    const dbContext = activeMemories
      .map((m) => `ID: "${m.id}", Text: "${m.text}", Tags: ${m.tags.join(",")}`)
      .join("\n");
    const prompt = `You are the memory manager for exis.Architect. Your job is to analyze the User's input and determine if it contains a new personal fact to remember, updates/appends to an existing stored fact, or deletes a fact.
Current Database Facts:
${dbContext || "No current facts."}
User Input: "${userInput}"
CRITICAL RULES FOR "none" (DO NOT TRIGGER ON CASUAL CHAT!):
- If the User's input is a question, you MUST set "action" to "none".
- If the User is just greeting you or saying short reactions like "ok", "cool", "привет", "ага", "ясно", you MUST set "action" to "none".
- ONLY set "action" to "add" or "update" if the user is EXPLICITLY stating a new concrete, permanent personal fact about themselves.
RULES FOR UPDATE/APPEND:
- If the User is adding a similar, complementary, or additional fact to something already stored, you MUST update the existing fact.
  1. Set "action" to "update".
  2. Set "delete_id" to the exact ID of the old fact (match strictly from the database list above).
  3. Set "add_text" to the COMBINED/CONSOLIDATED statement in Russian (e.g. "Пользователь любит апельсины и бананы").
Respond ONLY with a valid JSON object matching this schema (do NOT write any conversational text or thinking tags, just JSON):
{
  "action": "add", "update", "delete", or "none",
  "add_text": "the clean combined or new factual statement to remember, written in Russian",
  "delete_id": "the exact ID of the memory being replaced/deleted (match strictly from the database list above)",
  "delete_text": "the exact text of the memory being deleted/replaced",
  "tags": ["array", "of", "relevant", "short", "tags", "in", "english"]
}`;
    const isGeminiModel = selectedModel.startsWith("gemini-");
    let responseText = "";
    try {
      if (isGeminiModel) {
        if (!geminiApiKey.trim()) return;
        const res = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:generateContent?key=${geminiApiKey}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              contents: [{ role: "user", parts: [{ text: prompt }] }],
              generationConfig: {
                responseMimeType: "application/json", 
                temperature: 0.1
              }
            })
          }
        );
        const data = await res.json();
        responseText = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
      } 
      else {
        const res = await fetch("http://127.0.0.1:11500/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedModel,
            messages: [{ role: "user", content: prompt }],
            format: "json",
            stream: false,
            options: {
              temperature: 0.1,
              num_thread: numThreads
            }
          })
        });
        const data = await res.json();
        responseText = data.message?.content || "";
      }
      let cleanText = responseText.trim();
      if (cleanText.startsWith("```")) {
        cleanText = cleanText
          .replace(/^```json\s*/i, "")
          .replace(/```$/, "")        
          .trim();
      }
      const parsed = JSON.parse(cleanText);
      if (parsed.action && parsed.action !== "none") {
        setPendingChange({
          action: parsed.action,
          addText: parsed.add_text,
          deleteId: parsed.delete_id,
          deleteText: parsed.delete_text || activeMemories.find((m) => m.id === parsed.delete_id)?.text,
          tags: parsed.tags
        });
      }
    } catch (e) {
      console.error("Background memory extraction failed:", e);
    }
  };
const handleCreateModel = async () => {
    if (!newModelName.trim() || !newModelGgufPath.trim()) {
      setCreateModelError("All fields are required.");
      return;
    }
    setIsCreatingModel(true);
    setCreateModelError("");
    try {
      if (newModelGgufPath.toLowerCase().endsWith(".exiq")) {
        const registeredModelName = await invoke<string>("import_exiq_model", {
          name: newModelName,
          filePath: newModelGgufPath
        });
        setModels((prev) => Array.from(new Set([registeredModelName, ...prev])));
        setSelectedModel(registeredModelName);
        alert(`ExisQuant Model "${registeredModelName}" imported successfully!`);
      } 
      else {
        await invoke("create_local_model", { 
          name: newModelName, 
          ggufPath: newModelGgufPath 
        });
        const res = await fetch("http://127.0.0.1:11500/api/tags");
        const data = await res.json();
        if (data.models) {
          const names = data.models.map((m: any) => m.name);
          setModels(names);
          setSelectedModel(newModelName);
        }
        alert(`Model "${newModelName}" created successfully!`);
      }
      setNewModelName("");
      setNewModelGgufPath("");
      setIsCreateModelOpen(false);
    } catch (e: any) {
      console.error("Failed to create model:", e);
      setCreateModelError(e.toString());
    } finally {
      setIsCreatingModel(false);
    }
  };
  const handleBrowseFile = async () => {
    try {
      const selectedPath = await invoke<string | null>("pick_model_file");
      if (selectedPath) {
        setNewModelGgufPath(selectedPath);
        const fileName = selectedPath.split("\\").pop()?.split("/").pop() || "";
        const baseName = fileName.replace(/\.(exiq|gguf)$/i, "");
        if (!newModelName) {
          setNewModelName(baseName);
        }
      }
    } catch (e) {
      console.error("Failed to pick file:", e);
    }
  };
  const handleCaptureScreen = async () => {
    try {
      const base64Url = await invoke<string>("capture_screen");
      setScannerImage(base64Url);
    } catch (e) {
      console.error("Failed to capture screen:", e);
      alert(`Failed to capture screen:\n${e}`);
    }
  };
const handleMouseUp = () => {
    if (!cropStart || !cropCurrent || !scannerImage) return;
    const x = Math.min(cropStart.x, cropCurrent.x);
    const y = Math.min(cropStart.y, cropCurrent.y);
    const width = Math.abs(cropStart.x - cropCurrent.x);
    const height = Math.abs(cropStart.y - cropCurrent.y);
    if (width < 10 || height < 10) {
      setCropStart(null);
      setCropCurrent(null);
      return;
    }
    const img = new Image();
    img.src = scannerImage;
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(img, x, y, width, height, 0, 0, width, height);
        const croppedBase64 = canvas.toDataURL("image/jpeg");
        setImagePreview(croppedBase64);
        setAttachedImage(croppedBase64.split(",")[1]);
      }
      setScannerImage(null);
      setCropStart(null);
      setCropCurrent(null);
      getCurrentWindow().setFullscreen(false);
    };
  };
  const handleCommitChange = async () => {
    if (!pendingChange) return;
    let updatedMemories = [...exisDB.memories];
    if (pendingChange.action === "delete" || pendingChange.action === "update") {
      updatedMemories = updatedMemories.filter((m) => m.id !== pendingChange.deleteId);
    }
    if (pendingChange.action === "add" || pendingChange.action === "update") {
      const newMemory: Memory = {
        id: `mem-${Date.now()}`,
        text: pendingChange.addText || "",
        tags: pendingChange.tags || [],
        timestamp: new Date().toISOString()
      };
      updatedMemories.push(newMemory);
    }
    const updatedDB: ExisDB = { memories: updatedMemories };
    try {
      await invoke("write_exis_db", { data: JSON.stringify(updatedDB) });
      setExisDB(updatedDB);
    } catch (e) {
      console.error("Failed to write updated exisdb.json:", e);
    } finally {
      setPendingChange(null);
    }
  };
     if (windowLabel === "overlay") {
       return (
         <OverlayHUD
           models={models}
           selectedModel={selectedModel}
           setSelectedModel={setSelectedModel}
           systemPrompt={systemPrompt}
           numThreads={numThreads}
           temperature={temperature}
           repeatPenalty={repeatPenalty}
           geminiApiKey={geminiApiKey}
           setIsCreateModelOpen={setIsCreateModelOpen}
           handleCaptureScreen={handleCaptureScreen}
           throttleInBg={throttleInBg}
           throttlePercent={throttlePercent}
           presencePenalty={presencePenalty}
           frequencyPenalty={frequencyPenalty}
           topP={topP}
           topK={topK}
           numCtx={numCtx}
         />
       );
     }
  return (
    <div className="main-layout">
      {}
      <aside className={`sidebar ${!isSidebarOpen ? "collapsed" : ""}`}>
        <div className="new-chat-section">
          <button className="new-chat-btn" onClick={handleCreateNewChat}>
            NEW SCHEMATIC
          </button>
        </div>
        <div className="chat-list">
          {chats.map((c) => (
            <div
              key={c.id}
              className={`chat-item ${c.id === activeChatId ? "active" : ""}`}
              onClick={() => setActiveChatId(c.id)}
            >
              <span>[ {c.title} ]</span>
              {}
              {chats.length > 1 && (
                <button 
                  className="chat-delete-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteChat(c.id);
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          ))}
        </div>
        {}
          <div className="sidebar-footer">
          <button className="settings-btn" onClick={() => setIsSettingsOpen(true)}>
            CORE SETTINGS
          </button>
          {}
          <div className="sidebar-branding">
            created by examegd(ex.)
          </div>
        </div>
      </aside>
      {}
      <div className="chat-container">
        <header className="header">
          <div className="logo-section">
            <button 
              className="sidebar-toggle-btn"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            >
              {isSidebarOpen ? "◀" : "▶"}
            </button>
            <div className="logo-glyph"></div>
            <span className="title">exis.Architect</span>
          </div>
          {}
          <div className="model-selector-container">
            <span className="model-label">ACTIVE_SYSTEM:</span>
            <select
              className="model-select"
              value={selectedModel}
              onChange={(e) => {
                if (e.target.value === "CREATE_MODEL") {
                  setIsCreateModelOpen(true);
                } else {
                  setSelectedModel(e.target.value);
                }
              }}
              disabled={isGenerating}
            >
              <option value="CREATE_MODEL">[ CREATE_MODEL... ]</option>
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          {}
          <div className="system-monitor">
            <div className="metric">CPU: <span>{cpuUsage.toFixed(1)}%</span></div>
            <div className="metric">RAM: <span>{ramUsage.toFixed(1)} GB</span></div>
            <div className="metric">PORT: <span>11500</span></div>
          </div>
        </header>
        {}
        <main className="chat-area">
          {activeChat.messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.sender}`}>
              {}
              {msg.thinking && <ThinkingBlock content={msg.thinking} />}
              {}
              {msg.image && (
                <img src={msg.image} className="message-image" alt="user attachment" />
              )}
              {}
              {msg.sources && <SourcesBlock sources={msg.sources} />}
              <div className="message-text">
                {renderMessageContent(msg.text)}
              </div>
              {}
              {msg.speed && (
                <div className="generation-speed">
                  COMPUTATION_SPEED: {msg.speed.toFixed(1)} T/S
                </div>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </main>
        {pendingChange && (
          <div className="pending-change-box">
            <div className="change-header">
              <div className="change-header-left">
                <PrecursorGlyph />
                <span>MEMORIAL_UPDATE_PROTOCOL_PENDING...</span>
              </div>
              <span style={{ color: "#00bcff" }}>EXISDB_SYNC</span>
            </div>
            <div className="change-content">
              {}
              {(pendingChange.action === "delete" || pendingChange.action === "update") && (
                <div className="change-delete">
                  - [DELETE] "{pendingChange.deleteText}"
                </div>
              )}
              {}
              {(pendingChange.action === "add" || pendingChange.action === "update") && (
                <div className="change-add">
                  + [ADD] "{pendingChange.addText}"
                </div>
              )}
              {pendingChange.tags && pendingChange.tags.length > 0 && (
                <div className="change-tags-line">
                  PROPOSED TAGS: <span>{pendingChange.tags.map(t => `#${t}`).join(", ")}</span>
                </div>
              )}
            </div>
            <div className="change-actions">
              {}
              <button className="abort-btn" onClick={() => setPendingChange(null)}>
                ABORT_CHANGES
              </button>
              {}
              <button className="commit-btn" onClick={handleCommitChange}>
                COMMIT_CHANGES
              </button>
            </div>
          </div>
        )}
        {}
        {imagePreview && (
          <div className="image-preview-container">
            <img src={imagePreview} className="preview-thumbnail" alt="preview" />
            <div className="preview-info">
              IMAGE_ATTACHED: <span>IMAGE_READY_FOR_ANALYSIS</span>
            </div>
            <button className="preview-remove-btn" onClick={handleRemoveImage}>
              REMOVE
            </button>
          </div>
        )}
        {}
        <footer className="input-panel">
          {}
          <input 
            type="file"
            ref={fileInputRef}
            style={{ display: "none" }}
            accept="image/*"
            onChange={handleImageUpload}
          />
          {}
          <button 
            className="attach-file-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Attach Image"
            disabled={isGenerating}
          >
            <PaperclipIcon />
          </button>
          {}
          <button 
            className={`search-toggle-btn ${isWebSearchEnabled ? "active" : ""}`}
            onClick={() => {
              if (!tavilyApiKey) {
                alert("Please enter your Tavily Search API Key in Core Settings first.");
                setIsSettingsOpen(true);
              } else {
                setIsWebSearchEnabled(!isWebSearchEnabled);
              }
            }}
            title="Toggle Web Search"
            disabled={isGenerating}
          >
            <GlobeIcon />
          </button>
          {selectedModel.startsWith("gemini-") && (
            <label 
              style={{ 
                display: "flex", 
                alignItems: "center", 
                gap: "6px", 
                cursor: "pointer", 
                fontSize: "10px", 
                fontFamily: "monospace", 
                color: "#00ff88",
                userSelect: "none",
                marginRight: "5px"
              }}
            >
              <input
                type="checkbox"
                style={{ accentColor: "var(--glow-color)", cursor: "pointer" }}
                checked={useSearchGrounding}
                onChange={(e) => setUseSearchGrounding(e.target.checked)}
              />
              <span>Google Grounding</span>
            </label>
          )}
          <input
            type="text"
            className="chat-input"
            placeholder={isGenerating ? "COMPUTING RESPONSE..." : `EXECUTE DIRECTIVE ON ${selectedModel || "MODEL"}...`}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
              e.preventDefault();
              handleSend();
              }
            }}
            disabled={isGenerating}
          />
          <button 
            className={`send-button ${isGenerating ? "generating" : ""}`}
            onMouseEnter={() => setIsButtonHovered(true)}
            onMouseLeave={() => setIsButtonHovered(false)}
            onClick={isGenerating ? handleAbortGeneration : handleSend}
            disabled={!isGenerating && !inputValue.trim()}
          >
            {isGenerating ? (isButtonHovered ? "STOP" : "...") : "EXECUTE"}
          </button>
        </footer>
      </div>
      {isCreateModelOpen && (
        <div className="modal-overlay">
          <div className="modal-content" style={{ width: "500px" }}>
            <div className="modal-header">
              <span className="modal-title">MODEL_COMPILER_PROTOCOL</span>
              <button 
                className="modal-close-btn" 
                onClick={() => setIsCreateModelOpen(false)}
                disabled={isCreatingModel}
              >
                ABORT
              </button>
            </div>
            {createModelError && (
              <div style={{ color: "#ff4a4a", fontSize: "11px", fontFamily: "monospace", marginBottom: "15px" }}>
                ERROR: {createModelError}
              </div>
            )}
            <div className="settings-group">
              <label className="settings-label">Model Identifier (Name):</label>
              <input
                type="text"
                className="settings-input"
                placeholder="e.g. my-tinyllama or custom-qwen"
                value={newModelName}
                onChange={(e) => setNewModelName(e.target.value)}
                disabled={isCreatingModel}
              />
            </div>
            <div className="settings-group" style={{ marginTop: "15px" }}>
              <label className="settings-label">Model File Path (.exiq / .gguf):</label>
              <div style={{ display: "flex", gap: "8px" }}>
                <input
                  type="text"
                  className="settings-input"
                  placeholder="Select model file..."
                  value={newModelGgufPath}
                  onChange={(e) => setNewModelGgufPath(e.target.value)}
                  disabled={isCreatingModel}
                />
                <button
                  className="send-button"
                  style={{ borderColor: "#00bcff", color: "#00bcff", padding: "0 14px" }}
                  onClick={handleBrowseFile}
                  disabled={isCreatingModel}
                  type="button"
                >
                  BROWSE
                </button>
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "25px" }}>
              <button 
                className="send-button"
                onClick={handleCreateModel}
                disabled={isCreatingModel}
              >
                {isCreatingModel ? "PROCESSING..." : "REGISTER MODEL"}
              </button>
            </div>
          </div>
        </div>
      )}
      {}
      {isSettingsOpen && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <span className="modal-title">CORE PROTOCOL CONFIG</span>
              <button 
                className="modal-close-btn" 
                onClick={() => setIsSettingsOpen(false)}
              >
                SAVE & CLOSE
              </button>
            </div>
            {}
            <div className="settings-grid">
              {}
              <div className="settings-col-left">
                {}
                <div className="settings-group">
                  <label className="settings-label">System Directive (Prompt):</label>
                  <textarea
                    className="settings-textarea"
                    placeholder="Enter system instructions for the model..."
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                  />
                </div>
                {}
                <div className="settings-row">
                  <div className="settings-group">
                    <label className="settings-label">CPU Threads (Min 1):</label>
                    <input
                      type="number"
                      className="settings-input"
                      value={numThreads}
                      onChange={(e) => setNumThreads(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    />
                  </div>
                <div className="settings-group" style={{ marginTop: "8px" }}>
                  <label className="settings-label" style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", textTransform: "uppercase" }}>
                    <input
                      type="checkbox"
                      style={{ accentColor: "var(--glow-color)", cursor: "pointer" }}
                      checked={throttleInBg}
                      onChange={(e) => setThrottleInBg(e.target.checked)}
                    />
                    <span>Throttle computation when minimized to tray</span>
                  </label>
                  {}
                  {throttleInBg && (
                    <div style={{ marginTop: "10px", paddingLeft: "20px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                        <span className="settings-label" style={{ fontSize: "9px" }}>REDUCTION LEVEL:</span>
                        <span style={{ fontSize: "11px", color: "var(--glow-color)", fontFamily: "monospace", fontWeight: "bold" }}>
                          {throttlePercent}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="90"
                        step="5"
                        style={{ 
                          accentColor: "var(--glow-color)", 
                          width: "100%", 
                          height: "4px", 
                          background: "rgba(0, 0, 0, 0.4)", 
                          borderRadius: "2px", 
                          outline: "none", 
                          cursor: "pointer" 
                        }}
                        value={throttlePercent}
                        onChange={(e) => setThrottlePercent(Number(e.target.value))}
                      />
                    </div>
                  )}
                </div>
                  <div className="settings-group">
                    <label className="settings-label">Context Size (Min 512):</label>
                    <input
                      type="number"
                      className="settings-input"
                      value={numCtx}
                      onChange={(e) => setNumCtx(Math.max(512, parseInt(e.target.value, 10) || 512))}
                    />
                  </div>
                </div>
              <div className="settings-group" style={{ borderBottom: "1px solid var(--border-glow)", paddingBottom: "12px", marginBottom: "12px" }}>
                <label className="settings-label" style={{ color: "#00ff88", fontWeight: "bold" }}>
                  COMPUTE BACKEND (EXISQUANT):
                </label>
                <select
                  className="settings-input"
                  value={computeDevice}
                  onChange={(e) => setComputeDevice(e.target.value)}
                >
                  <option value="auto">🟢 AUTO {gpuName ? `(${gpuName})` : "(CPU Fallback)"}</option>
                  <option value="cuda">🚀 NVIDIA CUDA {gpuName ? `(${gpuName})` : ""}</option>
                  <option value="cpu">💻 CPU (AVX2 SIMD)</option>
                </select>
              </div>
                {}
                <div className="settings-row">
                  <div className="settings-group">
                    <label className="settings-label">Temperature (0 - 2):</label>
                    <input
                      type="number"
                      step="0.1"
                      className="settings-input"
                      value={temperature}
                      onChange={(e) => setTemperature(Math.min(2, Math.max(0, parseFloat(e.target.value) || 0)))}
                    />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">Repeat Penalty:</label>
                    <input
                      type="number"
                      step="0.1"
                      className="settings-input"
                      value={repeatPenalty}
                      onChange={(e) => setRepeatPenalty(Math.max(0, parseFloat(e.target.value) || 0))}
                    />
                  </div>
                </div>
                {}
                <div className="settings-row">
                  <div className="settings-group">
                    <label className="settings-label">Top-P (0 - 1):</label>
                    <input
                      type="number"
                      step="0.05"
                      className="settings-input"
                      value={topP}
                      onChange={(e) => setTopP(Math.min(1, Math.max(0, parseFloat(e.target.value) || 0)))}
                    />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">Top-K (Min 1):</label>
                    <input
                      type="number"
                      className="settings-input"
                      value={topK}
                      onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    />
                  </div>
                </div>
                {}
                <div className="settings-row">
                  <div className="settings-group">
                    <label className="settings-label">Presence Penalty:</label>
                    <input
                      type="number"
                      step="0.1"
                      className="settings-input"
                      value={presencePenalty}
                      onChange={(e) => setPresencePenalty(Math.max(0, parseFloat(e.target.value) || 0))}
                    />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">Frequency Penalty:</label>
                    <input
                      type="number"
                      step="0.1"
                      className="settings-input"
                      value={frequencyPenalty}
                      onChange={(e) => setFrequencyPenalty(Math.max(0, parseFloat(e.target.value) || 0))}
                    />
                  </div>
                </div>
              </div>
              {}
              <div className="settings-col-right">
                {}
                <div className="settings-group">
                  <label className="settings-label" style={{ color: "#00bcff" }}>Tavily Search API Key:</label>
                  <input
                    type="password"
                    className="settings-input"
                    placeholder="tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                    value={tavilyApiKey}
                    onChange={(e) => setTavilyApiKey(e.target.value)}
                  />
                </div>
                {}
                <div className="settings-group">
                  <label className="settings-label" style={{ color: "#00ff88" }}>GOOGLE API KEY (Cloud Models)</label>
                  <input
                    type="password"
                    className="settings-input"
                    placeholder="Paste your free Gemini API Key (starts with AIzaSy...)"
                    value={geminiApiKey}
                    onChange={(e) => setGeminiApiKey(e.target.value)}
                  />
                </div>
                {}
                <div className="settings-group" style={{ borderTop: "1px solid var(--border-glow)", paddingTop: "12px", marginTop: "5px" }}>
                  <label className="settings-label" style={{ color: "#00bcff" }}>MEMORY DATABASE (exisdb.json)</label>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <textarea
                      className="settings-textarea"
                      style={{ minHeight: "36px" }}
                      placeholder="Fact to remember..."
                      value={newMemoryText}
                      onChange={(e) => setNewMemoryText(e.target.value)}
                    />
                    <div style={{ display: "flex", gap: "8px" }}>
                      <input
                        type="text"
                        className="settings-input"
                        placeholder="Tags (separated by commas)..."
                        value={newMemoryTags}
                        onChange={(e) => setNewMemoryTags(e.target.value)}
                      />
                      <button 
                        className="send-button" 
                        style={{ borderColor: "#00bcff", color: "#00bcff", padding: "0 10px" }}
                        onClick={handleAddMemory}
                      >
                        ADD
                      </button>
                    </div>
                  </div>
                  {}
                  <div style={{ maxHeight: "80px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "4px" }}>
                    {exisDB.memories.length === 0 ? (
                      <div style={{ fontSize: "10px", color: "var(--text-muted)", fontStyle: "italic" }}>
                        No memories stored.
                      </div>
                    ) : (
                      exisDB.memories.map((mem) => (
                        <div 
                          key={mem.id} 
                          style={{ 
                            padding: "4px 8px", 
                            background: "rgba(255, 255, 255, 0.01)", 
                            border: "1px solid rgba(0, 188, 255, 0.1)", 
                            borderRadius: "4px" 
                          }}
                        >
                          <div style={{ fontSize: "10px", color: "var(--text-main)" }}>{mem.text}</div>
                          <div style={{ display: "flex", gap: "4px", marginTop: "2px", flexWrap: "wrap" }}>
                            {mem.tags.map((tag) => (
                              <span 
                                key={tag} 
                                style={{ 
                                  fontSize: "8px", 
                                  background: "rgba(0, 188, 255, 0.1)", 
                                  color: "#00bcff", 
                                  padding: "1px 4px", 
                                  borderRadius: "2px", 
                                  fontFamily: "monospace" 
                                }}
                              >
                                #{tag}
                              </span>
                            ))}
                          </div>
                          {}
      {scannerImage && (
        <div 
          className="scanner-overlay"
          style={{ backgroundImage: `url(${scannerImage})` }}
          onMouseDown={(e) => {
            setCropStart({ x: e.clientX, y: e.clientY });
            setCropCurrent({ x: e.clientX, y: e.clientY });
          }}
          onMouseMove={(e) => {
            if (cropStart) {
              setCropCurrent({ x: e.clientX, y: e.clientY });
            }
          }}
          onMouseUp={handleMouseUp}
        >
          <div className="scanner-mask">
            <div className="scanner-hint">
              DRAG MOUSE TO CROP REGION FOR AI INFERENCE (SURFACE_SCANNER_PROTOCOL)
            </div>
            {}
            {cropStart && cropCurrent && (
              <div 
                className="scanner-selection"
                style={{
                  left: `${Math.min(cropStart.x, cropCurrent.x)}px`,
                  top: `${Math.min(cropStart.y, cropCurrent.y)}px`,
                  width: `${Math.abs(cropStart.x - cropCurrent.x)}px`,
                  height: `${Math.abs(cropStart.y - cropCurrent.y)}px`
                }}
              />
            )}
          </div>
        </div>
      )}
      {}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
