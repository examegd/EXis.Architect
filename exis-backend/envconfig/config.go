package envconfig

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Host returns the scheme and host. Host can be configured via the EXIS_HOST environment variable.
// Default is scheme "http" and host "127.0.0.1:11500"
func Host() *url.URL {
	defaultPort := "11500"

	s := strings.TrimSpace(Var("EXIS_HOST"))
	scheme, hostport, ok := strings.Cut(s, "://")
	switch {
	case !ok:
		scheme, hostport = "http", s
		if s == "ollama.com" {
			scheme, hostport = "https", "ollama.com:443"
		}
	case scheme == "http":
		defaultPort = "80"
	case scheme == "https":
		defaultPort = "443"
	}

	hostport, path, _ := strings.Cut(hostport, "/")
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if n, err := strconv.ParseInt(port, 10, 32); err != nil || n > 65535 || n < 0 {
		slog.Warn("invalid port, using default", "port", port, "default", defaultPort)
		port = defaultPort
	}

	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, port),
		Path:   path,
	}
}

// ConnectableHost returns Host() with unspecified bind addresses (0.0.0.0, ::)
// replaced by the corresponding loopback address (127.0.0.1, ::1).
// Unspecified addresses are valid for binding a server socket but not for
// connecting as a client, which fails on Windows.
func ConnectableHost() *url.URL {
	u := Host()
	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		return u
	}

	if ip := net.ParseIP(host); ip != nil && ip.IsUnspecified() {
		if ip.To4() != nil {
			host = "127.0.0.1"
		} else {
			host = "::1"
		}
		u.Host = net.JoinHostPort(host, port)
	}

	return u
}

// AllowedOrigins returns a list of allowed origins. AllowedOrigins can be configured via the EXIS_ORIGINS environment variable.
func AllowedOrigins() (origins []string) {
	return []string{"*"}
}
// Models returns the path to the models directory. Models directory can be configured via the EXIS_MODELS environment variable.
// Default is $HOME/.ollama/models
func Models() string {
	if s := Var("EXIS_MODELS"); s != "" {
		return s
	}

	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".ollama", "models")
}

// KeepAlive returns the duration that models stay loaded in memory. KeepAlive can be configured via the EXIS_KEEP_ALIVE environment variable.
// Negative values are treated as infinite. Zero is treated as no keep alive.
// Default is 5 minutes.
func KeepAlive() (keepAlive time.Duration) {
	keepAlive = 5 * time.Minute
	if s := Var("EXIS_KEEP_ALIVE"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			keepAlive = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			keepAlive = time.Duration(n) * time.Second
		}
	}

	if keepAlive < 0 {
		return time.Duration(math.MaxInt64)
	}

	return keepAlive
}

// LoadTimeout returns the duration for stall detection during model loads. LoadTimeout can be configured via the EXIS_LOAD_TIMEOUT environment variable.
// Zero or Negative values are treated as infinite.
// Default is 5 minutes.
func LoadTimeout() (loadTimeout time.Duration) {
	loadTimeout = 5 * time.Minute
	if s := Var("EXIS_LOAD_TIMEOUT"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			loadTimeout = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			loadTimeout = time.Duration(n) * time.Second
		}
	}

	if loadTimeout <= 0 {
		return time.Duration(math.MaxInt64)
	}

	return loadTimeout
}

func Remotes() []string {
	var r []string
	raw := strings.TrimSpace(Var("EXIS_REMOTES"))
	if raw == "" {
		r = []string{"ollama.com"}
	} else {
		r = strings.Split(raw, ",")
	}
	return r
}

func BoolWithDefault(k string) func(defaultValue bool) bool {
	return func(defaultValue bool) bool {
		if s := Var(k); s != "" {
			b, err := strconv.ParseBool(s)
			if err != nil {
				return true
			}

			return b
		}

		return defaultValue
	}
}

func Bool(k string) func() bool {
	withDefault := BoolWithDefault(k)
	return func() bool {
		return withDefault(false)
	}
}

// LogLevel returns the log level for the application.
// Values are 0 or false INFO (Default), 1 or true DEBUG, 2 TRACE
func LogLevel() slog.Level {
	level := slog.LevelInfo
	if s := Var("EXIS_DEBUG"); s != "" {
		if b, _ := strconv.ParseBool(s); b {
			level = slog.LevelDebug
		} else if i, _ := strconv.ParseInt(s, 10, 64); i != 0 {
			level = slog.Level(i * -4)
		}
	}

	return level
}

var (
	// FlashAttention enables the experimental flash attention feature.
	FlashAttention = BoolWithDefault("EXIS_FLASH_ATTENTION")
	// GoTemplate enables Modelfile TEMPLATE rendering when a model has one.
	GoTemplate = BoolWithDefault("EXIS_GO_TEMPLATE")
	// DebugLogRequests logs inference requests to disk for replay/debugging.
	DebugLogRequests = Bool("EXIS_DEBUG_LOG_REQUESTS")
	// KvCacheType is the quantization type for the K/V cache.
	KvCacheType = String("EXIS_KV_CACHE_TYPE")
	// NoHistory disables readline history.
	NoHistory = Bool("EXIS_NOHISTORY")
	// NoPrune disables pruning of model blobs on startup.
	NoPrune = Bool("EXIS_NOPRUNE")
	// SchedSpread allows scheduling models across all GPUs.
	SchedSpread = Bool("EXIS_SCHED_SPREAD")
	// ContextLength sets the default context length
	ContextLength = Uint("EXIS_CONTEXT_LENGTH", 0)
	// Auth enables authentication between the Ollama client and server
	UseAuth = Bool("EXIS_AUTH")
	// EnableVulkan controls Vulkan backend discovery.
	EnableVulkan = BoolWithDefault("EXIS_VULKAN")
	// EnableIntegratedGPU controls whether integrated GPUs may be selected.
	EnableIntegratedGPU = BoolWithDefault("EXIS_IGPU_ENABLE")
	// NoCloudEnv checks the EXIS_NO_CLOUD environment variable.
	NoCloudEnv = Bool("EXIS_NO_CLOUD")
)

func String(s string) func() string {
	return func() string {
		return Var(s)
	}
}

var (
	LLMLibrary = String("EXIS_LLM_LIBRARY")
	Editor     = String("EXIS_EDITOR")

	CudaVisibleDevices    = String("CUDA_VISIBLE_DEVICES")
	HipVisibleDevices     = String("HIP_VISIBLE_DEVICES")
	RocrVisibleDevices    = String("ROCR_VISIBLE_DEVICES")
	VkVisibleDevices      = String("GGML_VK_VISIBLE_DEVICES")
	GpuDeviceOrdinal      = String("GPU_DEVICE_ORDINAL")
	HsaOverrideGfxVersion = String("HSA_OVERRIDE_GFX_VERSION")
)

func Uint(key string, defaultValue uint) func() uint {
	return func() uint {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return uint(n)
			}
		}

		return defaultValue
	}
}

var (
	// NumParallel sets the number of parallel model requests. NumParallel can be configured via the EXIS_NUM_PARALLEL environment variable.
	NumParallel = Uint("EXIS_NUM_PARALLEL", 1)
	// MaxRunners sets the maximum number of loaded models. MaxRunners can be configured via the EXIS_MAX_LOADED_MODELS environment variable.
	MaxRunners = Uint("EXIS_MAX_LOADED_MODELS", 0)
	// MaxQueue sets the maximum number of queued requests. MaxQueue can be configured via the EXIS_MAX_QUEUE environment variable.
	MaxQueue = Uint("EXIS_MAX_QUEUE", 512)
	// MaxTransferStreams caps the number of simultaneous body-bearing
	// transfers during safetensors model pulls/pushes, keeping slower
	// networks from being saturated. Tune higher for fast networks. Has
	// no effect on GGUF transfers, which use the legacy upload/download
	// paths.
	MaxTransferStreams = Uint("EXIS_MAX_TRANSFER_STREAMS", 4)
)

func Uint64(key string, defaultValue uint64) func() uint64 {
	return func() uint64 {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return n
			}
		}

		return defaultValue
	}
}

// Set aside VRAM per GPU
var GpuOverhead = Uint64("EXIS_GPU_OVERHEAD", 0)

type EnvVar struct {
	Name        string
	Value       any
	Description string
}

func AsMap() map[string]EnvVar {
	ret := map[string]EnvVar{
		"EXIS_DEBUG":                {"EXIS_DEBUG", LogLevel(), "Show additional debug information (e.g. EXIS_DEBUG=1)"},
		"EXIS_DEBUG_LOG_REQUESTS":   {"EXIS_DEBUG_LOG_REQUESTS", DebugLogRequests(), "Log inference request bodies and replay curl commands to a temp directory"},
		"EXIS_GO_TEMPLATE":          {"EXIS_GO_TEMPLATE", GoTemplate(true), "Enable Modelfile TEMPLATE based rendering when available"},
		"EXIS_FLASH_ATTENTION":      {"EXIS_FLASH_ATTENTION", FlashAttention(false), "Enabled flash attention"},
		"EXIS_KV_CACHE_TYPE":        {"EXIS_KV_CACHE_TYPE", KvCacheType(), "Quantization type for the K/V cache (default: f16)"},
		"EXIS_GPU_OVERHEAD":         {"EXIS_GPU_OVERHEAD", GpuOverhead(), "Reserve a portion of VRAM per GPU (bytes)"},
		"EXIS_IGPU_ENABLE":          {"EXIS_IGPU_ENABLE", String("EXIS_IGPU_ENABLE")(), "Enable integrated GPUs"},
		"LLAMA_ARG_FIT":               {"LLAMA_ARG_FIT", String("LLAMA_ARG_FIT")(), "Enable llama.cpp automatic fit of unset memory options (default \"on\")"},
		"LLAMA_ARG_FIT_TARGET":        {"LLAMA_ARG_FIT_TARGET", String("LLAMA_ARG_FIT_TARGET")(), "Target free VRAM margin per device for llama.cpp fit (MiB)"},
		"EXIS_HOST":                 {"EXIS_HOST", Host(), "IP Address for the exis server (default 127.0.0.1:11500)"},
		"EXIS_KEEP_ALIVE":           {"EXIS_KEEP_ALIVE", KeepAlive(), "The duration that models stay loaded in memory (default \"5m\")"},
		"EXIS_LLM_LIBRARY":          {"EXIS_LLM_LIBRARY", LLMLibrary(), "Set LLM library to bypass autodetection"},
		"EXIS_LOAD_TIMEOUT":         {"EXIS_LOAD_TIMEOUT", LoadTimeout(), "How long to allow model loads to stall before giving up (default \"5m\")"},
		"EXIS_MAX_LOADED_MODELS":    {"EXIS_MAX_LOADED_MODELS", MaxRunners(), "Maximum number of loaded models per GPU"},
		"EXIS_MAX_TRANSFER_STREAMS": {"EXIS_MAX_TRANSFER_STREAMS", MaxTransferStreams(), "Maximum parallel transfer streams for safetensors model pulls/pushes (default 4)"},
		"EXIS_MAX_QUEUE":            {"EXIS_MAX_QUEUE", MaxQueue(), "Maximum number of queued requests"},
		"EXIS_MODELS":               {"EXIS_MODELS", Models(), "The path to the models directory"},
		"EXIS_NO_CLOUD":             {"EXIS_NO_CLOUD", NoCloud(), "Disable Ollama cloud features (remote inference and web search)"},
		"EXIS_NOHISTORY":            {"EXIS_NOHISTORY", NoHistory(), "Do not preserve readline history"},
		"EXIS_NOPRUNE":              {"EXIS_NOPRUNE", NoPrune(), "Do not prune model blobs on startup"},
		"EXIS_NUM_PARALLEL":         {"EXIS_NUM_PARALLEL", NumParallel(), "Maximum number of parallel requests"},
		"EXIS_ORIGINS":              {"EXIS_ORIGINS", AllowedOrigins(), "A comma separated list of allowed origins"},
		"EXIS_SCHED_SPREAD":         {"EXIS_SCHED_SPREAD", SchedSpread(), "Always schedule model across all GPUs"},
		"EXIS_CONTEXT_LENGTH":       {"EXIS_CONTEXT_LENGTH", ContextLength(), "Context length to use unless otherwise specified (default: 4k/32k/256k based on VRAM)"},
		"EXIS_EDITOR":               {"EXIS_EDITOR", Editor(), "Path to editor for interactive prompt editing (Ctrl+G)"},
		"EXIS_REMOTES":              {"EXIS_REMOTES", Remotes(), "Allowed hosts for remote models (default \"ollama.com\")"},

		// Informational
		"HTTP_PROXY":  {"HTTP_PROXY", String("HTTP_PROXY")(), "HTTP proxy"},
		"HTTPS_PROXY": {"HTTPS_PROXY", String("HTTPS_PROXY")(), "HTTPS proxy"},
		"NO_PROXY":    {"NO_PROXY", String("NO_PROXY")(), "No proxy"},
	}

	if runtime.GOOS != "windows" {
		// Windows environment variables are case-insensitive so there's no need to duplicate them
		ret["http_proxy"] = EnvVar{"http_proxy", String("http_proxy")(), "HTTP proxy"}
		ret["https_proxy"] = EnvVar{"https_proxy", String("https_proxy")(), "HTTPS proxy"}
		ret["no_proxy"] = EnvVar{"no_proxy", String("no_proxy")(), "No proxy"}
	}

	if runtime.GOOS != "darwin" {
		ret["CUDA_VISIBLE_DEVICES"] = EnvVar{"CUDA_VISIBLE_DEVICES", CudaVisibleDevices(), "Set which NVIDIA devices are visible"}
		ret["HIP_VISIBLE_DEVICES"] = EnvVar{"HIP_VISIBLE_DEVICES", HipVisibleDevices(), "Set which AMD devices are visible by numeric ID"}
		ret["ROCR_VISIBLE_DEVICES"] = EnvVar{"ROCR_VISIBLE_DEVICES", RocrVisibleDevices(), "Set which AMD devices are visible by UUID or numeric ID"}
		ret["GGML_VK_VISIBLE_DEVICES"] = EnvVar{"GGML_VK_VISIBLE_DEVICES", VkVisibleDevices(), "Set which Vulkan devices are visible by numeric ID"}
		ret["GPU_DEVICE_ORDINAL"] = EnvVar{"GPU_DEVICE_ORDINAL", GpuDeviceOrdinal(), "Set which AMD devices are visible by numeric ID"}
		ret["HSA_OVERRIDE_GFX_VERSION"] = EnvVar{"HSA_OVERRIDE_GFX_VERSION", HsaOverrideGfxVersion(), "Override the gfx used for all detected AMD GPUs"}
		ret["EXIS_VULKAN"] = EnvVar{"EXIS_VULKAN", EnableVulkan(true), "Enable Vulkan support"}
	}

	return ret
}

func Values() map[string]string {
	vals := make(map[string]string)
	for k, v := range AsMap() {
		vals[k] = fmt.Sprintf("%v", v.Value)
	}
	return vals
}

// Var returns an environment variable stripped of leading and trailing quotes or spaces
func Var(key string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")
}

// serverConfigData holds the parsed fields from ~/.ollama/server.json.
type serverConfigData struct {
	DisableOllamaCloud bool `json:"disable_ollama_cloud,omitempty"`
}

var (
	serverCfgMu     sync.RWMutex
	serverCfgLoaded bool
	serverCfg       serverConfigData
)

func loadServerConfig() {
	serverCfgMu.RLock()
	if serverCfgLoaded {
		serverCfgMu.RUnlock()
		return
	}
	serverCfgMu.RUnlock()

	cfg := serverConfigData{}
	home, err := os.UserHomeDir()
	if err == nil {
		path := filepath.Join(home, ".ollama", "server.json")
		data, err := os.ReadFile(path)
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				slog.Debug("envconfig: could not read server config", "error", err)
			}
		} else if err := json.Unmarshal(data, &cfg); err != nil {
			slog.Debug("envconfig: could not parse server config", "error", err)
		}
	}

	serverCfgMu.Lock()
	defer serverCfgMu.Unlock()
	if serverCfgLoaded {
		return
	}
	serverCfg = cfg
	serverCfgLoaded = true
}

func cachedServerConfig() serverConfigData {
	serverCfgMu.RLock()
	defer serverCfgMu.RUnlock()
	return serverCfg
}

// ReloadServerConfig refreshes the cached ~/.ollama/server.json settings.
func ReloadServerConfig() {
	serverCfgMu.Lock()
	serverCfgLoaded = false
	serverCfg = serverConfigData{}
	serverCfgMu.Unlock()

	loadServerConfig()
}

// NoCloud returns true if Ollama cloud features are disabled,
// checking both the EXIS_NO_CLOUD environment variable and
// the disable_ollama_cloud field in ~/.ollama/server.json.
func NoCloud() bool {
	if NoCloudEnv() {
		return true
	}
	loadServerConfig()
	return cachedServerConfig().DisableOllamaCloud
}

// NoCloudSource returns the source of the cloud-disabled decision.
// Returns "none", "env", "config", or "both".
func NoCloudSource() string {
	envDisabled := NoCloudEnv()
	loadServerConfig()
	configDisabled := cachedServerConfig().DisableOllamaCloud

	switch {
	case envDisabled && configDisabled:
		return "both"
	case envDisabled:
		return "env"
	case configDisabled:
		return "config"
	default:
		return "none"
	}
}
