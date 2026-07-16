//go:build windows

package imagegen

import (
	"context"
	"errors"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

type Server struct{}

func NewServer(modelName string) (*Server, error) {
	return nil, errors.New("image generation is not supported on Windows")
}

func Execute(args []string) error {
	return errors.New("image generation is not supported on Windows")
}

// Реализация интерфейса llm.LlamaServer для Windows
func (s *Server) ModelPath() string { return "" }
func (s *Server) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	return nil, errors.New("image generation is not supported on Windows")
}
func (s *Server) Ping(ctx context.Context) error             { return nil }
func (s *Server) WaitUntilRunning(ctx context.Context) error { return nil }
func (s *Server) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	return nil
}
func (s *Server) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	return nil
}
func (s *Server) ApplyChatTemplate(ctx context.Context, req llm.ChatRequest) (string, error) {
	return "", nil
}
func (s *Server) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, nil
}
func (s *Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, nil
}
func (s *Server) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", nil
}
func (s *Server) Close() error                                       { return nil }
func (s *Server) MemorySize() (total, vram uint64)                   { return 0, 0 }
func (s *Server) VRAMByGPU(id ml.DeviceID) uint64                    { return 0 }
func (s *Server) Pid() int                                           { return 0 }
func (s *Server) HasExited() bool                                    { return true }
func (s *Server) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo { return nil }
func (s *Server) ContextLength() int                                 { return 0 }
func (s *Server) GetPort() int                                       { return -1 }
