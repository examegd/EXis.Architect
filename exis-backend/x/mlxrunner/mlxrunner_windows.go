//go:build windows

package mlxrunner

import (
	"context"
	"errors"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

type Client struct{}

// Новая сигнатура с поддержкой размера контекста NumCtx
func NewClient(modelName string, numCtx int) (*Client, error) {
	return nil, errors.New("mlx runner is not supported on Windows")
}

func Execute(args []string) error {
	return errors.New("mlx runner is not supported on Windows")
}

// Реализация интерфейса llm.LlamaServer для Windows
func (c *Client) ModelPath() string { return "" }
func (c *Client) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	return nil, errors.New("mlx is not supported on Windows")
}
func (c *Client) Ping(ctx context.Context) error             { return nil }
func (c *Client) WaitUntilRunning(ctx context.Context) error { return nil }
func (c *Client) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	return nil
}
func (c *Client) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	return nil
}
func (c *Client) ApplyChatTemplate(ctx context.Context, req llm.ChatRequest) (string, error) {
	return "", nil
}
func (c *Client) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, nil
}
func (c *Client) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, nil
}
func (c *Client) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", nil
}
func (c *Client) Close() error                                       { return nil }
func (c *Client) MemorySize() (total, vram uint64)                   { return 0, 0 }
func (c *Client) VRAMByGPU(id ml.DeviceID) uint64                    { return 0 }
func (c *Client) Pid() int                                           { return 0 }
func (c *Client) HasExited() bool                                    { return true }
func (c *Client) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo { return nil }
func (c *Client) ContextLength() int                                 { return 0 }
func (c *Client) GetPort() int                                       { return -1 }
