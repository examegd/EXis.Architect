//go:build windows

package client

import (
	"errors"
	"io"
)

// Изменили: теперь функция вызывается без параметров
func QuantizeSupported() bool {
	return false
}

// Изменили: теперь функция возвращает []byte вместо io.Reader
func quantizeTensor(r io.Reader, tensorName, dtype string, shape []int32, quantize string) ([]byte, error) {
	return nil, errors.New("quantization is not supported on Windows")
}

func quantizePackedGroup(groupName string, tensors any) ([]byte, error) {
	return nil, errors.New("quantization is not supported on Windows")
}
