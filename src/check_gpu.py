"""Mostra as informações da GPU e confirma que o PyTorch está usando CUDA.

Rode no terminal do VSCode e tire o print para a entrega:
    python src/check_gpu.py
"""

import platform
import time

import torch


def main():
    print("=" * 60)
    print("VERIFICAÇÃO DE GPU / CUDA")
    print("=" * 60)
    print(f"Python        : {platform.python_version()}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA (torch)  : {torch.version.cuda}")
    print(f"cuDNN         : {torch.backends.cudnn.version()}")
    print(f"CUDA ativo    : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n>>> CUDA NÃO disponível. Instale o PyTorch com CUDA (veja README).")
        return

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}         : {p.name}")
        print(f"Compute cap.  : {p.major}.{p.minor}")
        print(f"VRAM total    : {p.total_memory / 1024**3:.2f} GB")
        print(f"Multiprocess. : {p.multi_processor_count} SMs")

    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f"Driver NVIDIA : {pynvml.nvmlSystemGetDriverVersion()}")
        print(f"VRAM em uso   : {mem.used / 1024**3:.2f} GB")
        print(f"Temperatura   : {pynvml.nvmlDeviceGetTemperature(h, 0)} °C")
    except Exception as e:  # nvml é opcional
        print(f"(pynvml indisponível: {e})")

    print("\nTeste de desempenho (matmul 8192x8192 bf16)...")
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Desempenho    : {20 * 2 * 8192**3 / dt / 1e12:.1f} TFLOPS (bf16)")
    print("\n>>> GPU funcionando com CUDA!")


if __name__ == "__main__":
    main()
