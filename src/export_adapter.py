"""Exporta um adaptador LoRA treinado para models/, em bf16, para ser versionado no GitHub.

O treino salva os pesos LoRA em fp32 (~74 MB para r=16). Em bf16 — a mesma precisão usada na
inferência — o arquivo cai pela metade e fica abaixo do limite recomendado pelo GitHub (50 MB).

Uso:
    python src/export_adapter.py --run-name ft_lr2e-4_r16_ep1
"""

import argparse
import shutil

from common import MODELS_DIR, OUTPUTS_DIR

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    src, dst = OUTPUTS_DIR / args.run_name, MODELS_DIR / args.run_name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")

    weights = load_file(src / "adapter_model.safetensors")
    save_file({k: v.to(torch.bfloat16) for k, v in weights.items()}, dst / "adapter_model.safetensors",
              metadata={"format": "pt"})

    before = (src / "adapter_model.safetensors").stat().st_size / 1024**2
    after = (dst / "adapter_model.safetensors").stat().st_size / 1024**2
    print(f"{len(weights)} tensores | {before:.1f} MB (fp32) -> {after:.1f} MB (bf16) | salvo em {dst}")


if __name__ == "__main__":
    main()
