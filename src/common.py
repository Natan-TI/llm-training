"""Funções compartilhadas: prompt, parsing de rótulos e carregamento do modelo.

Importe este módulo ANTES de transformers/datasets/peft: as variáveis de ambiente abaixo precisam
estar definidas antes do huggingface_hub ser carregado.
"""

import os

# O backend de download "hf_xet" exige instruções de CPU ausentes em alguns processadores e derruba o
# processo com "Fatal Error: HW capability found ...". Usamos o download HTTP tradicional.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"   # adaptadores LoRA treinados (não versionados)
RESULTS_DIR = ROOT / "results"   # métricas, logs e gráficos (versionados)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["negativo", "neutro", "positivo"]

SYSTEM_PROMPT = (
    "Você é um classificador de sentimento de avaliações de produtos de e-commerce brasileiro. "
    "Responda apenas com uma palavra: positivo, negativo ou neutro."
)


def rating_to_label(rating: int) -> str:
    """Nota 1-2 -> negativo, 3 -> neutro, 4-5 -> positivo."""
    if rating <= 2:
        return "negativo"
    if rating == 3:
        return "neutro"
    return "positivo"


def build_messages(review: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Avaliação: {review.strip()}\n\nQual é o sentimento?"},
    ]


def parse_label(text: str) -> str | None:
    """Extrai o primeiro rótulo válido da resposta do modelo (None se inválida)."""
    text = text.lower()
    found = [(m.start(), lbl) for lbl in LABELS for m in [re.search(rf"\b{lbl}", text)] if m]
    return min(found)[1] if found else None


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA não disponível: este projeto exige GPU NVIDIA com PyTorch CUDA.")
    return torch.device("cuda")


def load_model(model_name: str = DEFAULT_MODEL, adapter: str | None = None):
    """Carrega modelo base em bf16 na GPU e, opcionalmente, um adaptador LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # necessário para geração em lote
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[list[dict]], max_new_tokens: int = 8,
             temperature: float = 0.0, top_p: float = 1.0) -> list[str]:
    """Gera respostas em lote para uma lista de conversas (formato chat)."""
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    sampling = temperature > 0
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=sampling,
        temperature=temperature if sampling else None,
        top_p=top_p if sampling else None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
