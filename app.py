"""Frontend (Gradio) para o usuário criar prompts e testar o modelo base vs. os modelos treinados.

Uso:
    python app.py            # abre em http://127.0.0.1:7860
"""

import json
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import (DEFAULT_MODEL, MODELS_DIR, OUTPUTS_DIR, RECOMMENDED_ADAPTER,  # noqa: E402
                    RESULTS_DIR, SYSTEM_PROMPT, build_messages, load_model, parse_label)

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from transformers import TextIteratorStreamer  # noqa: E402

BASE = "Modelo base (sem fine-tuning)"
EMOJI = {"positivo": "😊 POSITIVO", "negativo": "😠 NEGATIVO", "neutro": "😐 NEUTRO"}


def find_adapters(base_model: str) -> dict[str, Path]:
    """Adaptadores treinados sobre o mesmo modelo base: os versionados em models/ (disponíveis logo
    após clonar o repositório) e os gerados localmente em outputs/ pelos experimentos."""
    found = {}
    for folder in (MODELS_DIR, OUTPUTS_DIR):
        for d in sorted(folder.glob("*/adapter_config.json")):
            cfg = json.loads(d.read_text(encoding="utf-8"))
            if cfg.get("base_model_name_or_path") == base_model:
                found.setdefault(d.parent.name, d.parent)
    return found


print("Carregando modelo na GPU...")
adapters = find_adapters(DEFAULT_MODEL)
model, tokenizer = load_model(DEFAULT_MODEL)
if adapters:
    from peft import PeftModel

    names = list(adapters)
    model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(adapters[n], adapter_name=n)
    model.eval()
CHOICES = [BASE] + list(adapters)
DEFAULT_CHOICE = RECOMMENDED_ADAPTER if RECOMMENDED_ADAPTER in adapters else CHOICES[-1]
gpu_lock = threading.Lock()
print(f"Pronto. Modelos disponíveis: {CHOICES}")


def adapter_ctx(choice: str):
    """Seleciona o adaptador escolhido ou desliga o LoRA para usar o modelo base."""
    if not adapters:
        return nullcontext()
    if choice == BASE:
        return model.disable_adapter()
    model.set_adapter(choice)
    return nullcontext()


def gen_kwargs(temperature, top_p, max_new_tokens):
    kw = {"max_new_tokens": int(max_new_tokens), "pad_token_id": tokenizer.pad_token_id}
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kw


def stream_generate(messages, choice, temperature, top_p, max_new_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([text], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**enc, streamer=streamer, **gen_kwargs(temperature, top_p, max_new_tokens))

    def run():
        with gpu_lock, adapter_ctx(choice), torch.inference_mode():
            model.generate(**kw)

    threading.Thread(target=run, daemon=True).start()
    out, t0, n = "", time.time(), 0
    for piece in streamer:
        out += piece
        n += 1
        yield out, n, time.time() - t0


def classify(review, choice, temperature):
    if not review.strip():
        return "Digite uma avaliação.", "", ""
    raw, n, dt = "", 0, 0.0
    for raw, n, dt in stream_generate(build_messages(review), choice, temperature, 1.0, 8):
        pass
    label = parse_label(raw)
    shown = EMOJI.get(label, "❓ Resposta fora do formato esperado")
    return shown, raw.strip(), f"{dt * 1000:.0f} ms na GPU"


def free_prompt(prompt, system, choice, temperature, top_p, max_new_tokens):
    if not prompt.strip():
        yield "Digite um prompt.", ""
        return
    messages = ([{"role": "system", "content": system}] if system.strip() else []) + [
        {"role": "user", "content": prompt}]
    for out, n, dt in stream_generate(messages, choice, temperature, top_p, max_new_tokens):
        yield out, f"{n} tokens · {dt:.1f}s · {n / max(dt, 1e-6):.1f} tokens/s"


def compare_all(review):
    if not review.strip():
        return []
    rows = []
    for choice in CHOICES:
        label, raw, ms = classify(review, choice, 0.0)
        rows.append([choice, label, raw, ms])
    return rows


def gpu_status():
    p = torch.cuda.get_device_properties(0)
    lines = [f"**GPU:** {p.name}  ", f"**CUDA (PyTorch):** {torch.version.cuda} · "
             f"**PyTorch:** {torch.__version__}  ",
             f"**VRAM alocada pelo app:** {torch.cuda.memory_allocated() / 1024**3:.2f} GB  "]
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        lines += [f"**VRAM total em uso:** {mem.used / 1024**3:.2f} / {mem.total / 1024**3:.2f} GB  ",
                  f"**Utilização:** {util.gpu}%  ",
                  f"**Temperatura:** {pynvml.nvmlDeviceGetTemperature(h, 0)} °C  ",
                  f"**Consumo:** {pynvml.nvmlDeviceGetPowerUsage(h) / 1000:.0f} W"]
    except Exception:
        pass
    return "\n".join(lines)


def results_md():
    f = RESULTS_DIR / "comparison.md"
    return f.read_text(encoding="utf-8") if f.exists() else "Rode `python src/compare_results.py`."


EXAMPLES = [
    "Produto chegou antes do prazo e funciona perfeitamente. Recomendo!",
    "Veio com defeito e a loja não responde. Não comprem.",
    "É ok. Cumpre o que promete, mas o acabamento poderia ser melhor.",
    "A entrega atrasou duas semanas, mas pelo menos o produto é bom.",
    "Esperava mais pelo preço, achei mediano.",
]

with gr.Blocks(title="LLM Sentimento PT-BR") as demo:
    gr.Markdown("# 🧠 Classificador de Sentimento com LLM (fine-tuning LoRA)\n"
                f"Modelo base: `{DEFAULT_MODEL}` · Rodando em **{torch.cuda.get_device_name(0)}**")

    with gr.Tab("Classificar avaliação"):
        with gr.Row():
            with gr.Column(scale=3):
                review = gr.Textbox(label="Avaliação do produto", lines=5,
                                    placeholder="Escreva aqui uma avaliação de produto...")
                gr.Examples(EXAMPLES, inputs=review)
            with gr.Column(scale=2):
                model_choice = gr.Dropdown(CHOICES, value=DEFAULT_CHOICE, label="Modelo")
                temp_c = gr.Slider(0, 1.5, value=0, step=0.05, label="Temperatura (0 = determinístico)")
                btn = gr.Button("Classificar", variant="primary")
                label_out = gr.Textbox(label="Sentimento")
                raw_out = gr.Textbox(label="Resposta bruta do modelo")
                time_out = gr.Textbox(label="Tempo")
        btn.click(classify, [review, model_choice, temp_c], [label_out, raw_out, time_out])
        review.submit(classify, [review, model_choice, temp_c], [label_out, raw_out, time_out])

    with gr.Tab("Comparar modelos"):
        review_cmp = gr.Textbox(label="Avaliação", lines=4)
        gr.Examples(EXAMPLES, inputs=review_cmp)
        btn_cmp = gr.Button("Comparar todos os modelos", variant="primary")
        table = gr.Dataframe(headers=["Modelo", "Sentimento", "Resposta bruta", "Tempo"],
                             wrap=True)
        btn_cmp.click(compare_all, review_cmp, table)

    with gr.Tab("Prompt livre"):
        with gr.Row():
            with gr.Column(scale=3):
                system = gr.Textbox(label="System prompt (opcional)", value=SYSTEM_PROMPT, lines=2)
                prompt = gr.Textbox(label="Seu prompt", lines=6)
                btn_p = gr.Button("Gerar", variant="primary")
                answer = gr.Textbox(label="Resposta", lines=10)
                stats = gr.Markdown()
            with gr.Column(scale=2):
                model_p = gr.Dropdown(CHOICES, value=DEFAULT_CHOICE, label="Modelo")
                temp_p = gr.Slider(0, 1.5, value=0.7, step=0.05, label="Temperatura")
                top_p = gr.Slider(0.05, 1, value=0.9, step=0.05, label="Top-p")
                max_tok = gr.Slider(8, 1024, value=256, step=8, label="Máx. tokens novos")
        btn_p.click(free_prompt, [prompt, system, model_p, temp_p, top_p, max_tok], [answer, stats])

    with gr.Tab("GPU"):
        gpu_md = gr.Markdown(gpu_status())
        gr.Button("Atualizar").click(gpu_status, None, gpu_md)
        gr.Timer(2).tick(gpu_status, None, gpu_md)

    with gr.Tab("Resultados dos experimentos"):
        gr.Markdown(results_md())
        for img in ["comparison_f1.png", "loss_curves.png"]:
            if (RESULTS_DIR / img).exists():
                gr.Image(str(RESULTS_DIR / img), show_label=False)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
