"""Fine-tuning com LoRA (PEFT) para classificação de sentimento, 100% na GPU.

Uso:
    python src/train.py --run-name lr2e-4_r16 --lr 2e-4 --lora-r 16 --epochs 1
"""

import argparse
import json
import time

from common import DATA_DIR, DEFAULT_MODEL, OUTPUTS_DIR, RESULTS_DIR, build_messages, require_gpu

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,  # noqa: E402
                          TrainingArguments, set_seed)


def tokenize_example(example, tokenizer, max_len):
    """Tokeniza prompt + resposta; a loss é calculada só nos tokens da resposta."""
    prompt = tokenizer.apply_chat_template(
        build_messages(example["review"]), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["label"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[: max_len - len(answer_ids)]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2, help="batch efetivo = batch-size * grad-accum")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=None, help="padrão: 2 * r")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--train-samples", type=int, default=None, help="limita o nº de exemplos de treino")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.lora_alpha = args.lora_alpha or 2 * args.lora_r

    require_gpu()
    set_seed(args.seed)
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} | VRAM {gpu.total_memory / 1024**3:.1f} GB | CUDA {torch.version.cuda}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Parâmetros treináveis: {trainable:,} de {total:,} ({100 * trainable / total:.2f}%)")

    ds = load_dataset("json", data_files={"train": str(DATA_DIR / "train.jsonl"),
                                          "val": str(DATA_DIR / "val.jsonl")})
    if args.train_samples:
        ds["train"] = ds["train"].select(range(args.train_samples))
    ds = ds.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_len": args.max_len},
                remove_columns=ds["train"].column_names)

    out_dir = OUTPUTS_DIR / args.run_name
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # float = fração dos steps (transformers v5)
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        train_sampling_strategy="group_by_length",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=Collator(tokenizer.pad_token_id))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    run_dir = RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "run_name": args.run_name,
        "model": args.model,
        "hyperparams": {k: v for k, v in vars(args).items() if k not in ("run_name", "model")},
        "train_examples": len(ds["train"]),
        "trainable_params": trainable,
        "total_params": total,
        "train_time_s": round(elapsed, 1),
        "peak_gpu_mem_gb": round(peak_mem, 2),
        "gpu": gpu.name,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "final_eval_loss": trainer.evaluate()["eval_loss"],
    }
    (run_dir / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (run_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2),
                                            encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Adaptador salvo em {out_dir}")


if __name__ == "__main__":
    main()
