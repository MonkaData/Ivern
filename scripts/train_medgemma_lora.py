"""Experimental MedGemma QLoRA training on the prepared RSNA JSONL files.

Run this on the RTX 4090 machine, not on a small local GPU unless you are only
testing argument parsing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import HF_TOKEN, MEDGEMMA_ID  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_messages(item: dict[str, Any], image: Any) -> tuple[list[dict], list[dict]]:
    prompt = item["messages"][0]["content"]
    answer = item["messages"][1]["content"]
    user = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    return user, full


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/finetune_medgemma_250")
    parser.add_argument("--output-dir", default="outputs/medgemma_lora_250")
    parser.add_argument("--model-id", default=MEDGEMMA_ID)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from peft import LoraConfig, prepare_model_for_kbit_training
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    class JsonlDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]], processor: Any, root: Path):
            self.rows = rows
            self.processor = processor
            self.root = root

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = self.rows[idx]
            image = Image.open(self.root / item["image_path"]).convert("RGB")
            user_messages, full_messages = _build_messages(item, image)
            full = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt = self.processor.apply_chat_template(
                user_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            example = {key: value.squeeze(0) for key, value in full.items()}
            labels = example["input_ids"].clone()
            prompt_len = min(prompt["input_ids"].shape[-1], labels.shape[-1])
            labels[:prompt_len] = -100
            example["labels"] = labels
            return example

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        out: dict[str, Any] = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key in {"input_ids", "attention_mask", "labels"}:
                padding_value = -100 if key == "labels" else (0 if key == "attention_mask" else pad_id)
                out[key] = pad_sequence(values, batch_first=True, padding_value=padding_value)
            else:
                out[key] = torch.stack(values) if values[0].shape == values[-1].shape else values
        if out["input_ids"].shape[1] > args.max_length:
            for key in ("input_ids", "attention_mask", "labels"):
                out[key] = out[key][:, : args.max_length]
        return out

    root = Path.cwd()
    data_dir = Path(args.data_dir)
    train_rows = _load_jsonl(data_dir / "train.jsonl")
    val_rows = _load_jsonl(data_dir / "val.jsonl")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    auth = {"token": HF_TOKEN} if HF_TOKEN else {}
    processor = AutoProcessor.from_pretrained(args.model_id, **auth)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
        **auth,
    )
    model = prepare_model_for_kbit_training(model)
    model.add_adapter(
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=JsonlDataset(train_rows, processor, root),
        eval_dataset=JsonlDataset(val_rows, processor, root),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[OK] LoRA adapter saved to {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
