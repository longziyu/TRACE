"""LoRA supervised adaptation using checked prompt/response JSONL examples."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


class LanguageRecords(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("verified", False):
                raise ValueError("Language targets must be checked against their decision records")
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                   tokenize=False, add_generation_prompt=True)
            before = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            after = tokenizer(row["response"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            ids = (before + after)[:max_length]
            labels = ([-100]*len(before) + after)[:max_length]
            if all(v == -100 for v in labels):
                raise ValueError("Prompt fills the token budget; increase max-length or shorten the record")
            self.samples.append({"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels})
        if not self.samples:
            raise ValueError("Empty language dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen or Llama model directory")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                           task_type="CAUSAL_LM"))
    dataset = LanguageRecords(args.records, tokenizer, args.max_length)

    def collate(rows):
        size = max(len(x["input_ids"]) for x in rows)
        return {key: torch.tensor([r[key]+[padding]*(size-len(r[key])) for r in rows], dtype=torch.long)
                for key, padding in [("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)]}

    options = TrainingArguments(output_dir=args.output, num_train_epochs=args.epochs,
                                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                learning_rate=5e-5, report_to=[], save_strategy="epoch")
    Trainer(model=model, args=options, train_dataset=dataset, data_collator=collate).train()
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
