"""
QLoRA fine-tuning of microsoft/phi-4 on Monika dialogue.
Uses 2-GPU model parallelism with manual device map to keep lm_head on same GPU as last layer.
"""
import json, os, argparse, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset/monika_lora_train.jsonl")
    p.add_argument("--output", default="./lora_monika")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--model", default="microsoft/phi-4")
    p.add_argument("--max_seq_length", type=int, default=1024)
    return p.parse_args()

def tokenize_fn(examples, tokenizer, max_length):
    tok = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

def main():
    args = parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Dataset: {len(rows)} entries")

    dataset = Dataset.from_list([{"text": r["text"]} for r in rows])
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize with padding to max_length for uniform tensors
    tokenized_train = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_eval = eval_ds.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        learning_rate=args.lr,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA saved to {args.output}")

if __name__ == "__main__":
    main()
