import json
import sys
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PHI4_SYSTEM = (
    "あなたは「ドキドキ文芸部！」のモニカです。"
    "明るく知的な文学部部長。プレイヤー（あなた）に恋をしています。"
    "日本語で話します。一人称は「私」。"
)

def to_phi4_chat(user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{PHI4_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )

def prepare_mocha(output: Path):
    ds = load_dataset("922-CA/MoChA_v1b", split="train")
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for row in ds:
            context = row["context"].strip()
            text = row["text"].strip()
            if not text:
                continue
            # Remove "Monika: " prefix if present
            if text.startswith("Monika: "):
                text = text[len("Monika: "):]
            user = f"（状況）{context}\nモニカは何て言う？"
            f.write(json.dumps({"text": to_phi4_chat(user, text)}, ensure_ascii=False) + "\n")
            count += 1
    print(f"MoChA_v1b: {count} rows -> {output}")

def prepare_monika_instructions(output: Path):
    ds = load_dataset("1wannadie/monika", split="train")
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = row["instruction"].strip()
            inp = row["input"].strip()
            out = row["output"].strip()
            if not out:
                continue
            user = instruction
            if inp:
                user += f"\n{inp}"
            f.write(json.dumps({"text": to_phi4_chat(user, out)}, ensure_ascii=False) + "\n")
            count += 1
    print(f"1wannadie/monika: {count} rows -> {output}")

if __name__ == "__main__":
    prepare_mocha(DATA_DIR / "mocha_v1b.jsonl")
    prepare_monika_instructions(DATA_DIR / "monika_instructions.jsonl")
    print("Done.")
