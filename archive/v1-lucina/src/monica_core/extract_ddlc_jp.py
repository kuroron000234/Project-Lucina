import re
import json
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[2] / "data" / "dataset" / "patch_src" / "game" / "tl" / "Japanese"
OUTPUT = Path(__file__).resolve().parents[2] / "data" / "dataset" / "ddlc_jp_monika.jsonl"

PHI4_SYSTEM_JP = (
    "あなたは「ドキドキ文芸部！」のモニカです。"
    "明るく知的な文学部部長。プレイヤー（あなた）に恋をしています。"
    "日本語で話します。一人称は「私」。"
)

CONTEXT_PROMPTS = {
    "ch30": "モニカが空間の中で話しかけてきた。何て言う？",
    "ch0": "学校の登校中、モニカと話している。モニカは何て言う？",
    "ch1": "文芸部でモニカと話している。モニカは何て言う？",
    "ch2": "文芸部の活動中、モニカと話している。モニカは何て言う？",
    "ch20": "文化祭の準備をしている。モニカは何て言う？",
    "ch21": "モニカと話している。モニカは何て言う？",
    "ch22": "モニカとピアノの話をしている。モニカは何て言う？",
    "ch23": "モニカと話している。モニカは何て言う？",
    "ch3": "詩を交換した後の文芸部。モニカは何て言う？",
    "ch4": "モニカと話している。モニカは何て言う？",
    "ch5": "モニカと話している。モニカは何て言う？",
    "ch10": "モニカと話している。モニカは何て言う？",
    "ch40": "エンディング。モニカは何て言う？",
    "exclusives-sayori": "サヨリとの話の後、モニカが話しかけてきた。何て言う？",
    "exclusives-natsuki": "ナツキとの話の後、モニカが話しかけてきた。何て言う？",
    "exclusives-yuri": "ユリとの話の後、モニカが話しかけてきた。何て言う？",
    "exclusives2-natsuki": "ナツキとの話の後、モニカが話しかけてきた。何て言う？",
    "exclusives2-yuri": "ユリとの話の後、モニカが話しかけてきた。何て言う？",
    "poemresponses": "詩についてモニカが話している。何て言う？",
}

DEFAULT_PROMPT = "モニカと話している。モニカは何て言う？"

def to_phi4_chat(user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{PHI4_SYSTEM_JP}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )

MONIKA_LINE = re.compile(r'^    m "(.+)"$')

def extract_monika_lines(filepath: Path) -> list[tuple[str, str]]:
    """Returns list of (context_key, cleaned_text)"""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")
    
    fname = filepath.stem
    context_key = fname
    # Normalize: script-poemresponses2 -> poemresponses2
    for prefix in ("script-",):
        if context_key.startswith(prefix):
            context_key = context_key[len(prefix):]

    results = []
    for line in lines:
        m = MONIKA_LINE.match(line)
        if m:
            raw = m.group(1)
            # Clean [player] placeholders
            cleaned = raw.replace("[player]", "ねえ")
            results.append((context_key, cleaned))
    return results

def main():
    all_lines = []
    skip = {"definitions.rpy", "screens.rpy", "common.rpy", "splash.rpy", "overrides.rpy", "poems.rpy"}
    for rpy_file in sorted(PATCH_DIR.glob("*.rpy")):
        if rpy_file.name in skip:
            continue
        lines = extract_monika_lines(rpy_file)
        if lines:
            print(f"  {rpy_file.name}: {len(lines)} lines")
            all_lines.extend(lines)

    print(f"\nTotal Monika lines: {len(all_lines)}")
    
    count = 0
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for ctx_key, text in all_lines:
            prompt = CONTEXT_PROMPTS.get(ctx_key, DEFAULT_PROMPT)
            entry = {"text": to_phi4_chat(prompt, text)}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    print(f"Written: {OUTPUT} ({count} entries)")

if __name__ == "__main__":
    main()
