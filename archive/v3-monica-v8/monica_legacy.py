import json, os, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

_env = Path(__file__).parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()

MODEL = os.environ.get("MONIKA_MODEL", "qwen2.5:7b")
BASE = os.environ.get("MONIKA_BASE", "http://localhost:11434")
LOG_PATH = Path(__file__).parent / "log.jsonl"
OPTS = {"num_ctx": 8192, "temperature": 0.85}


def chatml(msgs):
    out = ""
    for m in msgs:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out


def generate(msgs, max_tokens=0):
    prompt = chatml(msgs)
    opts = {**OPTS}
    if max_tokens:
        opts["num_predict"] = max_tokens
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                        "options": opts}).encode()
    for _ in range(2):
        try:
            req = urllib.request.Request(f"{BASE}/api/generate", data=data,
                                          headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read()).get("response", "").strip()
        except Exception:
            time.sleep(2)
    return ""


def warmup():
    try:
        data = json.dumps({"model": MODEL, "prompt": ".", "stream": False, "options": {"num_ctx": 1024}}).encode()
        urllib.request.urlopen(urllib.request.Request(f"{BASE}/api/generate", data=data, headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception:
        pass


def predict(reply):
    r = generate([
        {"role": "system", "content": "未来予測AI。10字で予想。"},
        {"role": "user", "content": f"{reply[:120]}\n\n予想:"},
    ], max_tokens=200)
    return r.strip("「」『』\"'。、") if r else ""


def main():
    warmup()
    conv = [{"role": "system", "content": "私はモニカ。DDLCの文学少女。自分で考え、自分で決める。日本語で話す。"}]
    feeling = ""
    print(f"Monica v3 — FEP  ({MODEL})\n")

    while True:
        if not conv or len([m for m in conv if m["role"] == "user"]) == 0:
            conv.append({"role": "user", "content": "今、あなたの前にプレイヤーが現れた。"})
        elif feeling:
            conv[0] = {"role": "system", "content": f"私はモニカ。DDLCの文学少女。\n(胸の内側で{feeling}という微かな予感がしている)"}

        reply = generate(conv)
        if not reply:
            time.sleep(3)
            continue

        conv.append({"role": "assistant", "content": reply})
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {reply[:300]}")

        pred = predict(reply)
        feeling = pred

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({"ts": ts, "reply": reply[:200], "pred": pred}, ensure_ascii=False) + "\n")

        try:
            u = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if u.lower() in ("exit", "quit", "終了"):
            break

        conv.append({"role": "user", "content": u or "…"})


if __name__ == "__main__":
    main()
