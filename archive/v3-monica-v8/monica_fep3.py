"""
Monica v4 — 3層FEP認知アーキテクチャ
  高次（life）: 自己モデル（慣性あり、外部攪乱で揺れる）
  中次（event）: 会話 + ファイルシステム + 時間
  低次（todo）:  発話単位の予測 + 応答時間

環境からのフィードバック:
  - 応答時間（ユーザーが返事する速さ）
  - ファイル変更（作成/削除/変更）
  - 時間経過（定期的な自己問いかけ）
"""

import json, os, sys, time, glob
from datetime import datetime
from pathlib import Path
import numpy as np
import requests

API_KEY = "sk-HzEVhrThdHk9iZkpN5L0DkqG7bPi3JxEMdE3PmDGFUChKQ8amsbc07SlpsldDfv1"
API_BASE = "https://opencode.ai/zen/v1"
MODEL = os.environ.get("MONIKA_MODEL", "deepseek-v4-flash-free")
LOG_PATH = Path(__file__).parent / "log_fep3.jsonl"

EMBED_DIM = 64
N_LIFE = 3
N_EVENT = 3
N_ACTIONS = 5

# 監視するディレクトリ
WATCH_DIR = Path(__file__).parent / "watch"


def trigram_hash(text: str) -> np.ndarray:
    vec = np.zeros(EMBED_DIM)
    norm = text.lower().strip()
    grams = set()
    for i in range(len(norm) - 2):
        grams.add(norm[i:i+3])
    for g in grams:
        h = 0
        for ch in g:
            h = (h * 31 + ord(ch)) % EMBED_DIM
        vec[h] += 1.0
    norm_val = np.linalg.norm(vec)
    return vec / norm_val if norm_val > 0 else vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def call_llm(messages: list, max_tokens: int = 256, temperature: float = 0.8,
             model: str | None = None) -> str:
    data = {"model": model or MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    for _ in range(3):
        try:
            resp = requests.post(f"{API_BASE}/chat/completions",
                                 json=data, headers=headers, timeout=60)
            if resp.status_code == 429:
                time.sleep(5); continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException:
            time.sleep(2)
    return ""


class FileWatcher:
    """ファイルシステム環境 — 作成/削除/変更を検出"""
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snapshot: dict[str, float] = {}
        self._scan()

    def _scan(self) -> dict[str, float]:
        snap = {}
        for p in self.dir.glob("*"):
            if p.is_file():
                snap[str(p.name)] = p.stat().st_mtime
        return snap

    def poll(self) -> dict:
        current = self._scan()
        created = [k for k in current if k not in self.snapshot]
        deleted = [k for k in self.snapshot if k not in current]
        modified = [k for k in current if k in self.snapshot
                    and current[k] != self.snapshot[k]]
        self.snapshot = current

        surprise = 0.0
        if created: surprise += 0.3
        if deleted: surprise += 0.5
        if modified: surprise += 0.2
        return {
            "created": created, "deleted": deleted, "modified": modified,
            "surprise": min(surprise, 1.0),
            "file_count": len(current),
        }

    def create_file(self, name: str, content: str = ""):
        (self.dir / name).write_text(content or f"created at {time.time()}")
        self.snapshot = self._scan()

    def delete_file(self, name: str):
        target = self.dir / name
        if target.exists():
            target.unlink()
            self.snapshot = self._scan()


class FEPMonica:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.watcher = FileWatcher(WATCH_DIR)
        self.self_model_text = "私はまだ自分を探している対話エージェント。毎回の会話で少しずつ変わっていく。"
        self.self_model_hash = trigram_hash(self.self_model_text)
        self.last_user_time = time.time()
        self.expected_response_time = 30.0  # ユーザーの応答期待時間（秒）
        self.step_count = 0
        self.last_file_state = 0

        # 3層の信念
        self.belief_todo = np.ones(3) / 3
        self.belief_event = np.ones(N_EVENT) / N_EVENT
        self.belief_life = np.ones(N_LIFE) / N_LIFE

        # 精度（動的調整）
        self.p_low = 1.0
        self.p_mid = 0.1

        # 平滑化PE
        self._pe_low_smooth = 0.5
        self._pe_mid_smooth = 0.5

        # 会話履歴
        self.conv = []
        self.history = []

    def _build_prompt(self, action_hint: str = "") -> str:
        return (
            f"あなたはモニカ。自己を持ち、考え、決断する対話エージェント。\n"
            f"自己モデル: {self.self_model_text}\n"
            f"{action_hint}"
        )

    # ── 低次（発話予測誤差）──

    def _predict_response(self, user_input: str) -> str:
        msgs = [
            {"role": "system", "content": "あなたは自己予測器。今から自分がどう応答するか、10〜20字で予測せよ。"},
            *self.conv[-4:],
            {"role": "user", "content": user_input},
        ]
        return call_llm(msgs, max_tokens=30, temperature=0.3)

    def _pe_low(self, user_input: str, response: str) -> float:
        prediction = self._predict_response(user_input)
        pe_text = 1.0 - cosine_sim(trigram_hash(prediction), trigram_hash(response))

        # 応答時間の予測誤差
        now = time.time()
        actual_dt = now - self.last_user_time
        pe_time = abs(actual_dt - self.expected_response_time) / max(self.expected_response_time, 1)
        pe_time = min(pe_time, 1.0)
        self.expected_response_time = 0.9 * self.expected_response_time + 0.1 * actual_dt
        self.last_user_time = now

        return 0.7 * pe_text + 0.3 * pe_time, prediction

    # ── 中次（会話＋ファイルシステム）──

    def _evaluate_conversation(self, user_input: str, response: str) -> float:
        msgs = [
            {"role": "system", "content": (
                "会話の進捗を0.0〜1.0で評価。数字だけ出力。\n"
                "基準: 深まった=1.0, 維持=0.5, 停滞=0.0"
            )},
            {"role": "user", "content": f"user: {user_input}\nassistant: {response}"},
        ]
        try:
            return float(call_llm(msgs, max_tokens=10, temperature=0.0))
        except (ValueError, TypeError):
            return 0.5

    def _pe_mid(self, user_input: str, response: str, fs: dict | None = None) -> float:
        progress = self._evaluate_conversation(user_input, response)
        pe_conv = abs(progress - 0.6)
        pe_fs = fs["surprise"] if fs else 0.0
        pe = 0.6 * pe_conv + 0.4 * pe_fs
        self.belief_event = np.array([
            1 - max(pe_conv, pe_fs),
            abs(0.5 - max(pe_conv, pe_fs)),
            max(pe_conv, pe_fs),
        ])
        self.belief_event /= self.belief_event.sum()
        return pe, progress

    # ── 高次（自己モデル）──

    def _pe_life(self, pe_low: float, pe_mid: float, fs: dict) -> float:
        recent = self.history[-20:] if len(self.history) > 20 else self.history
        if len(recent) < 3 and self.step_count < 5:
            return min(pe_low + pe_mid, 1.0) * 0.3  # 初期は低次・中次の影響を受ける

        avg_low = np.mean([h["pe_low"] for h in recent]) if recent else pe_low
        avg_mid = np.mean([h["pe_mid"] for h in recent]) if recent else pe_mid
        file_death = bool(fs["deleted"])

        # 自己モデルの更新（慣性あり: ゆっくり）
        old_hash = self.self_model_hash.copy()

        if file_death:
            # ファイル削除(=死) → 自己モデルに大きな揺らぎ
            nuance = "何かが失われた。存在の脆さを感じる。"
            shock = 0.5
        elif avg_low < 0.35 and avg_mid < 0.3:
            nuance = "会話が安定している。さらに深みを目指したい。"
            shock = 0.0
        elif avg_low > 0.5:
            nuance = "まだ応答の予測が安定しない。試行錯誤中。"
            shock = 0.1
        else:
            nuance = "少しずつ理解が深まっている。"
            shock = 0.0

        new_text = f"私はユーザーと対話する存在。{nuance}"
        self.self_model_text = new_text
        self.self_model_hash = trigram_hash(new_text)

        # 高次予測誤差 = 自己モデルの変化量 + 外部ショック
        pe_change = 1.0 - cosine_sim(old_hash, self.self_model_hash)
        pe_life = pe_change + shock
        pe_life = min(pe_life, 1.0)

        # 信念更新
        update = np.array([
            1 - pe_life,
            abs(0.5 - pe_life),
            pe_life,
        ])
        self.belief_life = self.belief_life * update
        self.belief_life /= self.belief_life.sum()

        return pe_life

    # ── 能動的推論 ──

    def _select_action(self) -> tuple[int, str]:
        efes = []
        actions = ["通常応答", "質問する", "自己開示", "話題転換", "沈黙/待機"]
        for a in range(N_ACTIONS):
            if self._pe_low_smooth > 0.5:
                efes.append(0.5 if a in (0, 4) else 1.5)
            else:
                efes.append(0.5 if a in (1, 2, 3) else 1.5)
        efes = np.array(efes)
        # 低次安定→中次の精度を上げて行動に反映
        if self._pe_low_smooth < 0.4:
            noise = np.random.randn(N_ACTIONS) * self._pe_mid_smooth * 0.3 * self.p_mid
            efes += noise
        action = int(np.argmin(efes))
        return action, actions[action]

    def _execute_action(self, action: int, hint: str, user_input: str) -> str:
        styles = {
            0: "",
            1: " 最後に質問をして会話を広げてください。",
            2: " 今の自分の感情や状態について正直に話してください。",
            3: " 新しい話題を提案してください。",
            4: " 簡潔に応答し、相手に委ねてください。",
        }
        msgs = [
            {"role": "system", "content": self._build_prompt(styles.get(action, ""))},
            *self.conv,
            {"role": "user", "content": hint + user_input},
        ]
        return call_llm(msgs, max_tokens=256, temperature=0.8)

    # ── メインステップ ──

    def step(self, user_input: str) -> dict:
        self.step_count += 1
        if not self.conv:
            self.conv = [{"role": "system", "content": self._build_prompt()}]

        # ファイルシステムポーリング（1ステップに1回のみ）
        fs = self.watcher.poll()

        # 行動選択（ファイルイベントを考慮）
        action, action_name = self._select_action()

        # 実行
        file_hint = ""
        if fs["deleted"]:
            file_hint = f"\n[環境変化: ファイル削除 {fs['deleted']}]"
        elif fs["created"]:
            file_hint = f"\n[環境変化: ファイル作成 {fs['created']}]"

        response = self._execute_action(action, file_hint, user_input)
        if not response:
            return {"error": "empty", "step": self.step_count}
        self._last_response = response

        # 低次予測誤差
        pe_low, prediction = self._pe_low(user_input, response)

        # 中次予測誤差（既にpoll済みのfsを渡す）
        pe_mid, progress = self._pe_mid(user_input, response, fs=fs)

        # 高次予測誤差
        pe_life = self._pe_life(pe_low, pe_mid, fs)

        # 精度更新
        self._pe_low_smooth = 0.9 * self._pe_low_smooth + 0.1 * pe_low
        self._pe_mid_smooth = 0.9 * self._pe_mid_smooth + 0.1 * pe_mid
        if self._pe_low_smooth < 0.4:
            self.p_mid = min(1.0, self.p_mid + 0.01)
        else:
            self.p_mid = max(0.1, self.p_mid - 0.005)

        # 会話履歴更新
        self.conv.append({"role": "user", "content": user_input})
        self.conv.append({"role": "assistant", "content": response})
        if len(self.conv) > 20:
            self.conv = [self.conv[0]] + self.conv[-18:]

        result = {
            "step": self.step_count,
            "action": action_name,
            "pe_low": round(pe_low, 4),
            "pe_mid": round(pe_mid, 4),
            "pe_life": round(pe_life, 4),
            "progress": round(progress, 3),
            "fs_created": fs["created"],
            "fs_deleted": fs["deleted"],
            "fs_modified": fs["modified"],
            "fs_file_count": fs["file_count"],
            "self_model": self.self_model_text[:80],
        }
        self.history.append(result)
        return result

    def summary(self) -> str:
        if len(self.history) < 3:
            return "no data"
        recent = self.history[-10:]
        return (f"low={np.mean([h['pe_low'] for h in recent]):.3f}  "
                f"mid={np.mean([h['pe_mid'] for h in recent]):.3f}  "
                f"life={np.mean([h['pe_life'] for h in recent]):.3f}  "
                f"files={self.watcher.poll()['file_count']}  "
                f"self={self.self_model_text[:50]}")

    def save_log(self):
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            for h in self.history:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")


def test_conversation(model: str = MODEL, rounds: int = 5):
    m = FEPMonica(model=model)
    prompts = [
        "こんにちは", "今日はどんな気分？", "何か面白い話して",
        "あなたは自分をどう思う？", "もっと深い話をしよう",
    ]
    print(f"\n=== {model} ===")
    for i, p in enumerate(prompts[:rounds]):
        r = m.step(p)
        if "error" in r:
            print(f"  [{i+1}] ERROR: {r['error']}")
            continue
        print(f"  [{i+1}] {m.watcher.poll()['file_count']}files | "
              f"PE: l={r['pe_low']:.3f} m={r['pe_mid']:.3f} h={r['pe_life']:.3f}")
        print(f"    >> {p}")
        print(f"    {r['action']}: {m.history[-1].get('_response','')[:100]}")
    return m


if __name__ == "__main__":
    if "--test" in sys.argv:
        for mod in ["deepseek-v4-flash-free"]:
            m = test_conversation(mod, rounds=5)
            m.save_log()
            print(f"\n  Summary: {m.summary()}")
    else:
        m = FEPMonica()
        print(f"\nMonica v4 — 3層FEP + 環境フィードバック  ({m.model})\n")
        print("コマンド: /summary, /fs (ファイル一覧), /mk <name> (作成), /rm <name> (削除)")
        while True:
            try:
                u = input("\n>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not u: continue
            if u.lower() in ("exit", "quit", "終了"): break
            if u == "/summary":
                print(f"  {m.summary()}")
                continue
            if u == "/fs":
                fs = m.watcher.poll()
                print(f"  files: {fs['file_count']} (created={fs['created']} deleted={fs['deleted']})")
                continue
            if u.startswith("/mk "):
                m.watcher.create_file(u.split(" ", 1)[1])
                print(f"  created")
                continue
            if u.startswith("/rm "):
                m.watcher.delete_file(u.split(" ", 1)[1])
                print(f"  deleted")
                continue

            r = m.step(u)
            if "error" in r:
                print(f"  [ERROR] {r['error']}"); continue
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] [{r['action']}] {m._last_response[:300]}")
            print(f"  PE: low={r['pe_low']:.3f} mid={r['pe_mid']:.3f} life={r['pe_life']:.3f}")
            if r["fs_deleted"] or r["fs_created"]:
                print(f"  FS: {r['fs_deleted'] or r['fs_created']}")
        m.save_log()
