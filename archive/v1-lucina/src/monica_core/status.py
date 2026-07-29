"""Monika 状態ダッシュボード — 全データを一覧表示"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
SRC = BASE / "src"


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(n: float) -> str:
    return f"{n:.0f}"


def _bar(v: float, width: int = 12) -> str:
    filled = max(0, min(width, int(v / 100 * width)))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def system_section() -> list[str]:
    lines = ["## システム状態"]
    try:
        res = subprocess.run(
            ["systemctl", "is-active", "monika.service"],
            capture_output=True, text=True, timeout=5,
        )
        sim_status = res.stdout.strip()
    except Exception:
        sim_status = "不明"
    lines.append(f"  シミュレーション: {sim_status}")

    try:
        res = subprocess.run(
            ["pgrep", "-f", "telegram_bot"],
            capture_output=True, text=True, timeout=5,
        )
        bot_pids = res.stdout.strip().split()
        lines.append(f"  Telegram bot:      {'稼働中 PID:' + bot_pids[0] if bot_pids else '停止中'}")
    except Exception:
        lines.append("  Telegram bot:      不明")

    uptime = "?"
    try:
        res = subprocess.run(
            ["systemctl", "show", "monika.service", "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True, timeout=5,
        )
        if res.stdout:
            ts_str = res.stdout.split("=", 1)[1].strip()
            ts = datetime.strptime(ts_str.rsplit(" ", 1)[0], "%a %Y-%m-%d %H:%M:%S")
            elapsed = datetime.now() - ts
            hours, rem = divmod(int(elapsed.total_seconds()), 3600)
            mins = rem // 60
            uptime = f"{ hours}時間{mins}分"
    except Exception:
        pass
    lines.append(f"  稼働時間:          {uptime}")
    lines.append("")
    return lines


def sim_section() -> list[str]:
    state = _load_json(DATA / "state.json")
    if not state:
        return ["## シミュレーション状態", "  (データなし)\n"]

    lines = ["## シミュレーション状態"]
    t = state.get("time", "?")
    lines.append(f"  時刻: {t}")

    s = state.get("state", {})
    lines.append(f"  エネルギー: {_fmt(s.get('energy', 0))} {_bar(s.get('energy', 0))}")
    lines.append(f"  空腹:        {_fmt(s.get('hunger', 0))} {_bar(s.get('hunger', 0))}")
    lines.append(f"  疲労:        {_fmt(s.get('fatigue', 0))} {_bar(s.get('fatigue', 0))}")
    lines.append(f"  孤独:        {_fmt(s.get('loneliness', 0))} {_bar(s.get('loneliness', 0))}")
    lines.append(f"  精神:        {_fmt(s.get('spirit', 0))} {_bar(s.get('spirit', 0))}")

    act = state.get("current_activity")
    rem = state.get("activity_remaining", 0)
    if act:
        lines.append(f"  現在の活動: {act} (あと{rem}分)")
    lines.append(f"  現在位置: {state.get('current_room', '?')}")

    memory = _load_json(DATA / "memory_store.json")
    if memory:
        entries = memory if isinstance(memory, list) else memory.get("entries", [])
        lines.append(f"  記憶: {len(entries)}件")

    reading = _load_json(DATA / "reading_state.json")
    if reading:
        title = reading.get("title", "?")
        pos = reading.get("position", 0)
        total = reading.get("total_chars", 1)
        pct = pos / total * 100
        lines.append(f"  読書: 「{title}」 {pct:.0f}% ({pos:,}/{total:,}字)")
    lines.append("")
    return lines


def history_section(n: int = 10) -> list[str]:
    state = _load_json(DATA / "state.json")
    if not state:
        return []

    lines = [f"## 最近の履歴 (直近{n}件)"]
    history = state.get("history", [])
    for e in history[-n:]:
        bef = e.get("state_before", {})
        aft = e.get("state_after", {})
        def _delta(key):
            return aft.get(key, 0) - bef.get(key, 0)
        deltas = " ".join(
            f"{k[0].upper()}:{_delta(k):+.0f}"
            for k in ["energy", "hunger", "fatigue", "loneliness", "spirit"]
        )
        lines.append(f"  [{e.get('time', '?')}] {e.get('activity', '?')}  ({deltas})")
    lines.append("")
    return lines


def messages_section(n: int = 5) -> list[str]:
    msgs = _load_json(DATA / "phone_messages.json")
    if not msgs:
        return []

    lines = [f"## メッセージ (直近{n}件)"]
    for m in msgs[-n:]:
        role = m.get("role", "?")
        text = m.get("text", "")[:60]
        ts = m.get("timestamp", "")
        lines.append(f"  {role:>5} [{ts}] {text}")
    lines.append("")
    return lines


def day_log_section() -> list[str]:
    state = _load_json(DATA / "state.json")
    if not state:
        return []
    log = state.get("day_log", [])
    if not log:
        return []
    lines = ["## 今日のログ"]
    for l in log[-15:]:
        lines.append(f"  {l}")
    lines.append("")
    return lines


def bot_log_section(n: int = 10) -> list[str]:
    log_path = BASE / "bot.log"
    if not log_path.exists():
        return []
    try:
        log_lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    if not log_lines:
        return []
    lines = [f"## Botログ (直近{n}行)"]
    for l in log_lines[-n:]:
        lines.append(f"  {l}")
    lines.append("")
    return lines


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Monika 状態レポート — {now}")
    print("=" * 40)

    sections = [
        system_section,
        sim_section,
        day_log_section,
        history_section,
        messages_section,
        bot_log_section,
    ]
    seen = set()
    for fn in sections:
        for line in fn():
            if line.strip():
                seen.add(line.strip())
            print(line)

    # ── サービス制御 ──
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd in ("restart", "stop", "start"):
            subprocess.run(
                ["sudo", "systemctl", cmd, "monika.service"],
                check=False,
            )
            print(f"→ systemctl {cmd} monika.service 実行")
        elif cmd == "bot":
            subprocess.run(["pkill", "-f", "telegram_bot"], check=False)
            subprocess.run(
                ["nohup", "python3", "-u", "-m", "monica_core.telegram_bot"],
                cwd=str(BASE),
                env={**os.environ, "PYTHONPATH": str(SRC)},
                stdout=open(BASE / "bot.log", "a"),
                stderr=subprocess.STDOUT,
            )
            print("→ Telegram bot 再起動")
        else:
            print(f"不明なコマンド: {cmd}")
            print("  使い方: python3 -m monica_core.status [restart|stop|start|bot]")
        return

    print("─" * 40)
    print("管理コマンド:")
    print("  状態表示:   python3 -m monica_core.status")
    print("  再起動:     python3 -m monica_core.status restart")
    print("  停止:       python3 -m monica_core.status stop")
    print("  Bot再起動: python3 -m monica_core.status bot")


if __name__ == "__main__":
    main()
