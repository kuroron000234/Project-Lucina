"""ExecutorAdapter — Lucina の質問・依頼を実行エージェントへルーティングする（v1.13）。

Lucina は「外部への働きかけ」として自然言語の質問文を発する（v1.10 の好奇心駆動の問いかけ）。
本クラスはその文を受け、定型依頼（時刻・URL取得）は**自前サンドボックス**、複雑な依頼
（調査・コード・設計等）は**Opencode CLI** へ振り分け、結果を返す。結果は run_agent が
InterruptChannel.inject() で Lucina に返す（好奇心 relief・応答待ち解除 → 対話ループが閉じる）。

設計の根拠（前回の教訓 v1.9・制御トークンCが実測0回）:
モデルに未学習のトークン/関数名を出力させる方式は信頼できないため、Lucina は普通の
日本語の質問文を出すだけにし、解釈は外側のルール（本クラス）が担う。

安全方針:
- サンドボックスは「読み取り専用・ステートレス」の定型操作のみ（date / URL取得）。
  任意コマンド実行・ファイル書き込みは許可しない（Opencode 側のサンドボックスに委ねる）。
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import urllib.request
from typing import Any

# 定型依頼 → サンドボックス（即時・無料・読み取り専用）
_DATE_KEYWORDS = ("時刻", "時間", "日付", "今何時", "何時ですか")
# 複雑な依頼 → Opencode CLI（汎用。コード・調査・Web・ファイル）
_OPENCODE_KEYWORDS = (
    "調べて", "調査", "分析", "コード", "書いて", "設計",
    "読んで", "ファイル", "実行して", "実行し", "調べる",
)

_URL_RE = re.compile(r"https?://[^\s）)]+")


class ExecutorAdapter:
    """質問文 → 実行バックエンドの分類・実行。結果は inject する側（run_agent）が扱う。"""

    def __init__(self, config: dict[str, Any]) -> None:
        c = config.get("executor", {})
        self.enabled = bool(c.get("enabled", False))
        self.opencode_cmd = str(c.get("opencode_command", "opencode"))
        # Opencode の既定モデルが未取得/未設定だと失敗するため、明示指定できるようにする
        # （例: "opencode/deepseek-v4-flash-free"。空なら opencode の既定モデルに任せる）
        self.opencode_model = str(c.get("opencode_model", "")).strip()
        self.opencode_timeout_sec = float(c.get("opencode_timeout_sec", 120))
        self.sandbox_timeout_sec = float(c.get("sandbox_timeout_sec", 15))
        self.url_max_chars = int(c.get("url_max_chars", 600))
        # セッション乱立対策（v1.13）: 同一 Lucina プロセス内では 1 つの Opencode セッションを
        # 使い回す（--format json で sessionID を取得 → 次回から --session で継続）。
        # これにより `opencode run` のたびに新セッションが乱立するのを防ぎ、
        # 実行エージェントに調査の継続記憶を持たせる。close() で削除して掃除する。
        self.reuse_session = bool(c.get("opencode_reuse_session", True))
        self._opencode_session: str | None = None

    # ------------------------------------------------------------------ #
    # ルーティング
    # ------------------------------------------------------------------ #
    def route(self, text: str) -> tuple[str, str] | None:
        """質問文を (バックエンド, ペイロード) に分類する。実行不能なら None（人間に委ねる）。"""
        t = (text or "").strip()
        if not t:
            return None
        if any(k in t for k in _DATE_KEYWORDS):
            return ("sandbox", "date")
        m = _URL_RE.search(t)
        if m:
            return ("sandbox", m.group(0))
        if any(k in t for k in _OPENCODE_KEYWORDS):
            return ("opencode", t)
        return None

    # ------------------------------------------------------------------ #
    # 実行
    # ------------------------------------------------------------------ #
    async def run(self, backend: str, payload: str) -> str | None:
        """ルーティング結果を実行し、結果テキストを返す。実行不能・無効時は None。"""
        if not self.enabled:
            return None
        if backend == "sandbox":
            return await asyncio.to_thread(self._run_sandbox, payload)
        if backend == "opencode":
            return await asyncio.to_thread(self._run_opencode, payload)
        return None

    def _run_sandbox(self, payload: str) -> str | None:
        """読み取り専用の定型操作。失敗は例外でなく結果文字列（（実行失敗）…）で返す。"""
        try:
            if payload == "date":
                res = subprocess.run(
                    ["date"], capture_output=True, text=True, timeout=self.sandbox_timeout_sec
                )
                return (res.stdout or res.stderr or "").strip() or None
            if payload.startswith(("http://", "https://")):
                with urllib.request.urlopen(payload, timeout=self.sandbox_timeout_sec) as resp:  # noqa: S310
                    data = resp.read(self.url_max_chars + 1)
                return data.decode("utf-8", errors="replace")[: self.url_max_chars]
        except Exception as exc:  # noqa: BLE001 - 実行失敗は結果として返す（Lucinaに届ける）
            return f"（実行失敗）{exc}"
        return None

    def _run_opencode(self, prompt: str) -> str | None:
        """Opencode CLI の非対話実行（`opencode run --format json …`）。未インストールなら None。

        - `--format json` で出力を構造化イベントとして受け取り、回答テキスト（part.text）と
          sessionID を抽出する。
        - セッション乱立対策: 取得した sessionID を保持し、次回から `--session` で同じ
          セッションを継続する（1プロセス=1セッション）。`--no-replay` で過去履歴の再表示を抑制。
        """
        if shutil.which(self.opencode_cmd) is None:
            return None
        try:
            cmd = [self.opencode_cmd, "run", "--format", "json", "--no-replay"]
            if self.opencode_model:
                cmd += ["--model", self.opencode_model]
            if self.reuse_session and self._opencode_session:
                cmd += ["--session", self._opencode_session]
            cmd.append(prompt)
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.opencode_timeout_sec,
            )
            # JSONイベントから回答テキストとセッションIDを抽出
            texts: list[str] = []
            session_id: str | None = None
            for line in (res.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "text":
                    part = d.get("part") or {}
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(part["text"])
                    if session_id is None:
                        session_id = d.get("sessionID")
            if self.reuse_session and session_id:
                self._opencode_session = session_id
            out = "".join(texts).strip()
            if out:
                return out[:2000]
            err = (res.stderr or "").strip()
            return err[:500] or "（出力なし）"
        except subprocess.TimeoutExpired:
            return f"（タイムアウト: {self.opencode_timeout_sec:.0f}秒）"
        except Exception as exc:  # noqa: BLE001
            return f"（実行失敗）{exc}"

    def close(self) -> None:
        """作成した Opencode セッションを削除する（セッション乱立の掃除）。

        run_agent の終了時に呼ばれる。削除に失敗しても握りつぶす（掃除はベストエフォート）。
        """
        if not self._opencode_session or shutil.which(self.opencode_cmd) is None:
            return
        try:
            subprocess.run(
                [self.opencode_cmd, "session", "delete", self._opencode_session],
                capture_output=True,
                timeout=10,
            )
        except Exception:  # noqa: BLE001 - 掃除失敗は無視（次回起動で新セッションになるだけ）
            pass
        self._opencode_session = None
