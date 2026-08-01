"""
OpencodeBridge

Opencode CLI を呼び出して、タスクを委託実行する。
web検索・URL取得・コード解析・自己改変などの複雑な処理を
Opencodeエージェントに任せることで高精度な実行を実現する。

重要: `opencode run` に `--format json` を渡さないこと。
opencode v1.18.7 では JSON形式出力時にリモートレジストリ(models.dev)
との同期処理でハングする。代わりにデフォルト(プレーンテキスト)形式を使用。
90秒のタイムアウトで保護しているが、JSON形式のハングはこれでも回避できない。

なお `opencode session list --format json` はハングせず即座に応答するため、
セッションIDの発見（v4.1.2）には安全に使用している。
"""

import ast
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import config
from core.agent.rate_limiter import RateLimiter

logger = logging.getLogger("OpencodeBridge")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # lucina-NA/

LUCINA_SESSION_ID_FILE = PROJECT_ROOT / "data" / ".lucina_session"

# Cloud model used by default
CLOUD_MODEL = "opencode/deepseek-v4-flash-free"

# Quick health-check: verify opencode CLI exists and is responsive
_HEALTH_CACHE: dict = {"ok": None, "checked_at": 0.0}
_HEALTH_TTL = 60.0  # re-check every 60 seconds


class OpencodeBridge:
    """
    Bridge to the Opencode CLI.

    Delegates complex tasks (web search, code analysis, self-modification)
    to the Opencode agent. Falls back gracefully when opencode is
    unavailable or hangs.
    """

    def __init__(self, model: str | None = None, timeout: int = 90,
                 max_requests: int = 10, window_seconds: int = 60):
        # Default to Freebuff cloud model
        self.model = model or CLOUD_MODEL
        self.timeout = timeout  # 90s for complex tasks
        self.rate_limiter = RateLimiter(max_requests, window_seconds)
        self._session_id: str | None = None
        self._load_session()

    # ── Session management ──

    def _load_session(self):
        if LUCINA_SESSION_ID_FILE.exists():
            sid = LUCINA_SESSION_ID_FILE.read_text().strip()
            if sid:
                self._session_id = sid
                logger.info(f"Loaded lucina session: {self._session_id}")

    def _save_session(self, sid: str):
        self._session_id = sid
        LUCINA_SESSION_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        LUCINA_SESSION_ID_FILE.write_text(sid)
        logger.info(f"Saved lucina session: {sid}")

    def _build_cmd(self, task: str) -> list[str]:
        """
        Build the opencode subprocess command.

        NOTE: Does NOT use --format json for `run` because opencode v1.18.7
        hangs in JSON format mode (tries to sync models.dev registry).
        Uses default plain text format instead, which is reliable.
        """
        cmd = ["opencode", "run", task]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.extend(["--title", "lucina-daemon"])  # no --format json (causes hang)
        if self._session_id:
            cmd.extend(["--session", self._session_id])
        return cmd

    def _discover_session_id(self) -> str | None:
        """
        v4.1.2: 最新の lucina-daemon セッションIDを `opencode session list` から取得する。

        従来は run() の stderr からセッションIDを抽出しようとしていたが、
        --format json を使わないため stderr にセッションIDは含まれず、
        毎回新規セッションが作成されて肥大化していた（74個の lucina-daemon
        セッション・opencode.db 1.6GB に膨張）。

        `opencode session list --format json` は即座に結果を返し、
        タイトルとディレクトリで絞り込んで最新セッションを特定できる。
        """
        try:
            result = subprocess.run(
                ["opencode", "session", "list", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return None
            sessions = json.loads(result.stdout)
        except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"Session discovery failed: {e}")
            return None

        target_dir = str(PROJECT_ROOT)
        candidates = [
            s for s in sessions
            if isinstance(s, dict)
            and s.get("title") == "lucina-daemon"
            and s.get("directory") == target_dir
            and isinstance(s.get("id"), str)
            and s["id"].startswith("ses_")
        ]
        if not candidates:
            return None
        # updated が最新のものを選択
        latest = max(candidates, key=lambda s: s.get("updated", 0))
        return latest["id"]

    # ── Health check ──

    @classmethod
    def health_check(cls, force: bool = False) -> bool:
        """
        Quick check whether opencode CLI binary exists and is executable.
        Returns True if available, False otherwise.
        Results are cached for _HEALTH_TTL seconds.

        Note: This only verifies `opencode --version` works.
        The 90s timeout in run() protects against other hang scenarios.
        """
        now = time.time()
        if not force and _HEALTH_CACHE["ok"] is not None and (now - _HEALTH_CACHE["checked_at"]) < _HEALTH_TTL:
            return _HEALTH_CACHE["ok"]

        try:
            result = subprocess.run(
                ["opencode", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            ok = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            ok = False

        _HEALTH_CACHE["ok"] = ok
        _HEALTH_CACHE["checked_at"] = now
        if not ok:
            logger.warning("Opencode CLI health check FAILED")
        return ok

    # ── Core: run a task ──

    def run(self, task: str) -> dict:
        """
        Delegate an arbitrary task to Opencode.

        Uses default (plain text) output format, not JSON format,
        because opencode v1.18.7 hangs with --format json.

        Returns:
            {"success": bool, "output": str, "error": str | None, "duration": float}

        On timeout (90s default) returns success=False with a descriptive error.
        """
        if not self.health_check():
            return {
                "success": False,
                "output": "",
                "error": "opencode CLI is not available",
                "duration": 0.0,
            }

        self.rate_limiter.acquire()
        start = time.time()
        cmd = self._build_cmd(task)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration = time.time() - start

            # Stdout = model response text (plain text, not JSON)
            output = result.stdout.strip()

            # Stderr = progress bars, logs (no session ID in plain format)
            stderr = result.stderr.strip()

            # v4.1.2: セッション付き実行が失敗した場合（保存済みセッションが
            # 削除済み等で無効化されている可能性があるため）、セッションを
            # リセットして1回だけ新規セッションでリトライする。
            # 失敗の原因がセッション以外（モデルエラー等）でもコスト2倍に
            # なるが、単発のリトライなので許容する。
            if result.returncode != 0 and self._session_id:
                logger.warning(
                    f"Opencode run failed with saved session {self._session_id}; "
                    f"retrying with a fresh session"
                )
                self._session_id = None
                LUCINA_SESSION_ID_FILE.unlink(missing_ok=True)
                cmd = self._build_cmd(task)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                duration = time.time() - start
                output = result.stdout.strip()
                stderr = result.stderr.strip()

            if result.returncode != 0:
                return {
                    "success": False,
                    "output": output,
                    "error": stderr or "Opencode returned non-zero exit code",
                    "duration": duration,
                }

            # v4.1.2: セッションIDの永続化（stderr からの抽出は廃止）。
            # 実行後に対応する lucina-daemon セッションを session list から
            # 発見して保存し、次回以降は --session で再利用する。
            if not self._session_id:
                sid = self._discover_session_id()
                if sid:
                    self._save_session(sid)

            return {
                "success": True,
                "output": output or "(task completed)",
                "error": None,
                "duration": duration,
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Opencode task timed out after {self.timeout}s: {task[:60]}")
            return {
                "success": False,
                "output": "",
                "error": f"Opencode task timed out after {self.timeout}s (model={self.model})",
                "duration": self.timeout,
            }
        except FileNotFoundError:
            _HEALTH_CACHE["ok"] = False
            return {
                "success": False,
                "output": "",
                "error": "opencode CLI not found. Install opencode first.",
                "duration": time.time() - start,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "duration": time.time() - start,
            }

    # ── Convenience methods ──

    def search(self, query: str) -> dict:
        """Web検索をOpencodeに委託する。"""
        return self.run(f"web search: {query}")

    def fetch(self, url: str) -> dict:
        """URL取得をOpencodeに委託する。"""
        return self.run(f"fetch the content of this URL and summarize it: {url}")

    def analyze(self, path: str, task: str = "analyze this code") -> dict:
        """コード解析をOpencodeに委託する。"""
        return self.run(f"{task}: {path}")

    def modify(self, task: str, target_file: str | None = None) -> dict:
        """
        自身のソースコードをOpencodeに編集させる。

        task: 変更内容の説明（例：「Agentに新しいツール 'file_count' を追加して」）
        target_file: 編集対象ファイルの相対パス（例："core/agent/agent.py"）
        """
        project_context = self._build_project_context(target_file)
        full_task = f"""{task}

{project_context}

上記のタスクを実行してください。
- コードの変更後は構文チェック（`python3 -c "import ast; ast.parse(open(path).read())"`）を実行
- テストが存在する場合はテストも実行
- 変更内容を簡潔に報告してください"""
        return self.run(full_task)

    def backup(self, suffix: str | None = None) -> dict:
        """
        プロジェクト全体をタイムスタンプ付きでバックアップする。
        """
        timestamp = suffix or time.strftime("%Y%m%d_%H%M%S")
        backup_dir = PROJECT_ROOT.parent / f"lucina-NA_backup_{timestamp}"
        try:
            shutil.copytree(PROJECT_ROOT, backup_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            return {
                "success": True,
                "output": f"Backup created: {backup_dir}",
                "error": None,
                "duration": 0.0,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "duration": 0.0,
            }

    def validate_syntax(self, filepath: str) -> dict:
        """指定ファイルのPython構文をチェックする。"""
        full_path = PROJECT_ROOT / filepath if not os.path.isabs(filepath) else filepath
        try:
            with open(full_path) as f:
                ast.parse(f.read())
            return {"success": True, "output": f"Syntax OK: {filepath}", "error": None, "duration": 0.0}
        except SyntaxError as e:
            return {"success": False, "output": "", "error": f"Syntax error in {filepath}: {e}", "duration": 0.0}
        except FileNotFoundError:
            return {"success": False, "output": "", "error": f"File not found: {filepath}", "duration": 0.0}

    def run_tests(self, test_path: str = "tests/") -> dict:
        """テストを実行して結果を返す。"""
        return self.run(f"run tests in {PROJECT_ROOT / test_path} and report results")

    # ── Internal helpers ──

    def _build_project_context(self, target_file: str | None = None) -> str:
        """Opencodeに渡すプロジェクト構造のコンテキストを生成する。"""
        structure = []
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            if "__pycache__" in str(path) or ".pytest_cache" in str(path):
                continue
            rel = path.relative_to(PROJECT_ROOT)
            size = path.stat().st_size
            structure.append(f"  {rel} ({size} bytes)")

        files_str = "\n".join(structure)
        target_hint = f"\n特に編集が必要なファイル: {target_file}" if target_file else ""

        return f"""
これは lucina-NA エージェントシステムのプロジェクトです。
プロジェクトルート: {PROJECT_ROOT}

ディレクトリ構造:
{files_str}{target_hint}

注意:
- 各層は interface.py で入出力を定義し、同名の .py ファイルで実装する
- dataclass は interface.py に定義する
- 新しくツールを追加する場合は core/agent/agent.py の tools 辞書に追加し、
  core/agent/interface.py の TOOL_REGISTRY も更新する
"""
