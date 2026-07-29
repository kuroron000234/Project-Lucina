"""
エージェント層 (Agent)

責務: 行動計画の各ステップを実際に実行する。システムの最下流。
"""

import logging
import os
import subprocess
import time
from typing import Any

from core.agent.interface import (
    AgentInput,
    AgentOutput,
    TOOL_REGISTRY,
    StepResult,
)
from core.agent.opencode_bridge import OpencodeBridge
from core.planning.interface import Step

logger = logging.getLogger("Agent")


class Agent:
    """
    エージェント層: 行動計画の各ステップをツールを使って実行する。

    エッジケース:
    - 権限不足: StepResult にエラーとして記録、ツール無効としてマーク
    - 予期しない副作用: すべての副作用を記録
    - 部分成功: 各StepResultの成功/失敗を個別に記録
    """

    def __init__(self, opencode_bridge: OpencodeBridge | None = None):
        self.opencode = opencode_bridge or OpencodeBridge()
        self.tools = {
            "file_read": self._tool_file_read,
            "file_write": self._tool_file_write,
            "file_list": self._tool_file_list,
            "command_exec": self._tool_command_exec,
            "web_search": self._tool_web_search,
            "web_fetch": self._tool_web_fetch,
            "code_analyze": self._tool_code_analyze,
            "notify_user": self._tool_notify_user,
            "opencode_run": self._tool_opencode_run,
            "self_modify": self._tool_self_modify,
            "backup": self._tool_backup,
            "direct_execute": self._tool_direct_execute,
        }
        self.TOOL_REGISTRY = TOOL_REGISTRY

    @property
    def rate_limit_state(self) -> dict:
        return self.opencode.rate_limiter.state

    def execute(self, input: AgentInput) -> AgentOutput:
        """
        計画を実行する。ステップごとに実行・結果収集。

        各ステップを順次実行し、タイムアウト処理を行う。
        """
        start = time.time()
        step_results = []
        log_entries = []

        for step in input.plan.steps:
            try:
                result = self._execute_step_with_timeout(step, step.timeout)
            except TimeoutError:
                result = StepResult(
                    step_order=step.order,
                    action=step.action,
                    success=False,
                    output="",
                    error=f"Timeout after {step.timeout}s",
                    duration=step.timeout,
                    side_effects=None,
                )
                logger.warning(f"Step {step.order} timed out ({step.action})")
            except Exception as e:
                result = StepResult(
                    step_order=step.order,
                    action=step.action,
                    success=False,
                    output="",
                    error=str(e),
                    duration=time.time() - start,
                    side_effects=None,
                )
                logger.error(f"Step {step.order} failed: {e}")

            step_results.append(result)
            log_entries.append(
                f"[Step {result.step_order}] {result.action}: "
                f"{'✓' if result.success else '✗'} ({result.duration:.2f}s)"
                f"{' - ' + result.error[:50] if result.error else ''}"
            )

        execution_time = time.time() - start
        overall_success = all(r.success for r in step_results)

        return AgentOutput(
            plan_id=input.plan.plan_id,
            step_results=step_results,
            overall_success=overall_success,
            execution_time=execution_time,
            log="\n".join(log_entries),
        )

    def execute_step(self, step: Step) -> StepResult:
        """
        1ステップを実行する。
        """
        return self._execute_step_with_timeout(step, step.timeout)

    def call_tool(self, name: str, params: dict) -> Any:
        """
        ツールを名前で呼び出す。ツールレジストリから検索して実行。
        """
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self.tools.keys())}")
        return self.tools[name](params)

    def speak(self, text: str) -> str:
        """
        ユーザーに向けて発話する（標準出力に表示）。
        """
        print(f"\n[Lucina] {text}\n")
        return text

    def _execute_step_with_timeout(self, step: Step, timeout: float) -> StepResult:
        """
        タイムアウト付きで1ステップを実行する。
        """
        start = time.time()

        if step.action not in self.tools:
            return StepResult(
                step_order=step.order,
                action=step.action,
                success=False,
                output="",
                error=f"Unknown tool: {step.action}",
                duration=time.time() - start,
                side_effects=None,
            )

        try:
            result = self.tools[step.action](step.params)
            duration = time.time() - start

            success = True
            output = ""
            error = None
            side_effects = None

            if isinstance(result, dict):
                success = result.get("success", True)
                output = result.get("output", str(result))
                error = result.get("error")
                side_effects = result.get("side_effects")
            else:
                output = str(result)

            return StepResult(
                step_order=step.order,
                action=step.action,
                success=success,
                output=output,
                error=error,
                duration=duration,
                side_effects=side_effects,
            )
        except Exception as e:
            return StepResult(
                step_order=step.order,
                action=step.action,
                success=False,
                output="",
                error=str(e),
                duration=time.time() - start,
                side_effects=None,
            )

    # --- ツール実装 ---

    def _tool_file_read(self, params: dict) -> dict:
        """ファイルの内容を読み込む。"""
        path = params.get("path", "")
        encoding = params.get("encoding", "utf-8")
        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()
            return {"success": True, "output": content}
        except FileNotFoundError:
            return {"success": False, "output": "", "error": f"File not found: {path}"}
        except PermissionError:
            return {"success": False, "output": "", "error": f"Permission denied: {path}"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _tool_file_write(self, params: dict) -> dict:
        """ファイルに書き込む。"""
        path = params.get("path", "")
        content = params.get("content", "")
        mode = params.get("mode", "w")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "output": f"Written {len(content)} bytes to {path}"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _tool_file_list(self, params: dict) -> dict:
        """ディレクトリの内容を一覧する。"""
        path = params.get("path", os.getcwd())
        try:
            entries = sorted(os.listdir(path))
            output = "\n".join(entries)
            return {"success": True, "output": output}
        except FileNotFoundError:
            return {"success": False, "output": "", "error": f"Directory not found: {path}"}
        except PermissionError:
            return {"success": False, "output": "", "error": f"Permission denied: {path}"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _tool_command_exec(self, params: dict) -> dict:
        """シェルコマンドを実行する。"""
        command = params.get("command", "")
        timeout = params.get("timeout", 30)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            return {
                "success": result.returncode == 0,
                "output": output,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _tool_web_search(self, params: dict) -> dict:
        """Web検索をOpencodeに委託する。"""
        query = params.get("query", "")
        logger.info(f"Web search (opencode): {query[:80]}")
        return self.opencode.search(query)

    def _tool_web_fetch(self, params: dict) -> dict:
        """URL取得をOpencodeに委託する。"""
        url = params.get("url", "")
        logger.info(f"Web fetch (opencode): {url}")
        return self.opencode.fetch(url)

    def _tool_code_analyze(self, params: dict) -> dict:
        """コード解析をOpencodeに委託する。"""
        path = params.get("path", "")
        task = params.get("task", "analyze this code")
        logger.info(f"Code analysis (opencode): {path}")
        return self.opencode.analyze(path, task)

    def _tool_opencode_run(self, params: dict) -> dict:
        """任意のタスクをOpencodeに委託する。"""
        task = params.get("task", "")
        logger.info(f"Opencode run: {task[:100]}")
        return self.opencode.run(task)

    def _tool_notify_user(self, params: dict) -> dict:
        """ユーザーに通知する。"""
        message = params.get("message", "")
        level = params.get("level", "info")
        print(f"\n[{level.upper()}] {message}")
        return {"success": True, "output": f"Notified user: {message[:100]}"}

    def _tool_self_modify(self, params: dict) -> dict:
        """
        自身のソースコードをOpencodeに編集させる。

        プロジェクト構造のコンテキスト付きでOpencodeにタスクを委託し、
        変更後は構文チェックを自動実行する。

        パラメータ:
            task: 変更内容の説明（必須）
            target_file: 編集対象の相対パス（任意）
            run_tests: 変更後にテストを実行するか（省略可、デフォルトtrue）
        """
        task = params.get("task", "")
        target_file = params.get("target_file")
        should_test = params.get("run_tests", True)

        if not task:
            return {
                "success": False, "output": "", "error": "task is required",
                "duration": 0.0,
            }

        logger.info(f"Self-modify: {task[:100]}")

        result = self.opencode.modify(task, target_file)

        if result["success"] and should_test:
            validation = self.opencode.validate_syntax(target_file) if target_file else None
            if validation and not validation["success"]:
                return {
                    "success": False,
                    "output": result["output"],
                    "error": f"Syntax validation failed after modification: {validation['error']}",
                    "duration": result["duration"],
                }

        return result

    def _tool_backup(self, params: dict) -> dict:
        """
        プロジェクト全体をバックアップする。

        パラメータ:
            suffix: バックアップディレクトリ名のサフィックス（省略可）
        """
        suffix = params.get("suffix")
        return self.opencode.backup(suffix)

    def _tool_direct_execute(self, params: dict) -> dict:
        """
        自然言語の指示をそのままOpencodeに渡して実行する。
        Planning層での分解をスキップし、人格層の意図を直接実行する。

        パラメータ:
            instruction: 実行したい内容の自然言語指示（必須）
            context: 追加コンテキスト（省略可）
        """
        instruction = params.get("instruction", "")
        context = params.get("context", "")

        if not instruction:
            return {
                "success": False, "output": "", "error": "instruction is required",
                "duration": 0.0,
            }

        logger.info(f"Direct execute: {instruction[:100]}")
        task = f"実行指示: {instruction}"
        if context:
            task += f"\nコンテキスト: {context}"
        return self.opencode.run(task)
