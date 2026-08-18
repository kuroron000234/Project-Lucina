"""
環境層 (Environment)

責務: PCやユーザーから現在の状態を取得する。システムの最上流。
"""

import logging
import os
import subprocess
import time as time_module
from datetime import datetime
from pathlib import Path

import psutil

from environment.interface import (
    ActionResult,
    EnvironmentInput,
    EnvironmentOutput,
    FileInfo,
    NetworkState,
    SystemState,
)

logger = logging.getLogger("Environment")


class Environment:
    """
    環境層: OSのシステムコールを介して現在の環境状態を取得し、
    エージェントからの行動依頼を実行する。
    """

    def __init__(self, workspace_dir: str | None = None):
        self.workspace_dir = workspace_dir or os.getcwd()

    def observe(self, input: EnvironmentInput) -> EnvironmentOutput:
        """
        現在の環境状態を取得する。このメソッドが全処理の起点。

        エッジケース:
        - ユーザー入力がない場合: user_input = None として定期観測
        - センサー取得失敗: 該当フィールドをデフォルト値に
        - 初回起動時: trigger = "startup" で特別な初期化シーケンス
        """
        return EnvironmentOutput(
            timestamp=datetime.now(),
            user_input=self._read_stdin(input),
            system_state=self._get_system_state(),
            files=self._list_workspace_files(),
            network=self._get_network_state(),
            sensors={},
        )

    def execute_action(self, action: str, params: dict) -> ActionResult:
        """
        エージェント層からの行動依頼を実際にOSレベルで実行する。
        """
        start = time_module.time()
        try:
            if action == "command_exec":
                cmd = params.get("command", "")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=params.get("timeout", 30),
                )
                return ActionResult(
                    success=result.returncode == 0,
                    output=result.stdout or result.stderr,
                    error=result.stderr if result.returncode != 0 else None,
                    duration=time_module.time() - start,
                )
            elif action == "file_read":
                path = params.get("path", "")
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                return ActionResult(
                    success=True,
                    output=content,
                    duration=time_module.time() - start,
                )
            elif action == "file_write":
                path = params.get("path", "")
                content = params.get("content", "")
                mode = params.get("mode", "w")
                with open(path, mode, encoding="utf-8") as f:
                    f.write(content)
                return ActionResult(
                    success=True,
                    output=f"Written to {path}",
                    duration=time_module.time() - start,
                )
            elif action == "file_list":
                path = params.get("path", self.workspace_dir)
                entries = os.listdir(path)
                return ActionResult(
                    success=True,
                    output="\n".join(entries),
                    duration=time_module.time() - start,
                )
            else:
                return ActionResult(
                    success=False,
                    output="",
                    error=f"Unknown action: {action}",
                    duration=time_module.time() - start,
                )
        except subprocess.TimeoutExpired as e:
            return ActionResult(
                success=False,
                output="",
                error=f"Command timed out after {e.timeout}s",
                duration=time_module.time() - start,
            )
        except Exception as e:
            logger.error(f"Action execution failed: {action} - {e}")
            return ActionResult(
                success=False,
                output="",
                error=str(e),
                duration=time_module.time() - start,
            )

    def _read_stdin(self, input: EnvironmentInput) -> str | None:
        """標準入力やトリガーからユーザー入力を取得する。"""
        return input.user_message

    def _get_system_state(self) -> SystemState:
        """
        psutil を使用してシステム状態を取得する。

        エッジケース:
        - センサー取得失敗時のフォールバック値を設定
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except Exception:
            cpu_percent = 0.0
            logger.warning("Failed to get CPU percent")

        try:
            memory_percent = psutil.virtual_memory().percent
        except Exception:
            memory_percent = 0.0
            logger.warning("Failed to get memory percent")

        try:
            uptime = time_module.time() - psutil.boot_time()
        except Exception:
            uptime = 0.0
            logger.warning("Failed to get uptime")

        return SystemState(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_window=self._get_active_window(),
            uptime=uptime,
            current_directory=os.getcwd(),
        )

    def _get_active_window(self) -> str | None:
        """
        現在アクティブなウィンドウのタイトルを取得する。
        """
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _list_workspace_files(self) -> list[FileInfo]:
        """
        ワークスペースディレクトリのファイル一覧を取得する。

        エッジケース:
        - ディレクトリが存在しない場合: 空リストを返す
        - アクセス権限がない場合: 該当ファイルをスキップ
        """
        files = []
        try:
            for entry in os.scandir(self.workspace_dir):
                try:
                    stat_info = entry.stat()
                    files.append(FileInfo(
                        path=entry.path,
                        name=entry.name,
                        size=stat_info.st_size,
                        modified=datetime.fromtimestamp(stat_info.st_mtime),
                        type="directory" if entry.is_dir() else "file",
                    ))
                except (PermissionError, OSError):
                    continue
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Cannot list workspace files: {e}")
            return []
        return files

    def _get_network_state(self) -> NetworkState | None:
        """
        ネットワーク状態を取得する。

        エッジケース:
        - ネットワーク情報が取得できない場合: None を返す
        """
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
            return NetworkState(
                is_connected=True,
                ip_address=ip_address,
                signal_strength=None,
            )
        except Exception:
            return NetworkState(
                is_connected=False,
                ip_address=None,
                signal_strength=None,
            )
