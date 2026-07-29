from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    step_order: int
    action: str
    success: bool
    output: str
    error: str | None = None
    duration: float = 0.0
    side_effects: dict | None = None


@dataclass
class AgentInput:
    plan: "PlanningOutput"
    context: dict | None = None


@dataclass
class AgentOutput:
    plan_id: str
    step_results: list[StepResult]
    overall_success: bool
    execution_time: float
    log: str = ""


TOOL_REGISTRY: dict[str, str] = {
    "file_read": "ファイルの内容を読み込む",
    "file_write": "ファイルに書き込む",
    "file_list": "ディレクトリの内容を一覧する",
    "command_exec": "シェルコマンドを実行する",
    "web_search": "Web検索を実行する（Opencode委託）",
    "web_fetch": "URLの内容を取得する（Opencode委託）",
    "code_analyze": "コードを解析する（Opencode委託）",
    "notify_user": "ユーザーに通知する",
    "opencode_run": "任意のタスクをOpencodeに委託して実行する",
    "self_modify": "自身のソースコードをOpencodeに編集させる（新機能追加・改変）",
    "backup": "プロジェクト全体をバックアップする",
    "direct_execute": "自然言語の指示をそのままOpencodeに渡して実行する",
}


class Agent:
    def execute(self, input: AgentInput) -> AgentOutput: ...
    def execute_step(self, step: "Step") -> StepResult: ...
    def call_tool(self, name: str, params: dict) -> Any: ...
    def speak(self, text: str) -> str: ...
