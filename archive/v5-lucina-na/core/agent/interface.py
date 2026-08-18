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
    # v4.0: 意志フェーズ — 自分の部屋（ワークスペース）に自由に作品を作る
    "workspace_write": "自分の部屋 (data/workspace/) にノート・下書き・実験ファイルを作成する",
    "workspace_list": "自分の部屋 (data/workspace/) のファイル一覧を確認する",
}

# v4.1: ツールのパラメータ仕様（計画層がLLMに渡すためのスキーマ）
# 各ツールに必要なパラメータと例を示す。planning プロンプトに注入され、
# 空パラメータで実行されて0バイトのゴミファイルが作られる問題を防ぐ。
TOOL_PARAM_SCHEMAS: dict[str, dict] = {
    "file_read": {
        "required": ["path"],
        "example": {"path": "data/workspace/note.md"},
    },
    "file_write": {
        "required": ["path", "content"],
        "example": {"path": "data/workspace/report.md", "content": "# レポート\n内容..."},
    },
    "file_list": {
        "required": [],
        "example": {"path": "data/workspace"},
    },
    "command_exec": {
        "required": ["command"],
        "example": {"command": "python3 -m pytest tests/ -q"},
    },
    "web_search": {
        "required": ["query"],
        "example": {"query": "ゲーム理論 認知バイアス 統合モデル"},
    },
    "web_fetch": {
        "required": ["url"],
        "example": {"url": "https://example.com/article"},
    },
    "code_analyze": {
        "required": ["path"],
        "example": {"path": "core/agent/agent.py", "task": "このファイルの役割を分析して"},
    },
    "notify_user": {
        "required": ["message"],
        "example": {"message": "調査が完了しました"},
    },
    "opencode_run": {
        "required": ["task"],
        "example": {"task": "data/workspace/ にあるメモを要約して"},
    },
    "self_modify": {
        "required": ["task"],
        "example": {"task": "Agentに新しいツール 'file_count' を追加して", "target_file": "core/agent/agent.py"},
    },
    "backup": {
        "required": [],
        "example": {"suffix": "before_refactor"},
    },
    "direct_execute": {
        "required": ["instruction"],
        "example": {"instruction": "data/workspace/ に設計メモを作成し、中身を書いてください"},
    },
    # v4.0: 意志フェーズ
    "workspace_write": {
        "required": ["content"],
        "example": {
            "name": "simulation_note.md",
            "content": "ゲーム理論と認知バイアスを統合したシミュレーションの設計メモ...",
            "subdir": "experiments",
        },
    },
    "workspace_list": {
        "required": [],
        "example": {},
    },
}


class Agent:
    def execute(self, input: AgentInput) -> AgentOutput: ...
    def execute_step(self, step: "Step") -> StepResult: ...
    def call_tool(self, name: str, params: dict) -> Any: ...
    def speak(self, text: str) -> str: ...
