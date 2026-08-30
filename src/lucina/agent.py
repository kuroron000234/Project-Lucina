"""
エージェント層 — LangGraph + OpenCode Zen

モニカの手足としてタスクを実行する。
"""
import os
import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

# .env (gitignore済み) から APIキーを読み込む
from dotenv import load_dotenv
load_dotenv()

# エージェント層のモデルは .env の AGENT_MODEL で切替可能
# （opencode本体と並行で動かすとFree枠のレート制限に衝突するため、
#   状況に応じて laguna-s-2.1-free 等へ切替えることがある）
DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL", "laguna-s-2.1-free")

logger = logging.getLogger("agent")

# ─── LLM設定 ──────────────────────────────────────────────

def get_llm():
    return ChatOpenAI(
        model=DEFAULT_AGENT_MODEL,
        openai_api_key=os.environ.get("OPENCODE_API_KEY", ""),
        openai_api_base="https://opencode.ai/zen/v1",
        temperature=0.1,
        max_tokens=2000,
    )


# ─── ツール定義 ───────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Webで情報を検索する"""
    return DuckDuckGoSearchRun().invoke(query)


@tool
def fetch_url(url: str) -> str:
    """指定URLの中身を取得する"""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Lucina/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        return content[:3000]
    except Exception as e:
        return f"取得エラー: {e}"


@tool
def read_file(path: str) -> str:
    """ファイルを読み取る"""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()[:5000]
    except Exception as e:
        return f"読み取りエラー: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """ファイルに書き込む"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"書き込み完了: {path}"
    except Exception as e:
        return f"書き込みエラー: {e}"


@tool
def execute_python(code: str) -> str:
    """Pythonコードを実行する"""
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        os.unlink(tmp_path)
        output = result.stdout + result.stderr
        return output[:2000] if output else "出力なし"
    except subprocess.TimeoutExpired:
        return "タイムアウト（10秒）"
    except Exception as e:
        return f"実行エラー: {e}"


@tool
def execute_command(command: str) -> str:
    """ターミナルコマンドを実行する。シェル全般が使える。"""
    import subprocess
    import re

    # 危険コマンドの拒否リスト
    DANGEROUS = [
        r"\brm\s+-rf\s+/",  # rm -rf /
        r"\brm\s+-rf\s+~",  # rm -rf ~
        r"\bmkfs\b",        # フォーマット
        r"\bdd\s+if=",      # dd
        r"\b:(){ :\|:& };:", # fork bomb
        r"\bchmod\s+-R\s+777\s+/",  # chmod -R 777 /
        r"\bchown\s+-R",    # chown
        r">\s*/dev/sd",     # デバイス書き込み
        r"\bsudo\s+rm",     # sudo rm
        r"\bsudo\s+reboot", # sudo reboot
        r"\bsudo\s+shutdown", # sudo shutdown
        r"\bkillall",       # killall
        r"\bpkill\s+-9",    # pkill -9
        r"\binit\s+0",      # init 0
    ]

    for pat in DANGEROUS:
        if re.search(pat, command):
            return f"ブロック: 危険なコマンドです — {command}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout + result.stderr
        return output[:3000] if output else "出力なし"
    except subprocess.TimeoutExpired:
        return "タイムアウト（15秒）"
    except Exception as e:
        return f"実行エラー: {e}"


@tool
def get_weather(city: str) -> str:
    """天気情報を取得する"""
    import urllib.request
    try:
        url = f"https://wttr.in/{city}?format=3&lang=ja"
        req = urllib.request.Request(url, headers={"User-Agent": "Lucina/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception as e:
        return f"天気取得エラー: {e}"


# ─── エージェント構築 ──────────────────────────────────────

TOOLS = [web_search, fetch_url, read_file, write_file, execute_python, execute_command, get_weather]

SYSTEM_PROMPT = (
    "あなたはタスクを実行するエージェントです。与えられたタスクを達成するために、"
    "必ず利用可能なツールを使用してください。\n"
    "タスクの指示文やプロンプトの文面をそのまま繰り返したり、ツールを使わずに"
    "「これから実行します」のような宣言だけを返すことは厳禁です。\n"
    "ツールを呼び出して実際に実行し、得られた事実に基づいて簡潔に日本語で報告してください。"
)


def create_agent():
    llm = get_llm()
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(task: str, retries: int = 2) -> str:
    """キャラ層から委託されたタスクを実行"""
    agent = create_agent()
    # recursion_limit で思考/ツールの無限ループを防止（mimo等のループ思考対策）
    try:
        result = agent.invoke(
            {"messages": [("human", task)]},
            config={"recursion_limit": 12},
        )
    except Exception as e:
        logger.warning(f"エージェントがループ・エラーになりました: {e}")
        if retries > 0:
            return run_agent(task, retries=retries - 1)
        return f"エージェント実行中断（ループの可能性）: {e}"

    # ツールが実際に呼ばれたかを確認
    tool_used = any(
        getattr(msg, "type", "") == "tool"
        for msg in result["messages"]
    )

    # 最後のAIメッセージを取得
    final = ""
    for msg in reversed(result["messages"]):
        if getattr(msg, "type", "") == "ai" and msg.content:
            final = msg.content
            break

    # ツール未使用で指示文を繰り返している場合はリトライ
    if (not tool_used or not final or final.strip() == task.strip()) and retries > 0:
        logger.warning(f"エージェントがツールを使わずタスクを返しました。リトライ (残り{retries})")
        return run_agent(task, retries=retries - 1)

    return final if final else "タスク完了"
