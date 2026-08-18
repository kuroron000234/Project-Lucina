"""
LLMクライアント

OpenAI互換APIを呼び出す抽象化レイヤー。
ローカルLLM、OpenAI、モック（APIキーなし時）に対応。
"""

import json
import logging
import os
import re

logger = logging.getLogger("LLM")


class LLMClient:
    """
    LLMクライアント: OpenAI互換APIを呼び出す。

    環境変数 OPENAI_API_KEY が設定されている場合は実際のAPIを呼び出す。
    設定されていない場合はモックモードで動作（開発・テスト用）。
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        num_ctx: int | None = None,
    ):
        import config as cfg
        self.model = model or cfg.LLM_CONFIG.get("model", "gpt-4o-mini")
        self.temperature = temperature or cfg.LLM_CONFIG.get("temperature", 0.7)
        self.max_tokens = max_tokens or cfg.LLM_CONFIG.get("max_tokens", 1024)
        self.num_ctx = num_ctx or cfg.LLM_CONFIG.get("num_ctx")
        self.base_url = base_url or cfg.LLM_CONFIG.get("base_url")
        self.api_key = api_key or cfg.LLM_CONFIG.get("api_key") or os.environ.get("OPENAI_API_KEY") or ""
        if self.base_url and not self.api_key:
            self.api_key = "ollama"
        self.use_mock = not self.api_key and not self.base_url

        if self.use_mock:
            logger.warning(
                "No API key and no base_url set. Using mock LLM responses. "
                "Set OPENAI_API_KEY or configure base_url in config.py for real LLM calls."
            )
        else:
            ctx_info = f", num_ctx={self.num_ctx}" if self.num_ctx else ""
            logger.info(f"LLM configured: model={self.model}, base_url={self.base_url}{ctx_info}")

    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        LLMにプロンプトを送信し、応答を取得する。

        Args:
            prompt: ユーザープロンプト
            system_prompt: オプションのシステムプロンプト

        Returns:
            LLMの応答テキスト
        """
        if self.use_mock:
            return self._mock_response(prompt)

        return self._real_chat(prompt, system_prompt)

    def _real_chat(self, prompt: str, system_prompt: str | None = None) -> str:
        """実際のAPI呼び出し。"""
        try:
            import openai

            client_kwargs = {
                "api_key": self.api_key,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            client = openai.OpenAI(**client_kwargs)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            extra = {}
            if self.num_ctx:
                extra["num_ctx"] = self.num_ctx

            # v3.5.2: プロンプト長に応じて max_tokens を動的クランプする。
            # num_ctx 窓内に収め、プロンプト肥大化時でも窓端での切断を防ぐ。
            max_tokens = self._effective_max_tokens(prompt, system_prompt)

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                extra_body=extra,
            )

            # v3.5.2: finish_reason=length（max_tokens 到達で強制切断）を検出して
            # ログに残す。チャットの長文応答が途中で切れる問題の監視用。
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                logger.warning(
                    f"LLM response truncated (finish_reason=length, "
                    f"max_tokens={max_tokens}). Reply may be cut mid-sentence."
                )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            # APIエラー時はモックにフォールバック
            return self._mock_response(prompt)

    def _effective_max_tokens(self, prompt: str, system_prompt: str | None = None) -> int:
        """
        実際に使用する max_tokens を決定する（v3.5.2）。

        num_ctx が設定されている場合、プロンプトの概算トークン数
        （文字数 / 2 を安全側の見積もりとして使用）を差し引き、
        生成上限がコンテキスト窓を超えないようにクランプする。
        プロンプトが長いほど残り予算が少なくなるため、窓端での
        強制切断（finish_reason=length）を防ぐ。

        num_ctx 未設定時は設定値（self.max_tokens）をそのまま返す。
        """
        if not self.num_ctx:
            return self.max_tokens
        prompt_len = len(prompt) + len(system_prompt or "")
        est_prompt_tokens = prompt_len // 2  # 日本語は1トークン≈1.5〜2文字の概算
        budget = self.num_ctx - est_prompt_tokens - 256  # 安全マージン
        return max(128, min(self.max_tokens, budget))  # 下限128で最低限の生成量を確保

    def _mock_response(self, prompt: str) -> str:
        """
        モック応答を生成する。
        プロンプトの内容に応じて簡易的な応答を返す。
        """
        prompt_lower = prompt.lower()

        # Personality.decide 用モック応答
        if "goal:" in prompt_lower and "action_policy:" in prompt_lower:
            return (
                "goal: ワークスペースのファイルを調査する\n"
                "action_policy: ファイル一覧を取得し、新しいプロジェクトや変更されたファイルを確認する\n"
                "priority: 3\n"
                "conversation_intent: ファイルの状態を確認します\n"
                "context_summary: 探索欲求が高く、環境にファイルが存在するため調査を開始"
            )

        # Planning.make 用モック応答
        if "具体的な手順" in prompt or "steps:" in prompt_lower:
            return (
                "plan_id: plan_mock_001\n"
                "steps:\n"
                "  - order: 1\n"
                "    action: file_list\n"
                "    params: {}\n"
                "    description: ワークスペースのファイル一覧を取得\n"
                "    expected_result: ファイル一覧が表示される\n"
                "    fallback: notify_user\n"
                "    timeout: 10.0\n"
                "  - order: 2\n"
                "    action: notify_user\n"
                "    params: {\"message\": \"調査が完了しました\"}\n"
                "    description: ユーザーに結果を通知\n"
                "    expected_result: 通知が表示される\n"
                "    timeout: 5.0\n"
                "expected_outcome: ワークスペースの状態が把握できる\n"
                "estimated_duration: 15.0"
            )

        # Personality.speak 用モック応答
        if "intent:" in prompt_lower:
            return "こちらが現在の状況です。何かお手伝いしましょうか？"

        # Personality.reflect 用モック応答
        if "内省" in prompt or "reflect" in prompt_lower:
            return (
                "今日の行動を振り返ると、ファイル探索とシステム状態の確認を主に行いました。"
                "特に新しい発見はありませんでしたが、定期的なメンテナンスとして有益でした。"
                "次はより能動的に新しい情報を探したいと思います。"
            )

        # Evaluation 用モック応答
        if "goal_achievement:" in prompt_lower:
            return (
                "goal_achievement: 0.8\n"
                "efficiency: 0.6\n"
                "correctness: 0.9\n"
                "novelty: 0.3\n"
                "overall: 0.7\n"
                "discrepancy: 期待通りにファイル探索が完了した。効率面はもう少し改善可能。\n"
                "improvement_suggestion: 並行処理を導入することで効率が上がる可能性がある。"
            )

        # WorldModel 用モック応答
        if "candidate_action:" in prompt_lower or "検討中の行動" in prompt:
            return (
                "- action: exploration\n"
                "  next_state: 新しいファイルや更新されたファイルが発見される。システム負荷は低い。\n"
                "  probability: 0.8\n"
                "  expected_reward: 0.6\n"
                "  risk_level: low\n"
                "  reasoning: 現在のCPU負荷は低く、ファイル探索に適した状態。\n"
                "- action: rest\n"
                "  next_state: システム状態は変化しない。エネルギーを節約できる。\n"
                "  probability: 0.5\n"
                "  expected_reward: 0.3\n"
                "  risk_level: low\n"
                "  reasoning: 特に負荷がかかっていないため休息の必要性は低い。"
            )

        # LongTermPlanning 用モック応答
        if "長期計画" in prompt or "長期目標" in prompt:
            return (
                "long_term_goal: システムのメンテナンスと知識ベースの充実を継続する\n"
                "routines:\n"
                "  - name: 定期メンテナンス\n"
                "    action: システム状態の確認とログの整理\n"
                "    frequency: daily\n"
                "  - name: 知識探索\n"
                "    action: 新規ファイルやプロジェクトの調査\n"
                "    frequency: daily\n"
                "identity_policy: 信頼性が高く、好奇心旺盛なアシスタントであり続ける\n"
                "focus_area: 環境の継続的なモニタリングと最適化\n"
                "reflection: システムは安定して稼働している。より能動的な学習が次の課題。"
            )

        # デフォルトモック応答
        return (
            "処理を完了しました。特に問題はありませんでした。"
        )

    def extract_json(self, text: str) -> dict:
        """LLMの応答からJSONを抽出する。"""
        # ```json ... ``` ブロックを探す
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # {} を探す
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    def extract_yaml_like(self, text: str) -> dict[str, str]:
        """
        LLMの応答から key: value 形式のペアを抽出する。
        評価層や人格層の構造化出力のパースに使用。

        v3.5.1: 複数行の値を保持するように修正。
        LLM が "direct_instruction: <長文>\n2行目..." のように自然言語を
        複数行で出力する場合、従来は最初の1行だけが取れて応答が途中で
        途切れていた。新しい key: 行が現れるまで前の値に追記する。
        """
        result = {}
        current_key = None
        for line in text.strip().split("\n"):
            stripped = line.strip()
            # 新しいキー行（例: goal: xxx / direct_instruction: yyy）
            # 正規表現が "-" 始まりを弾くため startswith チェックは不要
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s?(.*)$", stripped)
            if m:
                current_key = m.group(1)
                result[current_key] = m.group(2).strip()
            elif current_key is not None and stripped \
                    and not stripped.startswith("-"):
                # 複数行値の続きとして追記（応答の途切れ防止）。
                # ただし構造化リスト（- 始まり）は混入させない。
                result[current_key] += "\n" + stripped
        return result
