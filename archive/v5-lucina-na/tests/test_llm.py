"""
LLMクライアント層の単体テスト（v3.5.1: 複数行値のパース修正 / v3.5.2: max_tokens クランプ）
"""

from core.llm import LLMClient


class TestEffectiveMaxTokens:
    """v3.5.2: プロンプト長に応じた max_tokens の動的クランプ"""

    def test_short_prompt_keeps_configured_max(self):
        """プロンプトが短ければ設定値（2048）のまま"""
        c = LLMClient(max_tokens=2048, num_ctx=4096)
        assert c._effective_max_tokens("短いプロンプト") == 2048

    def test_long_prompt_clamped_to_window(self):
        """プロンプトが長いと窓内に収まるようクランプされる"""
        c = LLMClient(max_tokens=2048, num_ctx=4096)
        # 8000文字 ≈ 4000トークン → 4096-4000-256 < 128 → 下限128
        long_prompt = "あ" * 8000
        assert c._effective_max_tokens(long_prompt) == 128

    def test_mid_prompt_partially_clamped(self):
        """中間の長さでは残り予算に応じて減少する"""
        c = LLMClient(max_tokens=2048, num_ctx=4096)
        # 3000文字 ≈ 1500トークン → budget = 4096-1500-256 = 2340 → 2048のまま
        prompt = "あ" * 3000
        assert c._effective_max_tokens(prompt) == 2048
        # 5000文字 ≈ 2500トークン → budget = 4096-2500-256 = 1340
        prompt2 = "あ" * 5000
        assert c._effective_max_tokens(prompt2) == 1340

    def test_num_ctx_none_returns_configured(self):
        """num_ctx 未設定時は設定値をそのまま返す"""
        c = LLMClient(max_tokens=1024)
        c.num_ctx = None  # __init__ は config 値にフォールバックするため明示的に無効化
        assert c._effective_max_tokens("あ" * 10000) == 1024

    def test_system_prompt_counts_toward_budget(self):
        """system_prompt も予算計算に含まれる"""
        c = LLMClient(max_tokens=2048, num_ctx=4096)
        sys = "あ" * 5000  # ≈2500トークン
        # 総文字数 5002 → 5002//2 = 2501 → budget = 4096-2501-256 = 1339
        assert c._effective_max_tokens("短い", sys) == 1339


class TestConfigAuthoritative:
    """v3.5.2: LLMClient のデフォルトが config 値を反映する（回帰テスト）"""

    def test_max_tokens_from_config(self):
        """無引数生成時に config.LLM_CONFIG['max_tokens'] が使われる"""
        import config as cfg
        c = LLMClient()
        assert c.max_tokens == cfg.LLM_CONFIG.get("max_tokens", 1024)

    def test_temperature_from_config(self):
        """無引数生成時に config.LLM_CONFIG['temperature'] が使われる"""
        import config as cfg
        c = LLMClient()
        assert c.temperature == cfg.LLM_CONFIG.get("temperature", 0.7)

    def test_num_ctx_from_config(self):
        """無引数生成時に config.LLM_CONFIG['num_ctx'] が使われる"""
        import config as cfg
        c = LLMClient()
        assert c.num_ctx == cfg.LLM_CONFIG.get("num_ctx")


class TestExtractYamlLike:
    def test_single_line_values(self):
        """単一行の key: value ペアが抽出できる"""
        c = LLMClient()
        text = (
            "goal: テスト目標\n"
            "action_policy: ポリシー\n"
            "priority: 3\n"
        )
        d = c.extract_yaml_like(text)
        assert d["goal"] == "テスト目標"
        assert d["action_policy"] == "ポリシー"
        assert d["priority"] == "3"

    def test_multiline_value_preserved(self):
        """複数行の値（direct_instruction 等）が途中で切れず保持される"""
        c = LLMClient()
        text = (
            "goal: テスト\n"
            "direct_mode: true\n"
            "direct_instruction: 1行目です。\n"
            "2行目です。ここは無視されない。\n"
            "3行目。\n"
            "action_policy: ポリシー\n"
        )
        d = c.extract_yaml_like(text)
        # v3.5.1: 複数行がすべて保持される（従来は1行目のみ）
        assert "1行目です。" in d["direct_instruction"]
        assert "2行目です。" in d["direct_instruction"]
        assert "3行目。" in d["direct_instruction"]
        # 次のキーで値が終了する
        assert d["action_policy"] == "ポリシー"

    def test_list_lines_not_treated_as_keys(self):
        """リスト形式（- 始まり）の行は新しいキーと誤認せず、値にも混入しない"""
        c = LLMClient()
        text = (
            "steps:\n"
            "  - order: 1\n"
            "    action: file_list\n"
            "expected_outcome: 完了\n"
        )
        d = c.extract_yaml_like(text)
        assert d["expected_outcome"] == "完了"
        assert d["action"] == "file_list"
        # "-" 始まりのリスト行はキーにも値にも混入しない
        assert d.get("order") is None
        assert "- order" not in d.get("steps", "")

    def test_empty_value_key(self):
        """値が空のキーも正しく扱える"""
        c = LLMClient()
        text = "conversation_intent: \ncontext_summary: テスト"
        d = c.extract_yaml_like(text)
        assert d["conversation_intent"] == ""
        assert d["context_summary"] == "テスト"
