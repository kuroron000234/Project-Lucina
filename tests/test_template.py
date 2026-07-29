"""
テストテンプレート

このファイルを各層のテストの雛形として使う。
tests/test_<layer>.py としてコピーして実装する。
"""

import pytest


class MockLLM:
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_history = []

    def chat(self, prompt: str) -> str:
        self.call_history.append(prompt)
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return response
        return self.responses.get("default", "")


# --- 単体テストのテンプレート ---

class TestLayer:
    """
    テスト項目（各層共通）:
    1. 入力 → 出力が正しい型で返る
    2. エッジケース（空入力、None）でエラーにならない
    3. 永続化が必要な層は save/load が正しく動く
    4. LLM依存の層はモックで動作確認
    """

    @pytest.fixture
    def mock_llm(self):
        return MockLLM({
            "default": "output_placeholder",
        })

    def test_normal_case(self, mock_llm):
        """正常系: 標準的な入力で期待される出力が得られる"""
        pass

    def test_edge_empty_input(self, mock_llm):
        """エッジケース: 空の入力でエラーにならない"""
        pass

    def test_edge_none_values(self, mock_llm):
        """エッジケース: Noneを含む入力でエラーにならない"""
        pass


# --- 結合テストのテンプレート ---

class TestIntegration:
    """
    結合テスト項目:
    1. 前の層の出力が次の層の入力としてそのまま使える
    2. 全層を通したデータの流れが途切れない
    3. 型の互換性が保たれている
    """

    def test_data_flow(self):
        """隣接する2層間のデータ受け渡しが正しい型で行われる"""
        pass

    def test_main_loop_once(self):
        """メインループを1サイクル回せる"""
        pass
