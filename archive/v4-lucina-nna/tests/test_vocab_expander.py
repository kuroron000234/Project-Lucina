"""DriveVocabExpander._encode_word の先頭空白トークン除去の検証。

T5系トークナイザ（llm-jp-4 等）は単語の先頭に空白トークンを付与するため、
先頭トークンへのみバイアスを加算する DriveLogitsProcessor と組み合わせると
空白トークンへバイアスが集中し「空白連続生成」ループに陥る。
本テストは _encode_word が先頭の空白のみトークンを除去することを固定化する。
"""

from __future__ import annotations

from lucina.drives.vocab import DriveVocabExpander


class _T5LikeTokenizer:
    """encode() が単語の先頭に空白トークン(0)を付与するトークナイザ（llm-jp-4 の挙動を模す）。"""

    _DECODE = {0: " ", 10: "話", 11: "寂しい", 12: "友達", 13: "会", 14: "一人"}

    def encode(self, text: str) -> list[int]:
        w = text.strip()
        table = {"話": [0, 10], "寂しい": [0, 11], "友達": [12], "会": [0, 13], "一人": [0, 14]}
        return list(table.get(w, [0, 999]))

    def decode(self, token_id: int) -> str:
        return self._DECODE.get(int(token_id), "x")

    def words(self, max_words: int = 0) -> list[str]:  # noqa: ARG002 - プロトコル互換
        return ["話", "寂しい", "友達"]

    def vocab_size(self) -> int:
        return 1000


class _NoopEmbedder:
    def embed(self, text: str):  # noqa: ARG002 - build_vocab_map 用のダミー
        import numpy as np

        return np.zeros(4)

    def embed_many(self, texts: list[str]):  # noqa: ARG002
        import numpy as np

        return np.zeros((len(texts), 4))


def test_encode_word_strips_leading_whitespace_token():
    tok = _T5LikeTokenizer()
    expander = DriveVocabExpander({}, tok, _NoopEmbedder())

    assert expander._encode_word("話") == [10]  # [0, 10] から空白トークンを除去
    assert expander._encode_word("寂しい") == [11]
    assert expander._encode_word("一人") == [14]


def test_encode_word_keeps_non_whitespace_leading_token():
    tok = _T5LikeTokenizer()
    expander = DriveVocabExpander({}, tok, _NoopEmbedder())

    assert expander._encode_word("友達") == [12]  # 先頭空白なしはそのまま


def test_build_vocab_map_has_no_whitespace_leading_sequences(tmp_path):
    """build_vocab_map の結果に先頭が空白のみの語彙列が含まれないことを確認。"""
    import yaml

    seed_path = tmp_path / "seed_vocab.yaml"
    seed_path.write_text(
        yaml.safe_dump({"loneliness": ["寂しい", "一人"]}, allow_unicode=True),
        encoding="utf-8",
    )
    tok = _T5LikeTokenizer()
    expander = DriveVocabExpander(
        {"top_k": 30, "sim_threshold": 0.0, "seed_vocab_path": str(seed_path)},
        tok,
        _NoopEmbedder(),
    )
    vocab_map = expander.build_vocab_map()

    for drive, seqs in vocab_map.items():
        for seq in seqs:
            assert seq, f"{drive} に空の語彙列がある"
            decoded = tok.decode(seq[0])
            assert decoded.strip(), f"{drive} の語彙列が空白トークンで始まる: {seq}"


def test_build_vocab_map_reports_progress_per_drive(tmp_path):
    """v1.15: on_progress コールバックが Drive ごとに (index, total, drive) で呼ばれる。"""
    import yaml

    seed_path = tmp_path / "seed_vocab.yaml"
    seed_path.write_text(
        yaml.safe_dump(
            {"loneliness": ["寂しい", "一人"], "boredom": ["話"]}, allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    tok = _T5LikeTokenizer()
    expander = DriveVocabExpander(
        {"top_k": 30, "sim_threshold": 0.0, "seed_vocab_path": str(seed_path)},
        tok,
        _NoopEmbedder(),
    )
    calls: list[tuple[int, int, str]] = []
    expander.build_vocab_map(on_progress=lambda i, t, d: calls.append((i, t, d)))

    assert calls == [(0, 2, "loneliness"), (1, 2, "boredom")]
    # キャッシュ済みの再呼び出しでは進捗は呼ばれない
    calls.clear()
    expander.build_vocab_map(on_progress=lambda i, t, d: calls.append((i, t, d)))
    assert calls == []
