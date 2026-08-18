"""テスト・モック実行用コンポーネント。

- 実モデル（GGUF）なしで全パイプライン（Drive力学系→語彙拡張→ロジットバイアス→
  relief→記憶）を検証できる。tests/ と scripts/run_agent.py --mock で使用。
- MockTokenizer / FakeEmbedder / MockTokenBackend / FakeSummarizer / build_mock_core
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .core import LucinaCore
from .drives.vocab import DriveVocabExpander
from .inference.backends import TokenBackend
from .inference.engine import InferenceEngine
from .io.logging import StructuredLogger
from .memory.store import HierarchicalMemoryStore, InMemoryVectorStore, MemoryCompressor

SEED_WORDS = [
    "寂しい", "会いたい", "一緒に", "話したい", "誰か",
    "つまらない", "新しい", "試してみる", "冒険", "何か",
    "休む", "疲れた", "ゆっくり", "眠い", "おやすみ",
]

# シード語彙に近い埋め込みを持つ拡張候補（FakeEmbedder の類似度構造で拡張される）
EXPAND_WORDS = {
    "loneliness": ["孤独", "友人"],
    "boredom": ["退屈", "刺激"],
    "fatigue": ["眠気"],
}


class MockTokenizer:
    """決定的な語→トークンID列（BPEを模して1語=2トークン）を持つトークナイザ。"""

    def __init__(self, vocab_size: int = 512, start_id: int = 100) -> None:
        self._vocab_size = int(vocab_size)
        self._start = int(start_id)
        self._words: list[str] = []
        self._word_ids: dict[str, list[int]] = {}
        self._id_word: dict[int, str] = {}
        for w in SEED_WORDS:
            self._add_word(w)
        for group in EXPAND_WORDS.values():
            for w in group:
                if w not in self._word_ids:
                    self._add_word(w)

    def _add_word(self, word: str) -> None:
        n = self._start + 2 * len(self._words)
        ids = [n, n + 1]
        self._words.append(word)
        self._word_ids[word] = ids
        self._id_word[ids[0]] = word

    def encode(self, text: str) -> list[int]:
        word = text.strip()
        if word in self._word_ids:
            return list(self._word_ids[word])
        # 未知語は語彙ID（<=138付近）と衝突しない高位側へ決定的にマッピング
        h = self._vocab_size - 3 - (abs(hash(word)) % (self._vocab_size - 200))
        return [h, h + 1]

    def first_token_ids(self) -> set[int]:
        """語彙の先頭トークンID集合（MockTokenBackend がタイブレーク用摂動から除外する）。"""
        return set(self._id_word.keys())

    def decode(self, token_id: int) -> str:
        tid = int(token_id)
        if tid in self._id_word:
            return self._id_word[tid]
        return f"tok{tid}"

    def words(self, max_words: int = 0) -> list[str]:
        all_words = self._words + [f"filler{i}" for i in range(60)]
        return all_words[:max_words] if max_words > 0 else all_words

    def vocab_size(self) -> int:
        return self._vocab_size


class FakeEmbedder:
    """決定的な埋め込み。シード語彙と拡張候補はDrive軸に近いベクトルを持つ。"""

    _AXIS = {
        "loneliness": np.array([1.0, 0.0, 0.0]),
        "boredom": np.array([0.0, 1.0, 0.0]),
        "fatigue": np.array([0.0, 0.0, 1.0]),
    }

    def __init__(self) -> None:
        self._vec: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(0)
        mapping: dict[str, str] = {
            "寂しい": "loneliness", "会いたい": "loneliness", "一緒に": "loneliness",
            "話したい": "loneliness", "誰か": "loneliness", "孤独": "loneliness", "友人": "loneliness",
            "つまらない": "boredom", "新しい": "boredom", "試してみる": "boredom",
            "冒険": "boredom", "何か": "boredom", "退屈": "boredom", "刺激": "boredom",
            "休む": "fatigue", "疲れた": "fatigue", "ゆっくり": "fatigue",
            "眠い": "fatigue", "おやすみ": "fatigue", "眠気": "fatigue",
        }
        for word, drive in mapping.items():
            base = self._AXIS[drive] * 0.95 + rng.normal(0.0, 0.01, 3)
            self._vec[word] = base

    def embed(self, text: str) -> np.ndarray:
        word = text.strip()
        if word in self._vec:
            return self._vec[word]
        # 未知語は零ベクトル: cosine=0 となり、語彙拡張の候補として混入しない
        return np.zeros(3, dtype=np.float64)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class MockTokenBackend(TokenBackend):
    """ニュートラルなベースロジットを返すバックエンド。

    実際のDriveバイアス適用（DriveLogitsProcessor）はエンジン側で行われるため、
    本バックエンドは極小の決定的な摂動のみを返す（タイブレーク用）。
    delay_ms: 実モデルのレイテンシを模して1トークンごとにスリープする（タイミング計測用）。
    """

    def __init__(self, tokenizer: MockTokenizer, delay_ms: float = 0.0) -> None:
        self._tokenizer = tokenizer
        self.delay_ms = float(delay_ms)
        self._step = 0

    def next_token_logits(self, context_text: str) -> np.ndarray:
        if self.delay_ms > 0.0:
            time.sleep(self.delay_ms / 1000.0)
        self._step += 1
        n = self._tokenizer.vocab_size()
        base = 0.001 * (((np.arange(n) * 2654435761) % 10000) / 10000.0)
        # 語彙の先頭トークンはDriveバイアス0の時に選択されないよう摂動を除外（決定性の確保）
        base[list(self._tokenizer.first_token_ids())] = 0.0
        return base

    def decode(self, token_id: int) -> str:
        return self._tokenizer.decode(int(token_id))

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size()


class FakeSummarizer:
    """要約器モック（テスト・モック実行用）。"""

    def summarize(self, text: str) -> str:
        return "[要約] " + text[:40]


def make_test_config(**overrides: Any) -> dict:
    """仕様書付属の config/default.yaml を読み込み、テスト向けに上書きした設定を返す。"""
    cfg = load_config()
    cfg["model"]["context_window"] = int(overrides.pop("context_window", 256))
    cfg["inference"]["sampling"] = overrides.pop("sampling", "greedy")
    cfg["inference"]["seed"] = int(overrides.pop("seed", 0))
    cfg["logging"]["log_dir"] = str(overrides.pop("log_dir", "./reports"))
    for key, value in overrides.items():
        # ドット区切りでネスト指定（例: "drive.relief.boredom.enabled": False）
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def build_mock_core(
    config: dict[str, Any] | None = None,
    *,
    token_delay_ms: float = 0.0,
    log_dir: str | Path = "./reports",
    with_embedder: bool = True,
    on_progress=None,
) -> LucinaCore:
    """MockTokenizer/FakeEmbedder/MockTokenBackend で組み立てた LucinaCore を返す。"""
    cfg = config if config is not None else make_test_config(log_dir=log_dir)
    tokenizer = MockTokenizer()
    embedder = FakeEmbedder() if with_embedder else None

    if embedder is not None:
        expander = DriveVocabExpander(cfg["drive"]["vocab_expansion"], tokenizer, embedder)
        vocab_map = expander.build_vocab_map()
    else:
        vocab_map = {}

    backend = MockTokenBackend(tokenizer, delay_ms=token_delay_ms)
    executor = ThreadPoolExecutor(max_workers=2)
    engine = InferenceEngine(
        llm_path="mock",
        executor=executor,
        backend=backend,
        vocab_map=vocab_map,
        config=cfg,
    )
    logger = StructuredLogger(log_dir)
    from .memory.classifier import RuleBasedMemoryClassifier

    memory = HierarchicalMemoryStore(
        InMemoryVectorStore(),
        embedder=embedder,  # type: ignore[arg-type]
        classifier=RuleBasedMemoryClassifier(),  # v1.11: 記憶分類器を実配線
    )
    compressor = MemoryCompressor(FakeSummarizer())
    core = LucinaCore(cfg, engine, vocab_map, memory=memory, compressor=compressor, logger=logger)
    core._executor = executor  # テスト終了時に close するための参照
    return core
