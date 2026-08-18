"""DriveVocabExpander — 語彙の半自動拡張（仕様書 v1.4 §5.2）。

- 起動時に1回だけ build_vocab_map() を呼び、結果はプロセス生存中キャッシュする。
- シード語彙（config/seed_vocab.yaml）を起点に、埋め込み類似度 sim_threshold 以上・
  上位 top_k 件の語を各Driveの語彙へ追加する。
- 語彙→トークンID列の解決（BPEサブワード）は tokenizer 経由で行う。
- 返り値: {drive: [[token_id, ...], ...]}。このマップはバイアス適用（§5.4）と
  relief判定（③）で共有される。
- 運用フック: 拡張結果は logging 経由で必ずINFOログに出力し、人間が目視レビューできるようにする。

候補語彙の選択（実モデル対応）:
    トークンIDの先頭から順に max_candidates 件を取ると、多くのモデルでは記号・
    特殊トークンばかりになり日本語語彙に到達しない。そのため、語彙全体から
    ランダムサンプリングし、日本語（ひらがな/カタカナ/漢字）を含むトークンを
    優先して候補とする（日本語性能の高いモデルほど効果が大きい）。
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from ..config import PROJECT_ROOT


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_id: int) -> str: ...
    def words(self, max_words: int = 0) -> list[str]: ...
    def vocab_size(self) -> int: ...


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    def embed_many(self, texts: list[str]) -> np.ndarray: ...


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(float(np.dot(a, b)) / (na * nb))


def _resolve_seed_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


class DriveVocabExpander:
    def __init__(self, config: dict[str, Any], tokenizer: Tokenizer, embedder: Embedder, *, logger=None):
        """config: drive.vocab_expansion 相当（top_k, sim_threshold, seed_vocab_path）。"""
        self._config = config
        self._tokenizer = tokenizer
        self._embedder = embedder
        self._logger = logger
        self._cache: dict[str, list[list[int]]] | None = None

    def build_vocab_map(self, on_progress=None) -> dict[str, list[list[int]]]:
        """Drive名 -> 語彙トークンID列のリスト。起動時に1回だけ呼ばれ、以降はキャッシュを返す。

        on_progress: 任意のコールバック (index, total, drive) -> None。
        Drive ごとの語彙拡張の進捗を報告する（v1.15・Web UI のロード進捗表示用）。
        キャッシュ済みの場合は呼ばれない。
        """
        if self._cache is not None:
            return self._cache

        seed_path = _resolve_seed_path(self._config["seed_vocab_path"])
        with seed_path.open("r", encoding="utf-8") as fh:
            seed_vocab: dict[str, list[str]] = yaml.safe_load(fh) or {}

        top_k = int(self._config["top_k"])
        sim_threshold = float(self._config["sim_threshold"])
        max_candidates = int(self._config.get("max_candidates", 0))  # 0 = 無制限
        candidates = self._sample_candidates(max_candidates)

        # 候補語彙は1回だけバッチ埋め込みし、シードとの類似度をベクトル化で計算する
        # （実モデルでは語彙が数万〜数十万になるため、1語ずつの埋め込みは非現実的）。
        cand_embs = self._embedder.embed_many(candidates)
        seed_set = {w for ws in seed_vocab.values() for w in ws}

        result: dict[str, list[list[int]]] = {}
        drives = list(seed_vocab.items())
        total = len(drives)
        for index, (drive, seeds) in enumerate(drives):
            if on_progress is not None:
                on_progress(index, total, drive)
            seed_embs = np.stack([self._embedder.embed(w) for w in seeds])
            sims = cand_embs @ seed_embs.T  # (N, n_seeds)
            max_sim = sims.max(axis=1)
            kept = list(seeds)  # シード語彙は常に保持
            order = np.argsort(-max_sim)
            for idx in order:
                word = candidates[idx]
                if word in seed_set:
                    continue
                if float(max_sim[idx]) < sim_threshold:
                    break  # 降順のため、以降は全て閾値未満
                kept.append(word)
                if len(kept) >= len(seeds) + top_k:
                    break
            result[drive] = [seq for seq in (self._encode_word(w) for w in kept) if seq]

        self._cache = result
        if self._logger is not None:
            self._logger.vocab_expansion(result)  # 運用フック: 必ずINFOログ出力
        return result

    # ------------------------------------------------------------------ #
    def _encode_word(self, word: str) -> list[int]:
        """語彙トークン列の解決。先頭の空白のみトークンを除去する。

        T5系トークナイザ（llm-jp-4 等）は単語の先頭に空白トークンを付与する
        （encode("話") -> [空白, 話]）。DriveLogitsProcessor は先頭トークンのみに
        バイアスを加算するため（§5.4⑤）、空白トークンが先頭だとバイアスが
        空白へ集中し「空白の連続生成」ループに陥る（実モデル検証で確認）。
        先頭の空白のみトークンを除去することで、意味のあるトークンへバイアスが
        届くようになる。BPE系（Qwen等）は先頭空白がないため無影響。
        """
        seq = self._tokenizer.encode(word)
        while seq and not self._tokenizer.decode(int(seq[0])).strip():
            seq = seq[1:]
        return seq

    def _sample_candidates(self, max_candidates: int) -> list[str]:
        """語彙拡張の候補語を選ぶ。

        - max_candidates <= 0: 語彙全体を対象（max_words=0 で全件復号）。
        - max_candidates > 0: 語彙全体からランダムサンプリングし、日本語トークンを優先。
          先頭IDから順に取る方式は記号ばかりになる問題があったため（実モデル調査）。
        """
        vocab_size = self._tokenizer.vocab_size()
        if max_candidates <= 0 or max_candidates >= vocab_size:
            return [w for w in self._tokenizer.words(0) if w]

        # 日本語らしいトークンを優先するための判定（ひらがな/カタカナ/漢字を含む）
        jp_re = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

        # サンプリング対象のIDをランダムに選ぶ（決定性を保つため固定シード）
        rng = random.Random(42)
        sampled_ids = rng.sample(range(vocab_size), max_candidates)
        candidates: list[str] = []
        seen: set[str] = set()
        # 日本語トークンを先に、記号トークンを後に並べる（類似度計算の対象を効率化）
        jp_words: list[str] = []
        other_words: list[str] = []
        for tid in sampled_ids:
            w = self._tokenizer.decode(int(tid)).strip()
            if not w or w in seen:
                continue
            seen.add(w)
            if jp_re.search(w):
                jp_words.append(w)
            else:
                other_words.append(w)
        candidates = jp_words + other_words
        # 候補が少なすぎる場合は補填（サンプリングで重複が多かった場合）
        if len(candidates) < max_candidates:
            extra_ids = [tid for tid in range(vocab_size) if tid not in set(sampled_ids)]
            rng.shuffle(extra_ids)
            for tid in extra_ids[: max_candidates - len(candidates)]:
                w = self._tokenizer.decode(int(tid)).strip()
                if w and w not in seen:
                    seen.add(w)
                    if jp_re.search(w):
                        candidates.insert(0, w)
                    else:
                        candidates.append(w)
        return candidates
