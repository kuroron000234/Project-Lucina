"""
記憶層 (Memory)

責務: エピソード記憶の保存・検索・要約を行う。経験のデータベース。
Phase 1: キーワードマッチング + 日時ソート + JSONファイル保存
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import config
from core.memory.interface import (
    Episode,
    EpisodeSummary,
    MemoryInput,
    MemoryOutput,
)

logger = logging.getLogger("Memory")


class Memory:
    """
    記憶層: エピソード記憶の保存・検索・要約を担当する。
    Phase 1 では単純なキーワード検索 + JSONファイル永続化。
    """

    def __init__(self, storage_path: str = "data/episodes/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.episodes: list[Episode] = []
        self._load()

    def search(self, input: MemoryInput) -> MemoryOutput:
        """
        クエリに基づいてエピソードを検索し、要約とともに返す。

        エッジケース:
        - エピソード0件: episodes=[], summary="まだ記憶がありません"
        - 重要度一様: すべて同じ重要度の場合、時系列順
        - クエリが空文字: 最近のエピソードを top_k 件返す
        """
        # クエリが空の場合は最近のエピソードを返す
        if not input.query.strip():
            results = sorted(
                self.episodes,
                key=lambda e: e.timestamp,
                reverse=True,
            )[:input.top_k]
            return MemoryOutput(
                episodes=results,
                summary=self._summarize(results) if results else "まだ記憶がありません",
                total_count=len(self.episodes),
            )

        # v5.0: ハイブリッド検索（キーワード完全一致 + 文字n-gram類似度）
        # 記憶保持ベンチマークで実測された弱点（言い換え・表記揺れで0ヒット）を
        # 改善する。キーワードヒットは常に採用し、類似度のみのヒットは閾値以上のみ。
        query_lower = input.query.lower()
        hybrid_cfg = config.MEMORY_CONFIG.get("hybrid", {})
        use_hybrid = input.use_hybrid and hybrid_cfg.get("enabled", True)
        n = int(hybrid_cfg.get("n_gram_size", 2))
        sim_weight = float(hybrid_cfg.get("similarity_weight", 1.0))
        min_sim = float(hybrid_cfg.get("min_similarity", 0.25))
        max_full = int(hybrid_cfg.get("max_episodes_full_scan", 5000))
        query_ngrams = self._char_ngrams(query_lower, n) if use_hybrid else None

        scored = []
        for ep in self.episodes:
            # 時間範囲フィルタ
            if input.time_range:
                start, end = input.time_range
                if not (start <= ep.timestamp <= end):
                    continue
            # 重要度フィルタ
            if ep.importance < input.min_importance:
                continue

            # イベント記述・コンテキスト・タグでの完全一致
            keyword = (
                query_lower in ep.event.lower()
                or query_lower in ep.context.lower()
                or query_lower in ep.result.lower()
                or any(query_lower in tag.lower() for tag in ep.tags)
            )

            if use_hybrid and query_ngrams and len(self.episodes) <= max_full:
                # 類似度スコア（文字n-gramコサイン）。完全一致は必ず採用。
                text = f"{ep.event} {ep.context} {ep.result} {' '.join(ep.tags)}"
                sim = self._ngram_similarity(query_ngrams, text.lower(), n)
                if keyword or sim >= min_sim:
                    score = 1.0 + sim_weight * sim if keyword else sim_weight * sim
                    scored.append((score, ep.importance,
                                   ep.timestamp.timestamp(), ep))
            else:
                if keyword:
                    scored.append((1.0, ep.importance,
                                   ep.timestamp.timestamp(), ep))

        # スコア → 重要度 → 時系列 の降順でソート
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        results = [s[3] for s in scored[:input.top_k]]

        return MemoryOutput(
            episodes=results,
            summary=self._summarize(results) if results else "まだ記憶がありません",
            total_count=len(self.episodes),
        )

    def save(self, episode: Episode) -> str:
        """
        新しいエピソードを保存する。戻り値はエピソードID。

        v3.2: 双方向リウェイトを適用（類似エピソードの繰り返しで既存の
        類似エピソードの重要度を下げ、多様性圧力を双方向にする）。

        エッジケース:
        - 保存失敗: ファイル書き込みエラー時の再試行は行わずログ出力
        """
        # IDが空の場合は自動生成
        if not episode.id:
            episode.id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # 双方向リウェイト: 保存前に既存の類似エピソードを減点
        self._reweight_duplicates(episode)

        self.episodes.append(episode)

        # JSONとして保存
        try:
            filepath = self.storage_path / f"{episode.id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._episode_to_dict(episode), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save episode {episode.id}: {e}")

        return episode.id

    def _episode_to_dict(self, episode: Episode) -> dict:
        """エピソードをJSON保存用dictに変換する。"""
        return {
            "id": episode.id,
            "timestamp": episode.timestamp.isoformat(),
            "event": episode.event,
            "context": episode.context,
            "emotion": episode.emotion,
            "result": episode.result,
            "importance": episode.importance,
            "tags": episode.tags,
            "source": getattr(episode, "source", "autonomous"),
            "driving_drive": getattr(episode, "driving_drive", ""),
        }

    def _canonical_goal(self, event: str) -> str:
        """
        イベント記述から類似判定用の正規化キーを生成する。

        助詞・補助動詞・空白を除去し、先頭12文字をキーにする。
        """
        if not event:
            return ""
        cleaned = event.strip().lower()
        # 日本語の助詞・補助動詞を除去
        for token in [" ", "　", "を", "の", "が", "に", "へ", "する", "したい", "たい"]:
            cleaned = cleaned.replace(token, "")
        return cleaned[:12]

    @staticmethod
    def _char_ngrams(text: str, n: int = 2) -> set:
        """文字n-gram集合を返す（v5.0: ハイブリッド検索用）。"""
        if not text:
            return set()
        if n <= 1 or len(text) < n:
            return {text}
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    @classmethod
    def _ngram_similarity(cls, query_ngrams: set, text: str, n: int) -> float:
        """文字n-gramコサイン類似度（0.0〜1.0）。"""
        text_ngrams = cls._char_ngrams(text, n)
        if not query_ngrams or not text_ngrams:
            return 0.0
        overlap = len(query_ngrams & text_ngrams)
        return overlap / (len(query_ngrams) * len(text_ngrams)) ** 0.5

    def repetition_count(self, goal: str, window: int = 10) -> int:
        """
        直近window件のエピソードで、指定goalと類似する件数を返す。
        """
        key = self._canonical_goal(f"goal={goal}")
        if not key:
            return 0
        recent = self.episodes[-window:]
        return sum(1 for ep in recent if self._canonical_goal(ep.event) == key)

    def _reweight_duplicates(self, new_episode: Episode, window: int = 10):
        """
        同一canonicalキーのエピソードが繰り返された場合、既存の類似
        エピソードの重要度を 0.03 ずつ減点して永続化する（多様性確保）。
        """
        key = self._canonical_goal(new_episode.event)
        if not key:
            return
        recent = self.episodes[-window:]
        similar = [ep for ep in recent if self._canonical_goal(ep.event) == key]
        if len(similar) >= 2:
            for ep in similar:
                if ep.importance > 0.05:
                    ep.importance = max(0.0, ep.importance - 0.03)
                    self._save_single(ep)

    def update_importance(self, episode_id: str, new_importance: float):
        """
        エピソードの重要度を更新する（学習層から呼ばれる）。
        """
        for ep in self.episodes:
            if ep.id == episode_id:
                old = ep.importance
                ep.importance = max(0.0, min(1.0, new_importance))
                logger.debug(f"Updated importance: {episode_id} {old:.2f} -> {ep.importance:.2f}")
                # JSONファイルも更新
                self._save_single(ep)
                return
        logger.warning(f"Episode not found for importance update: {episode_id}")

    def summarize(self, episodes: list[Episode]) -> str:
        """
        エピソードリストを自然言語で要約する。
        """
        return self._summarize(episodes)

    def forget(self, threshold: float = 0.1):
        """
        重要度が閾値以下のエピソードを削除する（メモリ節約）。

        エッジケース:
        - 全エピソードが閾値以上: 何もしない
        """
        before = len(self.episodes)
        # 削除対象（重要度が閾値未満）と保持対象に分ける
        kept = []
        forgotten = []
        for ep in self.episodes:
            if ep.importance >= threshold:
                kept.append(ep)
            else:
                forgotten.append(ep)

        self.episodes = kept
        removed = len(forgotten)
        if removed > 0:
            logger.info(f"Forgot {removed} episodes (threshold={threshold})")
            # 削除されたエピソードのファイルも削除
            for ep in forgotten:
                filepath = self.storage_path / f"{ep.id}.json"
                try:
                    if filepath.exists():
                        filepath.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete file for forgotten episode {ep.id}: {e}")

    def get_statistics(self) -> dict:
        """
        記憶の統計情報を返す（デバッグ/評価用）。
        """
        if not self.episodes:
            return {
                "total_episodes": 0,
                "avg_importance": 0.0,
                "date_range": None,
                "tag_distribution": {},
            }

        importances = [ep.importance for ep in self.episodes]
        tags = {}
        for ep in self.episodes:
            for tag in ep.tags:
                tags[tag] = tags.get(tag, 0) + 1

        return {
            "total_episodes": len(self.episodes),
            "avg_importance": sum(importances) / len(importances),
            "date_range": (
                min(ep.timestamp for ep in self.episodes).isoformat(),
                max(ep.timestamp for ep in self.episodes).isoformat(),
            ),
            "tag_distribution": dict(sorted(tags.items(), key=lambda x: -x[1])),
        }

    def _load(self):
        """ストレージから全エピソードをメモリに読み込む。"""
        loaded = 0
        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                episode = Episode(
                    id=data["id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    event=data.get("event", ""),
                    context=data.get("context", ""),
                    emotion=data.get("emotion", ""),
                    result=data.get("result", ""),
                    importance=data.get("importance", 0.5),
                    tags=data.get("tags", []),
                    source=data.get("source", "autonomous"),
                    driving_drive=data.get("driving_drive", ""),
                )
                self.episodes.append(episode)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load episode from {filepath}: {e}")
        logger.info(f"Loaded {loaded} episodes from {self.storage_path}")

    def _save_single(self, episode: Episode):
        """単一エピソードをJSONファイルに保存する。"""
        try:
            filepath = self.storage_path / f"{episode.id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._episode_to_dict(episode), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save episode {episode.id}: {e}")

    def _summarize(self, episodes: list[Episode]) -> str:
        """
        エピソードリストの要約を生成する。
        Phase 1: ルールベースの簡易要約
        """
        if not episodes:
            return "まだ記憶がありません"

        # 時間範囲
        timestamps = [ep.timestamp for ep in episodes]
        time_range = f"{min(timestamps).strftime('%m/%d %H:%M')}~{max(timestamps).strftime('%m/%d %H:%M')}"

        # 重要度の分布
        high_imp = sum(1 for ep in episodes if ep.importance >= 0.7)
        mid_imp = sum(1 for ep in episodes if 0.3 <= ep.importance < 0.7)
        low_imp = sum(1 for ep in episodes if ep.importance < 0.3)

        # 主なタグ
        tag_counts = {}
        for ep in episodes:
            for tag in ep.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:3]

        parts = [
            f"📅 期間: {time_range}",
            f"📊 {len(episodes)}件のエピソード (高重要度:{high_imp}, 中:{mid_imp}, 低:{low_imp})",
        ]
        if top_tags:
            tags_str = ", ".join(f"{t}({c}回)" for t, c in top_tags)
            parts.append(f"🏷️ 主なトピック: {tags_str}")

        # 直近のイベント
        recent = sorted(episodes, key=lambda e: e.timestamp, reverse=True)[:3]
        events = [f"  - {ep.timestamp.strftime('%H:%M')} {ep.event[:50]}" for ep in recent]
        parts.append("🕐 最近の出来事:")
        parts.extend(events)

        return "\n".join(parts)
