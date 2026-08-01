"""
記憶層 (Memory)

責務: エピソード記憶の保存・検索・要約を行う。経験のデータベース。
Phase 1: キーワードマッチング + 日時ソート + JSONファイル保存
"""

import json
import logging
from datetime import datetime
from pathlib import Path

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

        # キーワードマッチング
        query_lower = input.query.lower()
        results = []
        for ep in self.episodes:
            # イベント記述・コンテキスト・タグでマッチング
            if (
                query_lower in ep.event.lower()
                or query_lower in ep.context.lower()
                or query_lower in ep.result.lower()
                or any(query_lower in tag.lower() for tag in ep.tags)
            ):
                results.append(ep)

        # 時間範囲フィルタ
        if input.time_range:
            start, end = input.time_range
            results = [
                ep for ep in results
                if start <= ep.timestamp <= end
            ]

        # 重要度フィルタ
        results = [
            ep for ep in results
            if ep.importance >= input.min_importance
        ]

        # 重要度降順でソート（同率の場合は時系列降順）
        results.sort(key=lambda e: (e.importance, e.timestamp.timestamp()), reverse=True)
        results = results[:input.top_k]

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
