"""
記憶 — 階層メモリ: Episode永続化 + 多チャネル検索 + 日次要約（統合）

実装はエージェント記憶の実用OSS（Engram / Woven Imprint / Mem0等）の
共通パターンを流用:
- 階層: Working(会話) / Episodic(episode) / Semantic(要約・事実)
- 検索: キーワード / n-gram / 新しさ(recency) / 重要度(importance) を
  Reciprocal Rank Fusion 的に統合
- 統合(consolidation): 直近episodeをLLMで要約し、セッション/日次要約を生成
  → 常時注入できる「さっき何してたっけ」の土台になる
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import exp as _exp
from pathlib import Path

logger = logging.getLogger("memory")


@dataclass
class Episode:
    id: str
    timestamp: datetime
    event: str
    context: str
    emotion: str
    result: str
    importance: float
    tags: list[str] = field(default_factory=list)
    source: str = "autonomous"
    driving_drive: str = ""
    # —— 人間らしい想起のためのメタデータ（Stage1〜2）——
    poignancy: float = 0.5          # 重要度 0〜1（LLM評定、堅牢性のため0..1に正規化）
    strength: float = 1.0           # 想起頻度で強化される「記憶の強さ」
    last_recall: datetime | None = None   # 最後に想起された時刻（Ebbinghaus強化）
    entities: list[str] = field(default_factory=list)  # 登場エンティティ（Stage2連想）


# 「さっき」「最近」等の時間的文脈語を含む質問向けの、記憶検索に使う時刻の半減期（秒）
RECENCY_HALFLIFE = 6 * 3600  # 6時間（時間的文脈クエリ用）

#  人間らしい想起のスコア重み（Generative Agents式）:
#  relevance と poignancy を recency より大きくする → 古くても重要/関連ある記憶は生き残る
W_RECENCY = 0.5
W_RELEVANCE = 3.0
W_POIGNANCY = 2.0

# 時間的文脈語を含む質問は、単純なキーワード一致より「新しさ」を強く重視する
TIME_CONTEXT_WORDS = [
    "さっき", "先ほど", "直近", "最近", "今日", "昨日", "今日一日",
    "今朝", "さっきまで", "ついさっき", "この前", "前に", "先日",
    "何してた", "何してたっけ", "何をしてた", "どうしてた",
]


class Memory:
    def __init__(self, path: str = "data/episodes", summary_path: str = "data/summaries"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.summary_path = Path(summary_path)
        self.summary_path.mkdir(parents=True, exist_ok=True)
        self.episodes: list[Episode] = []
        self._load()

    # --- 検索（人間らしい想起: 忘却曲線 × 関連性 × 重要度）---

    def search(self, query: str, top_k: int = 5) -> list[Episode]:
        """多チャネル検索を、人間らしい想起ループに統合。

        スコア = w_recency·recency + w_relevance·relevance + w_poignancy·poignancy
        加えて、回想された記憶は strength を強化する（思い出すほど忘れにくい）。
        """
        if not self.episodes:
            return []

        q = query.strip().lower()
        is_time_context = any(w in query for w in TIME_CONTEXT_WORDS)

        if not q:
            results = self._recent(top_k)
            self._reinforce(results)
            return results

        # relevance: キーワード/n-gram 類似度（0..1）+ entityオーバーラップの連想ブースト
        q_entities = self._extract_query_entities(query)
        scored = []
        for ep in self.episodes:
            rel = self._relevance(q, ep)
            score = (
                W_RECENCY * self._recency(ep)
                + W_RELEVANCE * rel
                + W_POIGNANCY * ep.poignancy
            )
            # 連想ブースト: クエリに含まれるエンティティが記憶にもあれば加算
            if q_entities and set(q_entities) & set(ep.entities):
                score += W_RELEVANCE * 0.5
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 時間的文脈クエリは新しさを再加味
        if is_time_context:
            scored.sort(key=lambda x: (
                x[1].timestamp.timestamp(),
                x[0],
            ), reverse=True)

        results = [ep for _, ep in scored[:top_k]]
        # ほぼ全員がスコア≒0（関連性もない完全なミス）なら直近にフォールバック
        if max((s for s, _ in scored if scored), default=0) < 0.05 and not is_time_context:
            results = self._recent(top_k)

        self._reinforce(results)
        return results

    def _reinforce(self, episodes: list[Episode]):
        """想起された記憶を強化: strength += 1, last_recall 更新（MemoryBank式）"""
        now = datetime.now()
        for ep in episodes:
            ep.strength += 1.0
            ep.last_recall = now
        self._flush(episodes)

    def _relevance(self, q: str, ep: Episode) -> float:
        """関連性: n-gram類似度 + キーワード一致で 0..1"""
        text = self._text(ep)
        if not q or not text:
            return 0.0
        if q in text.lower():
            return 1.0
        return self._ngram_sim(q, text.lower())

    def _recency(self, ep: Episode) -> float:
        """新しさ: 半減期モデル（0..1）。last_recallがあればそこからの経過で再活性化"""
        base = ep.last_recall if ep.last_recall else ep.timestamp
        age = max(0.0, (datetime.now() - base).total_seconds())
        return 2 ** (-age / RECENCY_HALFLIFE)

    @staticmethod
    def _text(ep: Episode) -> str:
        return f"{ep.event} {ep.context} {ep.result} {' '.join(ep.tags)}"

    def _recent(self, top_k: int) -> list[Episode]:
        return sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:top_k]

    # --- 階層: 日次/セッション要約（Semantic層の統合）---

    def recent_episodes(self, n: int = 10) -> list[Episode]:
        """直近n件のepisodeを新しい順に返す（統合の入力用）"""
        return self._recent(n)

    def load_day_summary(self, date: str) -> str:
        """指定日の要約を読む（無ければ空文字）"""
        fp = self.summary_path / f"{date}.txt"
        if fp.exists():
            try:
                return fp.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Failed to read summary {fp}: {e}")
        return ""

    def save_day_summary(self, date: str, summary: str):
        """指定日の要約を保存"""
        fp = self.summary_path / f"{date}.txt"
        try:
            fp.write_text(summary.strip(), encoding="utf-8")
            logger.info(f"Day summary saved: {fp}")
        except Exception as e:
            logger.error(f"Failed to save summary {fp}: {e}")

    def latest_summary(self) -> str:
        """最新の日次要約（このセッションに跨って「何をしてきたか」の土台）"""
        for d in self._summary_dates_desc():
            s = self.load_day_summary(d)
            if s:
                return f"[{d} の記録]\n{s}"
        return ""

    def _summary_dates_desc(self) -> list[str]:
        return sorted(
            (fp.stem for fp in self.summary_path.glob("*.txt")),
            reverse=True,
        )

    def save(self, ep: Episode) -> str:
        """Save episode to disk."""
        if not ep.id:
            ep.id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.episodes.append(ep)
        self._write(ep)
        return ep.id

    def _write(self, ep: Episode):
        """単一episodeをディスクへ書き出す"""
        fp = self.path / f"{ep.id}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({
                "id": ep.id,
                "timestamp": ep.timestamp.isoformat(),
                "event": ep.event,
                "context": ep.context,
                "emotion": ep.emotion,
                "result": ep.result,
                "importance": ep.importance,
                "poignancy": ep.poignancy,
                "strength": ep.strength,
                "last_recall": ep.last_recall.isoformat() if ep.last_recall else None,
                "entities": ep.entities,
                "tags": ep.tags,
                "source": ep.source,
                "driving_drive": ep.driving_drive,
            }, f, ensure_ascii=False, indent=2)

    def _flush(self, episodes: list[Episode]):
        """複数episodeの変更をディスクへ反映"""
        for ep in episodes:
            if ep.id and any(x.id == ep.id for x in self.episodes):
                self._write(ep)

    def _load(self):
        """Load all episodes from disk."""
        for fp in self.path.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                last_recall = d.get("last_recall")
                self.episodes.append(Episode(
                    id=d["id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    event=d.get("event", ""),
                    context=d.get("context", ""),
                    emotion=d.get("emotion", ""),
                    result=d.get("result", ""),
                    importance=d.get("importance", 0.5),
                    poignancy=d.get("poignancy", d.get("importance", 0.5)),
                    strength=d.get("strength", 1.0),
                    last_recall=datetime.fromisoformat(last_recall) if last_recall else None,
                    entities=d.get("entities", []),
                    tags=d.get("tags", []),
                    source=d.get("source", "autonomous"),
                    driving_drive=d.get("driving_drive", ""),
                ))
            except Exception as e:
                logger.warning(f"Failed to load {fp}: {e}")
        logger.info(f"Loaded {len(self.episodes)} episodes")

    @staticmethod
    def _ngram_sim(a: str, b: str, n: int = 2) -> float:
        """Calculate n-gram similarity between two strings."""
        if not a or not b:
            return 0.0
        agrams = {a[i:i+n] for i in range(len(a) - n + 1)}
        bgrams = {b[i:i+n] for i in range(len(b) - n + 1)}
        if not agrams or not bgrams:
            return 0.0
        return len(agrams & bgrams) / (len(agrams) * len(bgrams)) ** 0.5

    @staticmethod
    def _extract_query_entities(query: str) -> list[str]:
        """クエリ中の既知の固有名詞・エンティティらしき語を軽く抽出（連想用）。

        完全なNERはせず、既存episodeのentitiesと突き合わせるために
        クエリに含まれる 2文字以上の名詞句をざっくり拾う。
        """
        if not query:
            return []
        import re as _re
        # 「"(句)"」「『句』」や、長いカタカナ・漢字連続を粗く拾う
        quoted = _re.findall(r'["「『]([^"」』]{2,})["」』]', query)
        words = quoted
        if not words:
            # 3文字以上のカタカナ語（固有名詞候補）
            words += _re.findall(r'[ァ-ヶー]{2,}', query)
        return [w for w in words if len(w) >= 2][:8]

    # --- 忘却（MemoryBank: Ebbinghaus忘却曲線 + 確率的忘却）---

    def forget(self, now: datetime | None = None, grace_days: float = 2.0) -> int:
        """人間らしい忘却: 忘却曲線に従い弱い記憶を確率的に忘れる。

        retention = exp(-経過日数 / (5 * strength))
        想起回数が多い(strength大)ほど残りやすい。
        grace_days: この日数より新しい記憶は一切忘れない（新規データ保護）。
        戻り値: 消去したepisode数
        """
        now = now or datetime.now()
        forgotten = 0
        alive = []
        for ep in self.episodes:
            base = ep.last_recall or ep.timestamp
            days = max(0.0, (now - base).total_seconds() / 86400.0)
            # 覚えている確率（Ebbinghaus）: strengthが高いほど長持ち
            retention = _exp(-days / (5.0 * ep.strength))
            # 極めて重要な記憶は忘れない（信念コア）
            kept_prob = 1.0 if (ep.importance >= 0.9 or ep.poignancy >= 0.9) else max(retention, 0.05)
            # 新しすぎる記憶は確実に保持（未定着データの消失防止）
            if days < grace_days:
                kept_prob = 1.0
            import random as _r
            if _r.random() <= kept_prob:
                alive.append(ep)
            else:
                fp = self.path / f"{ep.id}.json"
                try:
                    fp.unlink(missing_ok=True)
                except Exception:
                    pass
                forgotten += 1
        if forgotten:
            logger.info(f"Forgot {forgotten} memories")
        self.episodes = alive
        return forgotten

    # --- Stage2: 連想記憶（エンティティで結ぶ関連想起）---

    def search_by_entity(self, entity: str, top_k: int = 3) -> list[Episode]:
        """指定エンティティ（人・物・場所）を持つ記憶を連想で返す。
        Generative Agents の連想メモリ・Graphiti の entity→episode 想起の簡略版。
        """
        if not entity:
            return []
        hits = [
            ep for ep in self.episodes
            if entity in ep.entities
        ]
        hits.sort(key=lambda e: (e.timestamp.timestamp(), e.strength), reverse=True)
        return hits[:top_k]

    def related_by_entity(self, entities: list[str], exclude_id: str | None = None, top_k: int = 3) -> list[Episode]:
        """エンティティ共有で関連する記憶を返す（連想想起）。
        exclude_id: 現在のepisode自身を除くためのID
        """
        if not entities:
            return []
        scored = []
        for ep in self.episodes:
            if exclude_id and ep.id == exclude_id:
                continue
            overlap = set(entities) & set(ep.entities)
            if overlap:
                scored.append((len(overlap), ep.strength, ep.timestamp.timestamp(), ep))
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [s[3] for s in scored[:top_k]]

    # --- Stage3: 反射（洞察）メモリ ---

    def save_reflection(self, text: str, evidence_ids: list[str], timestamp: datetime | None = None) -> str:
        """反射(インサイト)を高重要度の記憶として保存。
        Generative Agents の generate_insights_and_evidence 相当。
        """
        ep = Episode(
            id="",
            timestamp=timestamp or datetime.now(),
            event=f"【洞察】{text}",
            context="自己反省",
            emotion="内省",
            result="",
            importance=0.8,
            poignancy=0.8,
            entities=[],
            tags=["洞察", "反省"],
            source="reflection",
        )
        # 洞察は裏付けるepisodeを参照として記録（可逆・証跡保持）
        ep.tags += ["根拠:" + eid for eid in evidence_ids]
        return self.save(ep)

    # --- バッチ注釈（poignancy / エンティティ/ 強さ更新）---

    def unannotated(self, limit: int = 20) -> list[Episode]:
        """poignancy/entities が未注釈のepisodeを返す（統合バッチで注釈）"""
        out = []
        for ep in sorted(self.episodes, key=lambda e: e.timestamp.timestamp(), reverse=True):
            if not ep.entities and ep.strength <= 1.0:
                out.append(ep)
            if len(out) >= limit:
                break
        return out

    def set_annotation(self, ep: Episode, poignancy: float | None = None, entities: list[str] | None = None):
        """episodeへ注釈を設定して保存"""
        if poignancy is not None:
            ep.poignancy = max(0.0, min(1.0, poignancy))
        if entities is not None:
            ep.entities = list(dict.fromkeys(entities))
        self._write(ep)
