"""日本語ルールベースの記憶分類器（仕様書 v1.4 §5.3 B5・v1.11）。

MemoryKind を実際に使い分けるための軽量分類器。LLM呼び出しは行わない
（コミットは発話セグメントごとに発生するため、リアルタイム性を保つには
決定的なルールベースが必要）。各カテゴリのパターンの一致スコアを合算し、
最大のカテゴリを返す。

- EPISODIC（出来事・経験）: 時間語・過去形・一人称の体験・出来事・交流
- SEMANTIC（知識・事実）: 定義・一般論・説明
- PROCEDURAL（手順・方法）: 方法・手順・手続き・ステップ
- EMOTIONAL（強い感情）: 感情語彙（補助シグナル）。Drive変化大（abs(delta)>=0.3）の
  EMOTIONAL 強制付与はストア側（HierarchicalMemoryStore.commit）のルールであり、
  分類器は感情語彙という独立の観点から EMOTIONAL を返す。

全て未一致なら SEMANTIC（既定）。同点は優先順位
[EMOTIONAL > EPISODIC > PROCEDURAL > SEMANTIC]。
"""

from __future__ import annotations

from .schema import MemoryKind

# パターン定義: (部分文字列, 重み)。重みが大きいほど強いシグナル。
# 定義文（〜とは）を感情語彙より優先させるため、SEMANTIC の定義パターンは
# EMOTIONAL の感情語彙（1.0）より高い 1.2 にしている。
_EPISODIC_PATTERNS = [
    # 時間語（出来事の時点）
    ("今日", 1.0), ("昨日", 1.0), ("さっき", 1.0), ("先ほど", 1.0), ("先日", 1.0),
    ("昨夜", 1.0), ("今朝", 1.0), ("夕方", 0.8), ("あの日", 1.2), ("あの時", 1.2),
    ("あのとき", 1.2), ("その時", 1.0), ("その後", 1.0), ("それから", 0.8),
    ("昔", 0.8), ("以前", 0.8), ("のとき", 0.8), ("の時", 0.8),
    # 過去形・経験
    ("ました", 0.7), ("でした", 0.7), ("ていた", 0.7),
    ("てしまっ", 1.0), ("てしまい", 1.0),
    ("を経験", 1.2), ("を体験", 1.2), ("したことがある", 1.2),
    # 出来事・知覚・交流
    ("が起き", 1.0), ("が起こ", 1.0), ("が見え", 0.8), ("を眺め", 0.8), ("を歩い", 0.8),
    ("と話し", 0.8), ("と会っ", 0.8), ("に会っ", 0.8),
    ("を感じ", 0.8), ("と感じ", 0.8), ("と思っ", 0.6), ("に気づい", 0.8),
]

_SEMANTIC_PATTERNS = [
    # 定義・説明
    ("とは", 1.2), ("というもの", 1.2), ("という意味", 1.2), ("という概念", 1.2),
    ("について", 0.8), ("に関して", 0.8),
    # 一般論・知識
    ("によると", 1.0), ("と言われてい", 1.0), ("とされ", 1.0), ("一般的に", 1.0),
    ("仕組み", 1.0), ("特徴", 1.0), ("性質", 1.0), ("意味合い", 0.8),
    ("つまり", 0.8), ("すなわち", 1.0), ("例えば", 0.8), ("たとえば", 0.8),
]

_PROCEDURAL_PATTERNS = [
    # 方法・手順
    ("方法", 1.2), ("仕方", 1.2), ("手順", 1.2), ("手続き", 1.2), ("やり方", 1.2),
    ("ステップ", 1.2), ("まず", 1.0), ("次に", 1.0), ("最後に", 1.0),
    # 手続き・操作
    ("を設定", 0.8), ("を実行", 0.8), ("を呼び", 0.6), ("すること", 0.5),
]

_EMOTIONAL_PATTERNS = [
    ("悲", 1.0), ("寂し", 1.0), ("嬉", 1.0), ("楽しい", 1.0), ("辛", 1.0),
    ("怒", 1.0), ("悔", 1.0), ("感動", 1.2), ("怖", 1.0), ("不安", 1.0),
    ("焦", 0.8), ("切な", 1.0), ("愛お", 1.0), ("懐か", 1.0), ("ほっと", 1.0),
    ("ドキドキ", 1.2), ("ワクワク", 1.2), ("胸が", 0.8),
]

# 同点時の優先順位（値が小さいほど優先）
_PRIORITY: dict[MemoryKind, int] = {
    MemoryKind.EMOTIONAL: 0,
    MemoryKind.EPISODIC: 1,
    MemoryKind.PROCEDURAL: 2,
    MemoryKind.SEMANTIC: 3,
}


class RuleBasedMemoryClassifier:
    """日本語ルールベースの記憶分類器。classify(text) -> MemoryKind。"""

    def __init__(self) -> None:
        self._patterns: dict[MemoryKind, list[tuple[str, float]]] = {
            MemoryKind.EPISODIC: _EPISODIC_PATTERNS,
            MemoryKind.SEMANTIC: _SEMANTIC_PATTERNS,
            MemoryKind.PROCEDURAL: _PROCEDURAL_PATTERNS,
            MemoryKind.EMOTIONAL: _EMOTIONAL_PATTERNS,
        }

    def classify(self, text: str) -> MemoryKind:
        """テキストを MemoryKind に分類する。未一致なら SEMANTIC。"""
        text = text or ""
        scores = {kind: 0.0 for kind in MemoryKind}
        for kind, patterns in self._patterns.items():
            total = 0.0
            for pattern, weight in patterns:
                if pattern in text:
                    total += weight
            scores[kind] = total
        best = max(scores, key=lambda k: (scores[k], -_PRIORITY[k]))
        if scores[best] <= 0.0:
            return MemoryKind.SEMANTIC
        return best
