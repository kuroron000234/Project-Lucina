"""DriveLogitsProcessor — Drive値→ロジットバイアス適用（仕様書 v1.4 §5.4）。

⑤ 複数トークン語彙へのバイアス適用方式:
    - build_vocab_map は {drive: [[token_id, ...], ...]} を返す（1語=複数トークン列）。
    - 各トークン列について**先頭トークンにのみ**バイアスを加算し、後続トークンには適用しない。
      理由: ロジットバイアスは「次の1トークン」の確率にしか作用できない。列内全トークンへ
      均等にバイアスをかけると、文脈上不自然な位置でも語を無理に完成させようとする挙動につながる。
    - 同じ先頭トークンIDが複数のDriveの語彙で重複する場合、バイアスは該当する全Driveの値の合算とする。
      二重カウントを避けるため、先頭トークンIDごとに一度だけ合算してからロジットへ加算する。
"""

from __future__ import annotations

import numpy as np

VocabMap = dict[str, list[list[int]]]


class DriveLogitsProcessor:
    def __init__(self, coefficient: float):
        self.coefficient = float(coefficient)

    def apply(self, logits: np.ndarray, drive_state: dict, vocab_map: VocabMap) -> np.ndarray:
        """ロジット配列へDriveバイアスを適用して返す（入力は変更しない）。"""
        out = np.asarray(logits, dtype=np.float64).copy()
        n = out.shape[0]
        # 先頭トークンIDごとに一度だけ合算（⑤: 重複Driveの合算・二重カウント防止）
        per_first_token: dict[int, float] = {}
        for drive, sequences in vocab_map.items():
            value = float(drive_state.get(drive, 0.0))
            if value <= 0.0 or self.coefficient == 0.0:
                continue
            bias = self.coefficient * value
            for seq in sequences:
                if not seq:
                    continue
                first = int(seq[0])
                if 0 <= first < n:
                    per_first_token[first] = per_first_token.get(first, 0.0) + bias
        for token_id, bias in per_first_token.items():
            out[token_id] += bias
        return out
