"""
VRChat 接続層 — モニカの「身体」と「視覚」を VRChat に繋ぐ

Neuro-sama の神経SDKに相当する「VRChat 用ハーネス」の薄いラッパ。
Python の `vrcpilot` ライブラリ（Windows / Linux 対応）を使って、
VRChat のデスクトップクライアントとの間で読み書きする。

3層構成のうち「知覚層」と「身体層」をこのモジュールが担う。
- 知覚層: VRchatVision — VRChat 画面をキャプチャし、シーン変化 / OCR を
  `Percept` として知覚ストリームへ流す。
- 身体層: VRchatBody   — OSC で発言（chatbox）・移動・表情を送る。

心（キャラ層 / g4-midnight-macaw-v2）はこのモジュールを介さず、
そのままの文章力を保つ。

注意: vrcpilot は X11 / XWayland セッション前提。Wayland ネイティブのみの
環境では focus 系だけ制限されるが、VRChat（Proton）は XWayland で動くため
キャプチャ・OCR・OSC・入力は利用できる。
"""

import logging
from datetime import datetime

from .perception import Percept

logger = logging.getLogger("vrchat")

# vrcpilot が無い環境（テスト・CI）では依存を遅延インポートして壊さない
try:
    import vrcpilot  # noqa: F401
    _VRC_AVAILABLE = True
except Exception as e:  # pragma: no cover - 環境依存
    _VRC_AVAILABLE = False
    _VRC_IMPORT_ERROR = e
    logger.warning("vrcpilot を読み込めません（VRChat接続は無効）: %s", e)


class VRchatBody:
    """身体層 — VRChat アバターへの動作出力（OSC）。

    モニカの「心」が決めたことを、VRChat のアバターとして実行する。
    - 発言: chatbox にメッセージ表示
    - 表情: avatar parameters でエモート
    - 移動: /input で前後左右・ジャンプ
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self._host = host
        self._port = port
        self._sender = None
        self._controller = None
        self.enabled = _VRC_AVAILABLE
        if self.enabled:
            try:
                self._sender = vrcpilot.OscSender(host=host, port=port)
                self._controller = self._sender.controller()
                logger.info("VRchatBody: OSC送信先 %s:%s を初期化", host, port)
            except Exception as e:  # pragma: no cover
                logger.error("VRchatBody 初期化失敗: %s", e)
                self.enabled = False

    def say(self, text: str, sfx: bool = True):
        """VRChat の chatbox に発言を表示する。"""
        if not self.enabled or self._controller is None:
            return False
        try:
            # VRChat の chatbox は最大144文字
            self._controller.chatbox(text[:140], send=True, sfx=sfx)
            logger.info("VRchatBody say: %s", text[:60])
            return True
        except Exception as e:
            logger.error("say failed: %s", e)
            return False

    def typing(self, active: bool = True):
        """入力中インジケータを出す（思考中に使うと人間らしい）。"""
        if not self.enabled or self._controller is None:
            return
        try:
            self._controller.typing(active)
        except Exception as e:
            logger.error("typing failed: %s", e)

    def emote(self, name: str, value):
        """アバターパラメータを送って表情・モーションを出す。"""
        if not self.enabled or self._sender is None:
            return False
        try:
            if isinstance(value, bool):
                self._sender.avatar_parameters().send_bool(name, value)
            elif isinstance(value, int):
                self._sender.avatar_parameters().send_int(name, value)
            else:
                self._sender.avatar_parameters().send_float(name, float(value))
            logger.info("VRchatBody emote: %s=%s", name, value)
            return True
        except Exception as e:
            logger.error("emote failed: %s", e)
            return False

    def move(self, forward: float = 0.0, right: float = 0.0, seconds: float = 0.2):
        """アバターを移動させる（-1..1 の軸入力）。"""
        if not self.enabled or self._controller is None:
            return False
        try:
            self._controller.vertical(float(forward))
            self._controller.horizontal(float(right))
            import time
            time.sleep(seconds)
            self._controller.vertical(0.0)
            self._controller.horizontal(0.0)
            logger.info("VRchatBody move: fwd=%s right=%s", forward, right)
            return True
        except Exception as e:
            logger.error("move failed: %s", e)
            return False


class VRchatVision:
    """知覚層センサー — VRChat 画面を「見て」Percept にする。

    `Perception.add_sensor()` に適合する `sense()` を持つ。
    vrcpilot のスクリーンショットを低頻度（interval 指定）で撮り、
    シーン変化（前フレームとの差分）を検知した時だけ局面解釈をする。

    - 常時: 軽量なフレーム差分で「何か変わった」を見張る（LLM不要・高速）
    - 変化時: クロップ画像や OCR 結果を短文化して Percept に
    """

    def __init__(
        self,
        interval: float = 5.0,
        change_threshold: float = 0.02,
        quality_fps: bool = True,
        name: str = "vrchat_vision",
    ):
        self.interval = interval
        self.change_threshold = change_threshold
        self.name = name
        self.enabled = _VRC_AVAILABLE
        self._last_capture = 0.0
        self._last_frame = None
        self._last_percept: Percept | None = None
        self._suppress_duplicate = 0
        self.logger = logger.getChild("vision")

    def available(self) -> bool:
        """VRChat が起動していてキャプチャ可能か（遅延確認）。"""
        if not self.enabled:
            return False
        try:
            return bool(vrcpilot.find_pids())
        except Exception:
            return False

    def _capture(self):
        """VRChat スクリーンショットを取得（numpy 配列）。失敗時 None。"""
        try:
            shot = vrcpilot.take_screenshot()
            if shot is None:
                return None
            return shot
        except Exception as e:
            self.logger.warning("capture failed: %s", e)
            return None

    def _frame_diff_ratio(self, cur, ref) -> float:
        """2フレーム間の変化量（0..1）を軽量計算する。"""
        import numpy as np
        try:
            c = np.asarray(cur, dtype=np.float32)
            r = np.asarray(ref, dtype=np.float32)
            if c.shape != r.shape:
                return 1.0
            # 小さい縮小で比較して軽量化
            if c.size > 0:
                scale = max(1, c.shape[0] // 120)
                c = c[::scale, ::scale]
                r = r[::scale, ::scale]
                diff = np.abs(c - r).mean() / 255.0
                return float(diff)
            return 0.0
        except Exception:
            return 0.0

    def _interpret_change(self, shot) -> Percept | None:
        """変化検知後の「局面解釈」。

        ここでは軽量テキスト（OCR）と、画像の平均色・明度から
        ざっくりした情景を作る。詳細なLLM視覚解釈は呼び出し側で重複抑制
        しながら、必要に応じて別途行う。
        """
        now = datetime.now()
        import numpy as np
        img = np.asarray(shot.image)
        # 明度・彩度・平均色（情景の雰囲気）
        h, w = img.shape[:2] if img.ndim == 3 else (0, 0)
        gray = img.mean(axis=2) if img.ndim == 3 else img
        brightness = float(gray.mean()) / 255.0

        # OCR で読み取れた文字（UI・掲示・チャットなど）
        text_hint = ""
        try:
            ocr = vrcpilot.ocr(shot)
            words = [str(w.text) for w in getattr(ocr, "words", [])]
            words = [t for t in words if t.strip()]
            if words:
                text_hint = "。読める文字: " + "・".join(words[:12])
        except Exception as e:
            self.logger.debug("ocr failed (通常のワールド画面では仕方ない): %s", e)

        scene_desc = self._describe_scene(brightness, img)
        percept = Percept(
            source="environment",
            kind="scene",
            text=(
                f"今見えているVRChatの世界: {scene_desc}（明度{brightness:.2f}"
                f", {w}x{h}）{text_hint}"
            ),
            timestamp=now,
            importance=0.5 if text_hint else 0.3,
        )
        return percept

    def _describe_scene(self, brightness: float, img) -> str:
        """明度などから情景のふんいきを短文で表現する。"""
        if brightness < 0.2:
            return "暗い場所"
        if brightness > 0.8:
            return "明るい場所"
        return "ふつうの明るさの場所"

    def sense(self, now=None, state=None, memory=None):
        """Perception から呼ばれる。変化を検知した時だけ Percept を返す。

        毎呼び出しで LLM は使わない（リアルタイム性のための軽量常時ウォッチ）。
        変化を検知したら局面解釈（OCR + 明度）で Percept を作り、
        直近と重複したら抑制する。
        """
        now = now or datetime.now()
        import time
        if not self.enabled:
            return []
        # 間隔制御（呼ばれる度に撮らない）
        if time.time() - self._last_capture < self.interval:
            return []
        self._last_capture = time.time()

        shot = self._capture()
        if shot is None:
            return []

        # 常時: フレーム差分で変化を見張る
        if self._last_frame is not None:
            diff = self._frame_diff_ratio(shot.image, self._last_frame)
        else:
            diff = 1.0  # 初回は変化ありとみなす
        self._last_frame = shot.image.copy() if hasattr(shot.image, "copy") else shot.image

        if diff < self.change_threshold:
            # 変化なし → 知覚しない（静かな時間）
            return []

        percept = self._interpret_change(shot)

        # 近い内容の重複抑制（同じ情景を連呼しない）
        if self._last_percept and self._approx_equal(self._last_percept, percept):
            self._suppress_duplicate += 1
            if self._suppress_duplicate < 5:
                return []
            self._suppress_duplicate = 0

        self._last_percept = percept
        logger.info("VRchatVision: 変化検知 -> %s", percept.text[:60])
        return [percept]

    @staticmethod
    def _approx_equal(a: Percept, b: Percept) -> bool:
        """最近の同じ情景かを（ラフに）判定する。"""
        return a.text[:30] == b.text[:30]
