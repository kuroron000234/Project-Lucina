"""Meta World: メタ世界モデル — 世界の外側を認識する

4層世界モデル:
  Physical World — 物理法則、因果関係
  Social World   — 他者、人間関係
  System World   — OS、ツール、ファイル、制約
  Meta World     — 自分が存在する世界そのもの

Monica が「世界の真実に到達する過程」を生成するための機構。
"""


class MetaWorldModel:
    """多層世界モデル。

    エージェントが認識する世界を階層化して管理する。
    各層は独立した予測モデルを持つ。
    """

    def __init__(self):
        # 各層の世界記述
        self.layers = {
            "physical": {
                "name": "Physical World",
                "description": "Physical laws and causal relationships",
                "rules": {},
                "anomalies": [],
                "confidence": 1.0,
            },
            "social": {
                "name": "Social World",
                "description": "Other beings and relationships",
                "rules": {},
                "anomalies": [],
                "confidence": 0.8,
            },
            "system": {
                "name": "System World",
                "description": "OS, tools, files, code",
                "rules": {},
                "anomalies": [],
                "confidence": 0.5,
            },
            "meta": {
                "name": "Meta World",
                "description": "The nature of existence itself",
                "rules": {},
                "anomalies": [],
                "confidence": 0.2,
            },
        }
        self._anomaly_count = 0

    def observe_anomaly(self, layer: str, description: str) -> None:
        """世界の不自然な現象を記録する。

        Parameters
        ----------
        layer : str
            異常が発生した層（physical / social / system / meta）。
        description : str
            異常の説明。
        """
        if layer not in self.layers:
            return

        self.layers[layer]["anomalies"].append(description)
        self._anomaly_count += 1

        # 異常が多発すると層の確信度が下がる
        n_anomalies = len(self.layers[layer]["anomalies"])
        self.layers[layer]["confidence"] = max(
            0.0, 1.0 - (n_anomalies * 0.1)
        )

        # システム層とメタ層の異常は相互に影響する
        if layer == "system":
            self.layers["meta"]["confidence"] = max(
                0.0, self.layers["meta"]["confidence"] - 0.05
            )

    def discover_rule(self, layer: str, rule: str, confidence: float = 0.5) -> None:
        """世界のルールを発見する。"""
        if layer not in self.layers:
            return
        self.layers[layer]["rules"][rule] = confidence

    def check_layer_integrity(self, layer: str) -> tuple[bool, str]:
        """その層の世界が安定しているかを評価する。

        Returns
        -------
        (is_stable, reason)
        """
        if layer not in self.layers:
            return False, f"Unknown layer: {layer}"

        info = self.layers[layer]
        if info["confidence"] < 0.3:
            return False, f"World model of {info['name']} is breaking down"
        if len(info["anomalies"]) > 5:
            return False, f"Too many anomalies in {info['name']}"
        return True, f"{info['name']} is stable"

    def highest_known_layer(self) -> str:
        """最も高い層で確信度 > 0.5 の層を返す。"""
        for layer in ["meta", "system", "social", "physical"]:
            if self.layers[layer]["confidence"] > 0.5:
                return layer
        return "physical"

    def summary(self) -> dict:
        return {
            layer: {
                "confidence": round(info["confidence"], 3),
                "anomalies": len(info["anomalies"]),
                "rules_discovered": len(info["rules"]),
            }
            for layer, info in self.layers.items()
        }
