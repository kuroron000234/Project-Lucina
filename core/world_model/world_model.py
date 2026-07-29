"""
世界モデル層 (WorldModel)

責務: 環境の内部モデルを持ち、「状態＋行動 → 次の状態」を予測する。
人間でいう「こうしたらどうなるだろう」という内的シミュレーション。

Phase 2 Step 3: LLMシミュレーション（最小実装）
"""

import logging

from core.llm import LLMClient
from core.world_model.interface import (
    Prediction,
    WorldModelInput,
    WorldModelOutput,
)

logger = logging.getLogger("WorldModel")


class WorldModel:
    """
    世界モデル層: ある状態で特定の行動を取った結果を予測する。

    実装の進化パス:
    Phase 2 Step 1: LLMに直接「この状態でこの行動をするとどうなる？」と聞く
    Phase 2 Step 2: 過去エピソードからの統計予測を追加（将来）
    Phase 2 Step 3: 簡易ニューラルネット（FFN）で近似（将来）

    エッジケース:
    - 未知の状態: 確信度を低く設定し、デフォルト予測を返す
    - 矛盾する予測: 確率で重み付けして複数予測を保持
    - 計算コスト: シミュレーションは深さ3までに制限
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()
        self.statistics: dict[str, dict] = {}  # (state, action) -> 結果の統計

    def predict(self, input: WorldModelInput) -> WorldModelOutput:
        """
        現在の環境・駆動状態から結果を予測する。

        主駆動値を「検討中の行動」とみなし、LLMシミュレーションを試みる。
        失敗した場合はルールベースの予測を返す。
        """
        # 主駆動から候補行動を自動設定
        candidate_action = input.candidate_action or input.drive.primary_drive

        # LLMシミュレーションを試行
        try:
            prompt = self._build_prediction_prompt(input, candidate_action)
            system_prompt = (
                "あなたは世界モデルです。与えられた環境状態と行動から、"
                "起こりうる結果を複数予測し、それぞれに確率とリスク評価をつけてください。"
            )
            response = self.llm.chat(prompt, system_prompt=system_prompt)
            predictions = self._parse_predictions(response)
        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")
            predictions = []

        # LLM予測がない場合はルールベースで生成
        if not predictions:
            predictions = self._rule_based_predict(candidate_action, input.environment)

        return WorldModelOutput(predictions=predictions)

    def simulate(self, state, plan) -> list[Prediction]:
        """
        計画全体をシミュレーションし、各ステップの予測を返す。
        深さ3までに制限。

        Phase 2 では各ステップに対する簡易予測を生成。
        """
        predictions = []
        max_depth = min(len(plan.steps), 3)

        for step in plan.steps[:max_depth]:
            # 各ステップに対して簡易予測
            action = step.action
            next_state = self._estimate_step_outcome(step)
            risk = "low" if action in ["file_read", "file_list", "notify_user"] else "medium"

            predictions.append(Prediction(
                action=action,
                next_state=next_state,
                probability=0.7 if risk == "low" else 0.5,
                expected_reward=0.6 if risk == "low" else 0.3,
                risk_level=risk,
                reasoning=f"Step {step.order}: {step.description}",
            ))

        return predictions

    def update(self, actual: "Episode", prediction: Prediction):
        """
        実際の結果と予測の差を学習してモデルを更新する。
        Phase 2 では統計データの蓄積のみ。
        """
        key = (prediction.action, prediction.next_state[:50])
        if key not in self.statistics:
            self.statistics[key] = {"count": 0, "total_error": 0.0}

        self.statistics[key]["count"] += 1
        # 実際の結果との誤差は簡易的に計算（Phase 2 では精度より学習プロセス）
        logger.debug(f"WorldModel updated: {key}, count={self.statistics[key]['count']}")

    def confidence(self, state: str, action: str) -> float:
        """
        特定の状態-行動ペアに対する予測の確信度を返す。

        エッジケース:
        - 未知の状態: 低い確信度（0.3）を返す
        """
        key = (action, state[:50])
        stats = self.statistics.get(key)
        if stats and stats["count"] > 5:
            return min(1.0, stats["count"] / 20.0)
        return 0.3  # 未知の状態は低確信度

    def _build_prediction_prompt(self, input: WorldModelInput, candidate_action: str) -> str:
        """予測プロンプトを構築する。"""
        env = input.environment
        lines = [
            "## 現在の状態",
            f"- CPU: {env.system_state.cpu_percent}%",
            f"- メモリ: {env.system_state.memory_percent}%",
            f"- アクティブウィンドウ: {env.system_state.active_window or 'N/A'}",
            f"- ディレクトリ: {env.system_state.current_directory}",
            f"- ユーザー入力: {env.user_input or 'なし'}",
            f"- ファイル数: {len(env.files) if env.files else 0}",
            "",
            "## 駆動状態",
        ]
        for name, value in input.drive.drives.items():
            lines.append(f"- {name}: {value:.2f}")
        lines.append(f"主駆動: {input.drive.primary_drive}")
        lines.append("")
        lines.append("## アクティブゴール")
        lines.append(input.active_goal)
        lines.append("")
        lines.append("## 検討中の行動")
        lines.append(candidate_action)
        lines.append("")
        lines.append("## 指示")
        lines.append("この行動を取った場合の結果を2〜3通り予測してください。")
        lines.append("各予測に確率とリスク評価をつけてください。")
        lines.append("")
        lines.append("出力形式 (YAML風):")
        lines.append("- action: <行動>")
        lines.append("  next_state: <予測される次の状態>")
        lines.append("  probability: <0.0〜1.0>")
        lines.append("  expected_reward: <-1.0〜1.0>")
        lines.append("  risk_level: low|medium|high")
        lines.append("  reasoning: <理由>")

        return "\n".join(lines)

    def _parse_predictions(self, response: str) -> list[Prediction]:
        """LLM応答から予測リストをパースする。"""
        predictions = []
        current_pred = {}

        for line in response.strip().split("\n"):
            stripped = line.strip()

            if stripped.startswith("- action:"):
                if current_pred and "action" in current_pred:
                    predictions.append(self._make_prediction(current_pred))
                current_pred = {"action": stripped.split(":", 1)[1].strip()}

            elif current_pred:
                if stripped.startswith("next_state:"):
                    current_pred["next_state"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("probability:"):
                    try:
                        current_pred["probability"] = float(stripped.split(":", 1)[1].strip())
                    except ValueError:
                        current_pred["probability"] = 0.5
                elif stripped.startswith("expected_reward:"):
                    try:
                        current_pred["expected_reward"] = float(stripped.split(":", 1)[1].strip())
                    except ValueError:
                        current_pred["expected_reward"] = 0.0
                elif stripped.startswith("risk_level:"):
                    current_pred["risk_level"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("reasoning:"):
                    current_pred["reasoning"] = stripped.split(":", 1)[1].strip()

        # 最後の予測を追加
        if current_pred and "action" in current_pred:
            predictions.append(self._make_prediction(current_pred))

        return predictions

    def _make_prediction(self, data: dict) -> Prediction:
        """辞書から Prediction オブジェクトを生成する。"""
        return Prediction(
            action=data.get("action", "unknown"),
            next_state=data.get("next_state", "変化なし"),
            probability=max(0.0, min(1.0, float(data.get("probability", 0.5)))),
            expected_reward=max(-1.0, min(1.0, float(data.get("expected_reward", 0.0)))),
            risk_level=data.get("risk_level", "low")
                if data.get("risk_level", "low") in ["low", "medium", "high"]
                else "low",
            reasoning=data.get("reasoning", ""),
        )

    def _rule_based_predict(self, action: str, env: "EnvironmentOutput") -> list[Prediction]:
        """
        ルールベースのデフォルト予測。
        LLM予測が使えない場合のフォールバック。
        """
        # 行動タイプに応じたデフォルト予測
        predictions = []

        if action == "exploration":
            predictions.append(Prediction(
                action="exploration",
                next_state="新しいファイルや情報が発見される。システム負荷は低い。",
                probability=0.7,
                expected_reward=0.5,
                risk_level="low",
                reasoning="ファイル探索は低リスクで新しい情報を得られる標準的な行動。",
            ))
            predictions.append(Prediction(
                action="exploration",
                next_state="特に新しいものは見つからず、時間を消費する。",
                probability=0.3,
                expected_reward=-0.1,
                risk_level="low",
                reasoning="環境に変化がない場合、探索の効果は限定的。",
            ))

        elif action == "rest":
            predictions.append(Prediction(
                action="rest",
                next_state="システム状態が安定する。CPU負荷が低下する。",
                probability=0.8,
                expected_reward=0.3,
                risk_level="low",
                reasoning="休息は常に安全で、システムリソースを節約できる。",
            ))

        elif action == "social":
            predictions.append(Prediction(
                action="social",
                next_state="ユーザーからの応答がある。インタラクションが発生する。",
                probability=0.5,
                expected_reward=0.4,
                risk_level="low",
                reasoning="ユーザーとの対話は中程度の確率で応答が得られる。",
            ))

        else:
            predictions.append(Prediction(
                action=action,
                next_state="行動が実行される。結果は状況による。",
                probability=0.5,
                expected_reward=0.0,
                risk_level="medium",
                reasoning=f"未知の行動 {action} のため、結果は不確定。",
            ))

        return predictions

    def _estimate_step_outcome(self, step) -> str:
        """ステップの結果を簡易推定する。"""
        outcomes = {
            "file_read": "ファイルの内容が読み込まれる",
            "file_write": "ファイルに書き込まれる",
            "file_list": "ファイル一覧が表示される",
            "command_exec": "コマンドが実行される",
            "notify_user": "ユーザーに通知が送られる",
        }
        return outcomes.get(step.action, f"アクション {step.action} が実行される")
