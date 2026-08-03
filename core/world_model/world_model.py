"""
世界モデル層 (WorldModel)

責務: 環境の内部モデルを持ち、「状態＋行動 → 次の状態」を予測する。
人間でいう「こうしたらどうなるだろう」という内的シミュレーション。

Phase 2 Step 3: LLMシミュレーション（最小実装）
"""

import logging
import math

import config
from core.llm import LLMClient
from core.world_model.interface import (
    ImaginedFuture,
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

        # LLMシミュレーションを試行（use_llm=False ならルールベースのみ）
        predictions = []
        if input.use_llm:
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

    def imagine(self, input: WorldModelInput) -> list[ImaginedFuture]:
        """
        v4.0: 未来の候補を想像する（「もし◯◯したらどうなる？」）。

        人格層が自分の好みと照らして行動を選ぶための材料を生成する。
        アクティブ推論における「好ましい未来の事前分布」に相当。
        LLMで生成を試み、失敗時は駆動からルールベースの候補を返す。
        use_llm=False（tier2）では LLM コストを払わずルールベースのみ。
        """
        if not input.use_llm:
            return self._rule_based_imagine(input)
        try:
            prompt = self._build_imagine_prompt(input)
            system_prompt = (
                "あなたは想像力豊かなエージェントです。\n"
                "自分の性格・記憶・駆動から、『やってみたいこと』の候補を具体的に想像してください。\n"
                "実現可能で具体的なものにしてください。\n"
                "出力形式 (YAML風):\n"
                "- action: <やってみたいこと>\n"
                "  next_state: <その結果どうなるか>\n"
                "  preference: <どれだけ望むか 0.0-1.0>\n"
                "  reasoning: <なぜそれを望むか>"
            )
            response = self.llm.chat(prompt, system_prompt=system_prompt)
            imagined = self._parse_imagined_futures(response)
            if imagined:
                return imagined
        except Exception as e:
            logger.warning(f"LLM imagination failed: {e}")
        return self._rule_based_imagine(input)

    def _build_imagine_prompt(self, input: WorldModelInput) -> str:
        """想像プロンプトを構築する。"""
        env = input.environment
        lines = [
            "## 現在の状態",
            f"- CPU: {env.system_state.cpu_percent}%",
            f"- メモリ: {env.system_state.memory_percent}%",
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
        lines.append("## 指示")
        lines.append("あなたが今『やってみたいこと』の候補を"
                     f"{config.WILL_CONFIG.get('imagination_count', 3)}つ想像してください。")
        lines.append("抽象的な理想ではなく、実際に行動に移せる具体性のあるものにしてください。")
        lines.append("各候補に、どれだけそれを望むか (preference) を数値でつけてください。")
        return "\n".join(lines)

    def _parse_imagined_futures(self, response: str) -> list[ImaginedFuture]:
        """LLM応答から ImaginedFuture リストをパースする。"""
        futures = []
        current = {}
        for line in response.strip().split("\n"):
            stripped = line.strip()
            if stripped.startswith("- action:"):
                if current and "action" in current:
                    futures.append(self._make_imagined(current))
                current = {"action": stripped.split(":", 1)[1].strip()}
            elif current:
                if stripped.startswith("next_state:"):
                    current["next_state"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("preference:"):
                    try:
                        current["preference"] = float(stripped.split(":", 1)[1].strip())
                    except ValueError:
                        current["preference"] = 0.5
                elif stripped.startswith("reasoning:"):
                    current["reasoning"] = stripped.split(":", 1)[1].strip()
        if current and "action" in current:
            futures.append(self._make_imagined(current))
        return [f for f in futures if f.action]

    def _make_imagined(self, data: dict) -> ImaginedFuture:
        """辞書から ImaginedFuture を生成する。"""
        return ImaginedFuture(
            action=data.get("action", ""),
            next_state=data.get("next_state", ""),
            preference=max(0.0, min(1.0, float(data.get("preference", 0.5)))),
            reasoning=data.get("reasoning", ""),
        )

    def _rule_based_imagine(self, input: WorldModelInput) -> list[ImaginedFuture]:
        """
        ルールベースの想像候補。
        LLMが使えない場合のフォールバック。駆動値に応じて候補を生成する。
        """
        drives = input.drive.drives
        primary = input.drive.primary_drive
        templates = {
            "exploration": (
                "ワークスペースの未踏ファイルを探索する",
                "新しい発見があり、知識が広がる",
            ),
            "social": (
                "ユーザーと対話して最近の出来事を共有する",
                "ユーザーとの関係が深まる",
            ),
            "achievement": (
                "コードベースに小さな改善を加える",
                "達成感が得られ、システムが良くなる",
            ),
            "rest": (
                "システム状態を静かに監視する",
                "エネルギーを節約し、状態が安定する",
            ),
            "maintenance": (
                "ログと記憶を整理する",
                "整理され、次の行動の準備ができる",
            ),
        }
        futures = []
        ordered = sorted(drives.items(), key=lambda x: -x[1])
        for name, value in ordered[:3]:
            if name in templates:
                action, next_state = templates[name]
                futures.append(ImaginedFuture(
                    action=action,
                    next_state=next_state,
                    preference=max(0.3, min(0.9, value)),
                    reasoning=f"{name}駆動が {value:.2f} と高いため",
                ))
        if not futures:
            futures.append(ImaginedFuture(
                action="現在の状態を維持する",
                next_state="安定が続く",
                preference=0.5,
                reasoning="特に強い駆動がないため",
            ))
        return futures

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

    @staticmethod
    def normalize_surprise(surprise: float) -> float:
        """
        v5.0: サプライズ値を 0.0〜1.0 に正規化する（単調写像 s/(1+s)）。
        """
        s = max(0.0, float(surprise))
        return s / (1.0 + s)

    def compute_surprise(self, actual_reward: float, prediction: Prediction) -> float:
        """
        v5.0: 観測後のサプライズ（実測された予測誤差）を計算する。

        FEPにおける負の対数尤度 −ln p(x | prediction) のガウス近似:
            S = (x − μ)² / σ² + ln σ
        - μ = expected_reward（-1..1 を 0..1 に変換）
        - x = 実際の評価値（0..1）
        - σ = prediction.uncertainty（SURPRISE_CONFIG.sigma_floor で下限クランプ）

        予測が当たっていれば S → 0（正確）、外れれば大きくなる（不確実）。
        高サプライズ = 学ぶべき・探索すべき時（能動的推論のエピステミック価値）。

        注意: max(0, ·) のクランプにより、|x−μ| が小さい誤差は0になる
        （デッドゾーン）。σ=0.3 では |x−μ| ≲ 0.33 の誤差は黙殺される。
        """
        mu = max(0.0, min(1.0, (prediction.expected_reward + 1.0) / 2.0))
        sigma = max(
            config.SURPRISE_CONFIG["sigma_floor"],
            getattr(prediction, "uncertainty", config.SURPRISE_CONFIG["default_sigma"]),
        )
        x = max(0.0, min(1.0, float(actual_reward)))
        surprise = ((x - mu) ** 2) / (sigma ** 2) + math.log(sigma)
        return max(0.0, surprise)

    def update(self, actual: "Episode", prediction: Prediction,
               actual_overall: float | None = None):
        """
        実際の結果と予測の差を学習してモデルを更新する。

        v3.2: 実誤差項を記録する。
        キーは (action, risk_level) で統計を集約（フリーテキストの
        next_state[:50] による断片化を解消）。
        誤差 = |prediction.expected_reward − actual_overall|。
        """
        key = (prediction.action, prediction.risk_level)
        if key not in self.statistics:
            self.statistics[key] = {"count": 0, "total_error": 0.0}

        # 実誤差の記録（actual_overall が無い場合は中立値0.5との差）
        actual_val = actual_overall if actual_overall is not None else 0.5
        error = abs(prediction.expected_reward - actual_val)
        self.statistics[key]["count"] += 1
        self.statistics[key]["total_error"] += error
        logger.debug(
            f"WorldModel updated: {key}, count={self.statistics[key]['count']}, "
            f"error={error:.3f}"
        )

    def confidence(self, state: str, action: str) -> float:
        """
        特定の状態-行動ペアに対する予測の確信度を返す。

        v3.2: 誤差項を反映。
        - update() の統計キーは (action, risk_level) なので、action 単位で集約する
        - 誤差が大きい（予測が外れている）ほど確信度が下がる
        - サンプル数が少ない（<3）は低い確信度（0.3）

        エッジケース:
        - 未知の状態: 低い確信度（0.3）を返す
        """
        relevant = [s for (a, _), s in self.statistics.items() if a == action]
        if not relevant:
            return 0.3  # 未知の状態は低確信度
        total_count = sum(s["count"] for s in relevant)
        total_error = sum(s["total_error"] for s in relevant)
        if total_count < 3:
            return 0.3
        sample_factor = min(1.0, total_count / 20.0)
        avg_error = total_error / total_count
        accuracy = max(0.1, 1.0 - avg_error)
        return sample_factor * accuracy

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
        lines.append("  uncertainty: <0.0〜1.0>  (予測の不確実性。小さいほど確信)")
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
                elif stripped.startswith("uncertainty:"):
                    try:
                        current_pred["uncertainty"] = float(stripped.split(":", 1)[1].strip())
                    except ValueError:
                        pass
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
            uncertainty=max(
                config.SURPRISE_CONFIG["sigma_floor"],
                min(1.0, float(data.get("uncertainty", config.SURPRISE_CONFIG["default_sigma"]))),
            ),
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
