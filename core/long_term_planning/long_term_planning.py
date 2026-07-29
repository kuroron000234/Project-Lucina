"""
長期行動計画層 (LongTermPlanning)

責務: 長期目標・ルーティン・アイデンティティを管理する。
単発の行動ではなく、数日〜数週間単位の一貫性を保証する。

Phase 2 Step 4: ファイルベースの目標管理 + LLM振り返り
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from core.llm import LLMClient
from core.long_term_planning.interface import (
    LongTermPlanningInput,
    LongTermPlanningOutput,
    Routine,
)

logger = logging.getLogger("LongTermPlanning")


class LongTermPlanning:
    """
    長期行動計画層: 長期目標・ルーティン・アイデンティティを管理する。

    エッジケース:
    - 目標が大きすぎる: サブゴールに分割（Phase 2 では簡易版）
    - ルーティンと単発タスクの衝突: 優先度で判断（人格層に委譲）
    - 目標未設定: 自動生成
    """

    def __init__(self, llm_client: LLMClient | None = None, storage_path: str = "data/"):
        self.llm = llm_client or LLMClient()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.goals: list[dict] = []
        self.routines: list[Routine] = []
        self.identity_policy: str = ""
        self.focus_area: str = ""
        self.last_plan_update: datetime | None = None

        self._load()

    def plan(self, input: LongTermPlanningInput) -> LongTermPlanningOutput:
        """
        長期計画を生成・更新する。定期的に呼ばれる。

        LLM振り返りを試行し、失敗した場合はルールベースで計画を更新。
        """
        # LLMによる振り返りと計画生成を試行
        try:
            prompt = self._build_planning_prompt(input)
            system_prompt = (
                "あなたは長期行動計画者です。評価履歴と人格状態に基づいて、"
                "長期的な目標・ルーティン・方針を決定してください。"
            )
            response = self.llm.chat(prompt, system_prompt=system_prompt)
            parsed = self._parse_plan_response(response)
        except Exception as e:
            logger.warning(f"LLM long-term planning failed: {e}")
            parsed = {}

        # パース結果を反映（マージ）
        self._merge_plan(parsed, input)

        # 重複目標を排除（LLMが同一目標を複数回追加する問題への対策）
        self._deduplicate_goals()
        self.last_plan_update = datetime.now()

        # 保存
        self._save()

        return LongTermPlanningOutput(
            long_term_goal=self.goals[0].get("goal", "システムの維持と改善") if self.goals else "システムの維持と改善",
            routines=self.routines,
            identity_policy=self.identity_policy or "信頼できるアシスタントであり続ける",
            focus_area=self.focus_area or "環境のモニタリングと最適化",
            reflection=parsed.get("reflection", "定期更新完了"),
        )

    def generate_routines(self, personality) -> list[Routine]:
        """
        人格に基づいてルーティンを提案する。

        personality の特性を反映したルーティンを生成する。
        好奇心が強い → 知識探索の頻度を上げる
        慎重 → メンテナンスの頻度を上げる
        """
        routines = [
            Routine(
                name="定期メンテナンス",
                action="システム状態の確認とログの整理",
                frequency="daily",
                last_executed=None,
                enabled=True,
            ),
            Routine(
                name="知識探索",
                action="ワークスペースの変更を確認する",
                frequency="daily",
                last_executed=None,
                enabled=True,
            ),
            Routine(
                name="長期振り返り",
                action="過去の活動を分析し、長期計画を更新する",
                frequency="weekly",
                last_executed=None,
                enabled=True,
            ),
        ]

        # 人格特性を反映
        if hasattr(personality, 'traits'):
            curiosity = personality.traits.get('curiosity', 0.5)
            if curiosity > 0.7:
                # 好奇心が強い: 知識探索を2回/日に
                routines[1].frequency = "daily"
                routines[1].name = "活発な知識探索"
            elif curiosity < 0.3:
                # 好奇心が低い: 知識探索を週1に
                routines[1].frequency = "weekly"

            caution = personality.traits.get('caution', 0.5)
            if caution > 0.7:
                # 慎重: メンテナンス強化
                routines.append(Routine(
                    name="詳細ヘルスチェック",
                    action="システムの各コンポーネントを詳細にチェックする",
                    frequency="daily",
                    last_executed=None,
                    enabled=True,
                ))

        return routines

    def review_period(self, days: int) -> str:
        """
        指定期間の活動を振り返り、洞察をテキストで返す。
        """
        cutoff = datetime.now() - timedelta(days=days)

        # この期間に更新された目標を確認
        recent_goals = [
            g for g in self.goals
            if g.get("updated_at") and datetime.fromisoformat(g["updated_at"]) > cutoff
        ]

        if not recent_goals:
            return f"過去{days}日間の活動記録はありません。"

        lines = [f"## 過去{days}日間の振り返り", ""]
        for g in recent_goals:
            lines.append(f"- 目標: {g.get('goal', '不明')}")
            lines.append(f"  進捗: {g.get('progress', 0):.0%}")
        lines.append("")
        lines.append(f"アクティブなルーティン: {sum(1 for r in self.routines if r.enabled)}件")

        return "\n".join(lines)

    def update_goal_progress(self, goal: str, progress: float):
        """
        長期目標の進捗を更新する。
        同一目標が既存の場合は更新、なければ新規追加（重複防止）。
        """
        for g in self.goals:
            if g["goal"] == goal:
                g["progress"] = max(0.0, min(1.0, progress))
                g["updated_at"] = datetime.now().isoformat()
                self._save()
                logger.info(f"Goal progress updated: {goal[:30]} -> {progress:.0%}")
                return

        # 新しい目標として追加（重複チェック済み）
        self.goals.append({
            "goal": goal,
            "progress": max(0.0, min(1.0, progress)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })
        self._save()

    def _build_planning_prompt(self, input: LongTermPlanningInput) -> str:
        """長期計画プロンプトを構築する。"""
        # 評価履歴の要約
        if input.evaluation_history:
            scores = [s.overall for s in input.evaluation_history[-7:]]
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0.5

        lines = [
            f"## 直近の平均評価: {avg_score:.2f}",
            "",
            f"## 人格状態",
            f"名前: {input.personality_state.name}",
            f"特性: {input.personality_state.traits}",
            f"ムード: {input.personality_state.mood}",
            "",
            f"## 最近の活動",
            f"{input.recent_episodes_summary[:200]}",
            "",
            f"## 現在の目標",
        ]
        if self.goals:
            for g in self.goals:
                lines.append(f"- {g.get('goal', '不明')} (進捗: {g.get('progress', 0):.0%})")
        else:
            lines.append("（未設定）")
        lines.append("")

        if self.routines:
            lines.append("## 現在のルーティン")
            for r in self.routines:
                status = "✓" if r.enabled else "✗"
                lines.append(f"- [{status}] {r.name} ({r.frequency})")

        lines.extend([
            "",
            "## 指示",
            "以下の観点で長期計画を更新してください。",
            "1. long_term_goal: 1週間〜1ヶ月単位の長期目標",
            "2. routines: 日次・週次のルーティン（最大3つ）",
            "3. identity_policy: 自分はどうありたいか（アイデンティティ方針）",
            "4. focus_area: 現在注力すべき領域",
            "5. reflection: 前回からの振り返り",
        ])
        return "\n".join(lines)

    def _parse_plan_response(self, response: str) -> dict:
        """LLM応答から長期計画をパースする。"""
        result = {}
        lines = response.strip().split("\n")

        # YAML風のパース
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()

                if key == "long_term_goal":
                    result["long_term_goal"] = value
                elif key == "identity_policy":
                    result["identity_policy"] = value
                elif key == "focus_area":
                    result["focus_area"] = value
                elif key == "reflection":
                    result["reflection"] = value

        # routines のパース（より複雑）
        routines = []
        in_routines = False
        current_routine = {}
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("routines:"):
                in_routines = True
                continue
            if in_routines:
                if stripped.startswith("- name:"):
                    if current_routine and "name" in current_routine:
                        routines.append(current_routine)
                    current_routine = {"name": stripped.split(":", 1)[1].strip()}
                elif current_routine:
                    if stripped.startswith("action:"):
                        current_routine["action"] = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("frequency:"):
                        current_routine["frequency"] = stripped.split(":", 1)[1].strip()

        if current_routine and "name" in current_routine:
            routines.append(current_routine)

        if routines:
            result["routines"] = routines

        return result

    def _merge_plan(self, parsed: dict, input: LongTermPlanningInput):
        """
        パースされた計画を現在の状態にマージする。

        LLM出力の一部だけを採用し、既存の状態を基本とする。
        同一目標の重複追加を防止する。
        """
        # 長期目標
        if "long_term_goal" in parsed:
            new_goal = parsed["long_term_goal"]
            # 重複チェック: 既存の目標と一致するものがあればスキップ
            if any(g["goal"] == new_goal for g in self.goals):
                logger.debug(f"Skipped duplicate goal: {new_goal[:40]}")
            elif len(self.goals) < 5:
                self.goals.append({
                    "goal": new_goal,
                    "progress": 0.0,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                })

        # アイデンティティ方針
        if "identity_policy" in parsed:
            self.identity_policy = parsed["identity_policy"]

        # 注力領域
        if "focus_area" in parsed:
            self.focus_area = parsed["focus_area"]

        # ルーティン
        if "routines" in parsed:
            for r_data in parsed["routines"]:
                if isinstance(r_data, dict):
                    existing = [r for r in self.routines if r.name == r_data.get("name")]
                    if not existing:
                        self.routines.append(Routine(
                            name=r_data.get("name", "未命名ルーティン"),
                            action=r_data.get("action", ""),
                            frequency=r_data.get("frequency", "daily"),
                            enabled=True,
                        ))

    def _deduplicate_goals(self):
        """
        目標リスト内の重複を排除する。
        同一テキストの目標は最初の1件のみ保持。
        """
        seen = set()
        unique = []
        for g in self.goals:
            goal_text = g.get("goal", "")
            if goal_text and goal_text not in seen:
                seen.add(goal_text)
                unique.append(g)
            else:
                logger.debug(f"Removed duplicate goal: {goal_text[:40]}")
        if len(unique) < len(self.goals):
            removed = len(self.goals) - len(unique)
            self.goals = unique
            logger.info(f"Deduplicated goals: {len(self.goals)} unique (removed {removed} duplicates)")

    def _load(self):
        """ストレージから状態を読み込む。"""
        try:
            filepath = self.storage_path / "long_term_plan.json"
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.goals = data.get("goals", [])
                self.routines = [
                    Routine(**r) if isinstance(r, dict) else r
                    for r in data.get("routines", [])
                ]
                self.identity_policy = data.get("identity_policy", "")
                self.focus_area = data.get("focus_area", "")
                last = data.get("last_plan_update")
                if last:
                    self.last_plan_update = datetime.fromisoformat(last)
                logger.info(f"Loaded long-term plan: {len(self.goals)} goals, {len(self.routines)} routines")
        except Exception as e:
            logger.warning(f"Failed to load long-term plan: {e}")

    def _save(self):
        """状態をストレージに保存する。"""
        try:
            filepath = self.storage_path / "long_term_plan.json"
            # Routine オブジェクトを dict に変換
            routines_data = []
            for r in self.routines:
                rd = {
                    "name": r.name,
                    "action": r.action,
                    "frequency": r.frequency,
                    "interval_hours": r.interval_hours,
                    "last_executed": r.last_executed.isoformat() if r.last_executed else None,
                    "enabled": r.enabled,
                }
                routines_data.append(rd)

            data = {
                "goals": self.goals,
                "routines": routines_data,
                "identity_policy": self.identity_policy,
                "focus_area": self.focus_area,
                "last_plan_update": self.last_plan_update.isoformat() if self.last_plan_update else None,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Long-term plan saved")
        except Exception as e:
            logger.error(f"Failed to save long-term plan: {e}")
