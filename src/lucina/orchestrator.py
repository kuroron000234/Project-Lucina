"""
オーケストレーター — キャラクター、記憶、LLMを統合する薄い層

デュアルモデル構成:
- キャラ層: G4-Midnight-Macaw (ローカル) — 何をしたいか決める
- エージェント層: hy3-free (API) — それを実行する
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from .agent import run_agent
from .character import Character
from .llm import LLM
from .memory import Episode, Memory
from .prompt import build_messages

logger = logging.getLogger("orchestrator")

# キャラ層の判断ブロックを検出するパターン
# LLMは閉じ括弧を欠いたり「}」で閉じたりするため、終端を「】」「改行」「文末」のいずれかで許容する
DELEGATION_PATTERN = r"【委託:\s*(検索|URL取得|ファイル作成|ファイル読み取り|コード実行|天気)[:\s]*(.+?)(?:】|\n|$)"


class Orchestrator:
    """Thin orchestration layer connecting character, memory, and LLM."""

    def __init__(
        self,
        model: str = "g4-midnight-macaw-v2",
        data_dir: str = "data",
    ):
        self.character = Character(f"{data_dir}/persistent.json")
        self.memory = Memory(f"{data_dir}/episodes", f"{data_dir}/summaries")
        self.llm = LLM(model=model)
        self.data_dir = Path(data_dir)
        self.chat_history: list[dict] = []
        self.max_history = 20

    def process(self, user_input: str) -> str:
        """Process user input and return response."""
        self.character.increment_interactions()

        # 1. 関連する記憶を検索（多チャネル + 連想）
        memories = self.memory.search(user_input, top_k=3)
        q_entities = self.memory._extract_query_entities(user_input)
        assoc = self.memory.related_by_entity(q_entities, top_k=2) if q_entities else []
        seen = {m.id for m in memories}
        combined = list(memories)
        for a in assoc:
            if a.id not in seen:
                combined.append(a)

        memory_texts = []
        latest_summary = self.memory.latest_summary()
        if latest_summary:
            memory_texts.append(latest_summary)
        for m in combined:
            stamp = m.timestamp.strftime('%Y-%m-%d %H:%M')
            note = m.event
            if m.result:
                # 起こったこと（応答の冒頭）も含めると思い出しやすい
                note += f" → {m.result[:60]}"
            memory_texts.append(f"[{stamp}] {note}")

        # 2. メッセージを構築
        inner_thought = self._latest_inner_thought()
        messages = build_messages(
            character_data={
                "seed": self.character.get_seeds(),
                "self_model": self.character.get_self_model(),
                "speech_patterns": self.character.data.get("speech_patterns", {}),
                "relationships": self.character.data.get("relationships", {}),
                "state": self.character.get_state(),
            },
            memory_texts=memory_texts,
            user_message=user_input,
            chat_history=self.chat_history[-self.max_history:],
            inner_thought=inner_thought,
        )

        # 3. キャラ層から応答を生成
        raw_response = self.llm.chat(messages)

        if not raw_response:
            return "……"

        # 4. 判断ブロックをパース
        dialogue, delegation = self._parse_response(raw_response)

        # 5. 委託があればエージェント層に実行させ、結果をキャラ層に渡す
        if delegation:
            agent_result = self._delegate_to_agent(delegation, user_input)
            if agent_result:
                task_type, task_content = delegation
                dialogue = self._regenerate_with_result(
                    messages, task_content, agent_result
                )
                # 再生成時に判断ブロックが出ていても除去（ループ防止）
                dialogue = self._strip_delegation(dialogue)

        # 6. チャット履歴を更新
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": dialogue})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

        # 7. 駆動値更新（対話成立: 親密度上昇・孤独/退屈の解消・モード再導出）
        self.character.on_interaction()

        # 8. エピソード記憶として保存
        self._save_episode(user_input, dialogue)

        return dialogue

    def _latest_inner_thought(self) -> str | None:
        """最も新しい内言（独り時間のひとりごと）を返す。なければ None。"""
        try:
            eps = self.memory.recent_episodes(n=20)
            for e in eps:
                if e.source == "autonomous" and "内言" in (e.tags or []):
                    if e.result:
                        return f"[{e.timestamp.strftime('%m/%d %H:%M')}] {e.result.strip()}"
            for e in eps:
                if e.source == "autonomous":
                    if e.event:
                        return f"[{e.timestamp.strftime('%m/%d %H:%M')}] {e.event.strip()}"
            return None
        except Exception:
            return None

    def _parse_response(self, raw: str) -> tuple[str, tuple[str, str] | None]:
        """応答を「セリフ」と「判断ブロック」に分離"""
        match = re.search(DELEGATION_PATTERN, raw)
        if match:
            # セリフ: 判断ブロックより前、空行で囲まれた部分を除去
            before = raw[:match.start()].strip()
            dialogue = before if before else "……"
            task_type = match.group(1).strip()
            task_content = re.sub(r"[】\}]+$", "", match.group(2).strip())
            delegation = (task_type, task_content)
            return dialogue, delegation
        return raw.strip(), None

    def _delegate_to_agent(self, delegation: tuple[str, str], context: str) -> str | None:
        """エージェント層にタスクを委託"""
        task_type, task_content = delegation
        logger.info(f"委託: [{task_type}] {task_content[:200]}")
        task_map = {
            "検索": f"Webで「{task_content}」を検索し、結果を簡潔にまとめて",
            "URL取得": f"URL「{task_content}」の中身を取得し、要点をまとめて",
            "ファイル作成": f"ファイル「{task_content}」を作成し、内容を報告して",
            "ファイル読み取り": f"ファイル「{task_content}」を読み取り、内容を報告して",
            "コード実行": (
                f"次のゴールを達成するためのコードを自分で設計・実装して実行し、"
                f"その結果を報告してください。コードを書くべきか、シェルコマンドを使うべきか"
                f"はあなたが判断してください。\nゴール: {task_content}"
            ),
            "天気": f"「{task_content}」の天気を取得して",
        }
        full_task = task_map.get(task_type, f"タスク: {task_content}")
        try:
            result = run_agent(full_task)
            logger.info(f"Agent result length: {len(result)} chars")
            logger.info(f"Agent result: {result}")
            return result
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return None

    def _regenerate_with_result(
        self, original_messages: list[dict], task: str, result: str
    ) -> str:
        """エージェント結果を『自分自身の思考結果』として注入し応答を再生成"""
        enriched = original_messages.copy()
        enriched.append({
            "role": "user",
            "content": (
                f"あなたは調べるために自らの手と目を使いました。"
                f"以下はあなた自身が得た調査結果であり、外部の誰かからもらった情報ではありません。"
                f"これを外部の存在が提供したように話してはいけません。"
                f"あくまであなた自身の観察・思考の材料として、あなたの言葉と視点で咀嚼して語ってください。\n\n"
                f"調査内容: {task}\n"
                f"調査結果: {result}\n\n"
                "セリフのみで応答してください。判断ブロックは不要です。"
            ),
        })
        return self.llm.chat(enriched) or result

    def _strip_delegation(self, response: str) -> str:
        """応答から判断ブロックを除去（ループ防止）"""
        return re.sub(DELEGATION_PATTERN, "", response).strip()

    def _save_episode(self, user_input: str, response: str):
        """Save interaction as episode memory."""
        importance = 0.5
        if len(user_input) > 50:
            importance = 0.7
        if any(kw in user_input for kw in ["大切", "重要", "約束", "忘れない"]):
            importance = 0.9

        ep = Episode(
            id="",
            timestamp=datetime.now(),
            event=user_input,
            context="対話",
            emotion=self.character.get_state().get("mode", "tatemae"),
            result=response[:200],
            importance=importance,
            tags=["対話"],
            source="user",
        )
        self.memory.save(ep)

    def reflect(self) -> str:
        """Periodic self-reflection (called by loop)."""
        recent = sorted(
            self.memory.episodes, key=lambda e: e.timestamp, reverse=True
        )[:5]
        if not recent:
            return ""

        summary = "\n".join([
            f"- [{e.timestamp.strftime('%m/%d %H:%M')}] {e.event[:50]}"
            for e in recent
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "あなたは自分の行動を振り返る存在です。"
                    "最近の出来事を踏まえて、あなた自身についての認識を更新してください。"
                ),
            },
            {
                "role": "user",
                "content": f"最近の出来事:\n{summary}\n\nあなた自身の認識を更新してください。",
            },
        ]

        new_self_model = self.llm.chat(messages)
        if new_self_model:
            self.character.update_self_model(new_self_model)

        return new_self_model

    def consolidate(self) -> str:
        """Segmented day-summary (consolidation) of today's episodes.

        背景メンテナンスパス（loopから呼ばれる）。人間らしい記憶のために:
         1. 忘却 (MemoryBank Ebbinghaus): 弱い記憶を確率的に忘れる
         2. 注釈 (poignancy + エンティティ抽出): 未注釈episodeにLLMで重要度・登場物を付与
         3. 反射 (Generative Agents reflect): 高重要度記憶から洞察を生成し記憶に蓄積
         4. 日次要約: 直近episodeを要約して保存（常時注入の土台）
        """
        # 1. 忘却
        try:
            self.memory.forget()
        except Exception as e:
            logger.error(f"Forget error: {e}")

        # 2. 注釈（poignancy + entities）— 独立LLM呼び出しの負荷を避け、未注釈分だけ
        try:
            self._annotate_batch()
        except Exception as e:
            logger.error(f"Annotate error: {e}")

        # 3. 反射（洞察）— 高重要度の最近記憶から
        try:
            reflected = self._reflect()
        except Exception:
            reflected = False

        # 4. 日次要約
        recent = self.memory.recent_episodes(n=12)
        if len(recent) < 3:
            return ""

        lines = []
        for e in recent:
            note = f"- [{e.timestamp.strftime('%H:%M')}] {e.event[:120]}"
            if e.result:
                note += f" → {e.result[:120]}"
            lines.append(note)

        prompt = (
            "あなたは自分の今日の記録を整理する存在です。\n"
            "以下は今日の出来事（古い順）。あなたが実際に経験し、行ったこと、"
            "ユーザーと交わした会話の要点です。\n\n"
            f"{chr(10).join(reversed(lines))}\n\n"
            "これを簡潔な「今日の要約」にまとめてください。\n"
            "ルール:\n"
            "- 具体的な出来事（何を調べた、何を作った、何を約束した等）を列挙。\n"
            "- 「詩を書いていた」等の捏造は絶対にしない。記録にあることだけ書く。\n"
            "- 100〜200字程度。箇条書きでなく自然な段落で。\n"
            "要約のみを出力してください。"
        )

        messages = [
            {"role": "system", "content": "あなたは自分の行動を要約・統合する存在です。"},
            {"role": "user", "content": prompt},
        ]
        summary = self.llm.chat(messages)
        if not summary:
            return ""

        today = datetime.now().strftime('%Y-%m-%d')
        self.memory.save_day_summary(today, summary)
        return summary

    def _annotate_batch(self, limit: int = 10):
        """未注釈episodeに poignancy(0..1) と エンティティ をLLMで付与"""
        candidates = self.memory.unannotated(limit=limit)
        if not candidates:
            return
        for ep in candidates:
            text = f"{ep.event} {ep.context} {ep.result}".strip()[:300]
            if not text:
                continue
            prompt = (
                "以下の出来事について、2点を返してください。特に説明は不要、JSONのみ。\n"
                "1) importance: この出来事がどれほど重要か 0.0〜1.0 の数値\n"
                "2) entities: 登場する固有名詞・重要な対象（人/物/場所/概念）の配列\n\n"
                f"出来事: {text}\n\n"
                '形式: {"importance": 0.6, "entities": ["東京", "ピアノ"]}'
            )
            msgs = [
                {"role": "system", "content": "あなたは記憶の重要度と登場対象を簡潔に判定する。"},
                {"role": "user", "content": prompt},
            ]
            out = self.llm.chat(msgs)
            data = self.llm.extract_json(out) if out else {}
            try:
                poig = float(data.get("importance", ep.importance))
            except (TypeError, ValueError):
                poig = ep.importance
            ents = data.get("entities", [])
            ents = [str(e).strip() for e in ents if str(e).strip()][:8]
            if poig != ep.poignancy or ents != ep.entities:
                self.memory.set_annotation(ep, poignancy=poig, entities=ents)

    def _reflect(self) -> bool:
        """高重要度の最近記憶から洞察を生成し、生成できれば記憶に保存。
        Generative Agents の generate_insights_and_evidence 相当。
        """
        important = self.memory.recent_episodes(n=15)
        important = [e for e in important if e.poignancy >= 0.7][:8]
        if len(important) < 3:
            return False
        lines = [
            f"- [{e.timestamp.strftime('%m/%d %H:%M')}] {e.event[:100]}"
            for e in important
        ]
        prompt = (
            "以下はあなたの最近の重要な出来事です。\n"
            f"{chr(10).join(lines)}\n\n"
            "これらから自分自身について、またはあなたと相手の関係について"
            "1つの気づき(洞察)を、あなたの言葉で1文だけ書いてください。\n"
            "記録に無いことは捏造しないこと。洞察のみを出力。"
        )
        msgs = [
            {"role": "system", "content": "あなたは自分の経験から学びを引き出す存在。"},
            {"role": "user", "content": prompt},
        ]
        insight = self.llm.chat(msgs)
        if not insight:
            return False
        self.memory.save_reflection(insight, evidence_ids=[e.id for e in important])
        logger.info("Reflection saved as insight memory")
        return True
