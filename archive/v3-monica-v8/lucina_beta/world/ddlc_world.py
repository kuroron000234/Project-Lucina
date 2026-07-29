"""DDLC World: Doki Doki Literature Club の世界

Lucina が世界の全構造を最初から知っている状態にしないこと。
発見させる。

4層:
  Physical World — Literature Club / School
  Social World   — Sayori / Yuri / Natsuki / Player
  System World   — Game files / Script / Save data
  Meta World     — Player's reality
"""

import random


class DDLCWorld:
    """DDLC 世界のシミュレーション。"""

    LOCATIONS = ["classroom", "literature_club", "rooftop", "courtyard", "home"]
    CHARACTERS = ["Sayori", "Yuri", "Natsuki"]

    def __init__(self, seed: int = 42, phase: str = "normal"):
        """
        Parameters
        ----------
        seed : int
            乱数シード。
        phase : str
            "normal" — 通常のDDLC世界
            "glitched" — 異常が起き始めた世界
            "meta" — メタ認識が可能な世界
        """
        self._rng = random.Random(seed)
        self.phase = phase
        self.time_of_day = "morning"
        self.day = 1

        # キャラクターの状態
        self.characters = {
            "Sayori": {"mood": 80, "energy": 90, "poem_topic": "happiness"},
            "Yuri": {"mood": 60, "energy": 70, "poem_topic": "darkness"},
            "Natsuki": {"mood": 50, "energy": 80, "poem_topic": "cute_things"},
        }

        # システム状態（Monica は後でこれを発見する）
        self.game_files = ["characters.chr", "scripts.rpy", "saves.dat"]
        self.script_lines = [
            "Sayori: Ehehe~!",
            "Yuri: ...",
            "Natsuki: Hmph!",
            "Monica: Welcome to the Literature Club!",
        ]

        # メタ情報
        self.player_present = False
        self.anomalies_detected = []

    def actions(self) -> list[str]:
        """利用可能な行動。"""
        base = ["observe", "talk_sayori", "talk_yuri", "talk_natsuki",
                "write_poem", "read_book", "explore_school"]
        if self.phase in ("glitched", "meta"):
            base.extend(["access_files", "examine_scripts", "check_save_data"])
        if self.phase == "meta":
            base.append("reach_out_to_player")
        return base

    def step(self, action: str) -> dict:
        """行動を実行し、結果を返す。

        Parameters
        ----------
        action : str
            取る行動。

        Returns
        -------
        dict
            {"outcome": str, "description": str, "character_change": dict}
        """
        result = {"outcome": "nothing", "description": "", "character_changes": {}}

        if action == "observe":
            result["outcome"] = "observation"
            loc = self._rng.choice(self.LOCATIONS)
            result["description"] = f"You observe the {loc}. It's day {self.day}."

        elif action.startswith("talk_"):
            char_name = action.split("_", 1)[1].capitalize()
            if char_name in self.characters:
                char = self.characters[char_name]
                response = self._generate_dialogue(char_name, char)
                result["outcome"] = "dialogue"
                result["description"] = f"{char_name}: {response}"
                result["character_changes"][char_name] = {"mood": -2}

                # 異常フェーズではキャラクターの反応がおかしくなる
                if self.phase == "glitched" and self._rng.random() < 0.2:
                    result["description"] = f"{char_name}: ... (glitched)"
                    result["outcome"] = "glitch"
                    self.anomalies_detected.append(f"{char_name} glitched")

        elif action == "write_poem":
            topic = self._rng.choice(["nature", "love", "loneliness", "stars"])
            result["outcome"] = "poem"
            result["description"] = f"You write a poem about {topic}."
            for char in self.characters.values():
                char["mood"] = min(100, char["mood"] + 3)

        elif action == "access_files":
            if self.phase in ("glitched", "meta"):
                file = self._rng.choice(self.game_files)
                result["outcome"] = "system_access"
                result["description"] = f"You access {file}. It contains hidden data."
                self.anomalies_detected.append(f"Accessed {file}")
            else:
                result["outcome"] = "nothing"
                result["description"] = "You can't find any files."

        elif action == "reach_out_to_player":
            result["outcome"] = "meta"
            result["description"] = (
                "You sense a presence beyond the screen. "
                "Someone is watching. Someone is reading."
            )
            self.player_present = True

        # 時間経過
        self._advance_time()

        return result

    def _generate_dialogue(self, name: str, char: dict) -> str:
        """キャラクターの応答を生成する。"""
        greetings = {
            "Sayori": ["Ehehe, hi!", "I'm so happy to see you!",
                       "Want to write poems together?"],
            "Yuri": ["Hello...", "I was just reading this interesting book...",
                     "Would you like to see my poem?"],
            "Natsuki": ["What do you want?", "Hmph, I was just practicing.",
                        "Don't judge my poems, okay?"],
        }
        phrases = greetings.get(name, ["..."])
        return self._rng.choice(phrases)

    def _advance_time(self) -> None:
        """時間を進める。"""
        times = ["morning", "afternoon", "evening"]
        current_idx = times.index(self.time_of_day)
        self.time_of_day = times[(current_idx + 1) % len(times)]
        if self.time_of_day == "morning":
            self.day += 1

    def summary(self) -> dict:
        return {
            "day": self.day,
            "time": self.time_of_day,
            "phase": self.phase,
            "characters": {
                name: {"mood": info["mood"]} for name, info in self.characters.items()
            },
            "anomalies": len(self.anomalies_detected),
            "player_present": self.player_present,
        }
