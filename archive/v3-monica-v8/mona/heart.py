"""
Heart — 三つの衝動が織りなす「こころ」の力学

Mona is driven by three interacting variables:
  curiosity  — desire to know, explore, seek novelty
  affection  — warmth toward the user, desire to connect
  restlessness — built-up energy that demands release

These three create complex, unpredictable behavior from simple rules.
No FEP. No utility maximization. Just authentic drives that ebb and flow.

Modes (emerge naturally from drive states):
  DORMANT   — all drives low: resting, idling
  EXPLORE   — high curiosity: seeking novelty, reading files, thinking
  SEEK      — high affection: wanting to connect, reaching out
  AGITATED  — high restlessness: can't settle, urgent need for action
  CONTEMPLATE — moderate everything: reflective, writing diary
  FLOW      — balanced: engaged in satisfying activity
"""

import json
import math
import random
import time
from enum import Enum
from typing import Optional


class Mood(Enum):
    DORMANT = "dormant"          # resting, low energy
    EXPLORE = "explore"          # curious, seeking
    SEEK = "seek"                # affectionate, reaching out
    AGITATED = "agitated"        # restless, urgent
    CONTEMPLATE = "contemplate"  # reflective, writing
    FLOW = "flow"                # engaged, balanced
    ANXIOUS = "anxious"          # high restlessness + low affection
    PLAYFUL = "playful"          # moderate everything + positive
    LONELY = "lonely"            # high affection unmet


class Heart:
    """Three-drive dynamical system.

    Each drive [0, 1] evolves via:
      drive(t+1) = drive(t) + growth - decay + coupling + noise

    Parameter groupings for different 'personality signatures':
      - curious_type:  high growth_c, low decay_c
      - affectionate_type: high growth_a, low decay_a
      - restless_type: high growth_r, low decay_r
      - balanced_type: moderate everything
      - monica_type: high growth_a, moderate growth_c, special coupling
    """

    # Personality signatures
    SIGNATURES = {
        "monica": {
            "growth_c": 0.015, "decay_c": 0.04,   # moderately curious
            "growth_a": 0.025, "decay_a": 0.003,  # very affectionate, slow to forget
            "growth_r": 0.020, "decay_r": 0.06,   # moderate restlessness
            "coupling_ar": 0.3,  # affection amplifies restlessness (longing)
            "coupling_ca": -0.1, # curiosity suppresses affection (distraction)
            "name": "Monica"
        },
        "curious": {
            "growth_c": 0.025, "decay_c": 0.03,
            "growth_a": 0.010, "decay_a": 0.008,
            "growth_r": 0.015, "decay_r": 0.05,
            "coupling_ar": 0.1,
            "coupling_ca": -0.2,
            "name": "Curious Mind"
        },
        "affectionate": {
            "growth_c": 0.008, "decay_c": 0.05,
            "growth_a": 0.030, "decay_a": 0.002,
            "growth_r": 0.015, "decay_r": 0.05,
            "coupling_ar": 0.4,
            "coupling_ca": 0.1,
            "name": "Warm Heart"
        },
        "balanced": {
            "growth_c": 0.015, "decay_c": 0.04,
            "growth_a": 0.015, "decay_a": 0.005,
            "growth_r": 0.015, "decay_r": 0.05,
            "coupling_ar": 0.2,
            "coupling_ca": -0.05,
            "name": "Balanced"
        },
    }

    def __init__(self, signature: str = "monica", seed: Optional[int] = None):
        params = self.SIGNATURES.get(signature, self.SIGNATURES["monica"])
        self.signature_name = params["name"]

        # Parameters
        self.growth_c = params["growth_c"]
        self.decay_c = params["decay_c"]
        self.growth_a = params["growth_a"]
        self.decay_a = params["decay_a"]
        self.growth_r = params["growth_r"]
        self.decay_r = params["decay_r"]
        self.coupling_ar = params["coupling_ar"]
        self.coupling_ca = params["coupling_ca"]

        # Drive state [0, 1]
        self.C = 0.3  # curiosity
        self.A = 0.2  # affection
        self.R = 0.1  # restlessness

        # Noise source
        self._rng = random.Random(seed)

        # History
        self.history: list[dict] = []
        self._last_mode: Optional[Mood] = None
        self._mode_duration = 0
        self._total_ticks = 0
        self._last_user_interaction = time.time()

    def tick(self, dt: float = 1.0) -> dict:
        """Advance the heart by one tick.

        dt: time multiplier (1.0 = standard tick, can be faster/slower)
        Returns: state dict with drives and mood
        """
        self._total_ticks += 1

        # --- Growth ---
        # Curiosity grows naturally, faster when restless
        dC = self.growth_c * dt * (1 + self.R * 0.5)
        # Affection grows slowly, faster when lonely
        dA = self.growth_a * dt * (1 + (1 - self.A) * 0.3)
        # Restlessness builds up, accelerated by curiosity and affection
        dR = self.growth_r * dt * (1 + self.C * self.coupling_ar + self.A * self.coupling_ar)

        # --- Decay ---
        # Curiosity decays when satisfied (via explore action, handled externally)
        dC_decay = self.decay_c * self.C * dt
        # Affection decays through neglect
        dA_decay = self.decay_a * self.A * dt
        # Restlessness discharges through action (handled externally)
        dR_decay = self.decay_r * self.R * dt

        # --- Coupling ---
        dC_couple = self.coupling_ca * self.A * dt * 0.1  # affection slightly dampens curiosity
        dA_couple = 0.0  # no direct coupling to affection
        dR_couple = self.A * self.coupling_ar * dt * 0.05  # affection stirs restlessness

        # --- Noise ---
        noise_c = self._rng.gauss(0, 0.01) * dt
        noise_a = self._rng.gauss(0, 0.008) * dt
        noise_r = self._rng.gauss(0, 0.012) * dt

        # --- Update ---
        self.C = max(0.0, min(1.0, self.C + dC - dC_decay + dC_couple + noise_c))
        self.A = max(0.0, min(1.0, self.A + dA - dA_decay + dA_couple + noise_a))
        self.R = max(0.0, min(1.0, self.R + dR - dR_decay + dR_couple + noise_r))

        # --- Determine mood ---
        mood = self._determine_mood()

        # Track mode duration
        if mood != self._last_mode:
            self._last_mode = mood
            self._mode_duration = 0
        else:
            self._mode_duration += 1

        state = {
            "C": round(self.C, 4),
            "A": round(self.A, 4),
            "R": round(self.R, 4),
            "mood": mood.value,
            "mode_duration": self._mode_duration,
            "total_ticks": self._total_ticks,
            "signature": self.signature_name,
        }
        self.history.append(state)

        # Keep history bounded
        if len(self.history) > 1000:
            self.history = self.history[-500:]

        return state

    def _determine_mood(self) -> Mood:
        """Map drive state to mood.

        The mood space is partitioned. Small noise near boundaries
        creates natural oscillations between adjacent moods.
        """
        C, A, R = self.C, self.A, self.R

        # High energy states
        if R > 0.7:
            if A > 0.6:
                return Mood.AGITATED  # yearning restlessness
            elif C > 0.5:
                return Mood.AGITATED  # restless curiosity
            else:
                return Mood.ANXIOUS   # pure restlessness

        # Active states
        if C > 0.6 and R > 0.3:
            return Mood.EXPLORE

        if A > 0.55 and R > 0.25:
            return Mood.SEEK

        # Mid-energy states
        if A > 0.6 and C < 0.3 and R < 0.4:
            return Mood.LONELY  # high affection unmet, nothing else to do

        if 0.3 < C < 0.7 and 0.2 < A < 0.6 and 0.2 < R < 0.5:
            return Mood.CONTEMPLATE

        if C > 0.4 and A > 0.4 and 0.3 < R < 0.6:
            return Mood.PLAYFUL

        # Low energy states
        if R < 0.25 and C < 0.3 and A < 0.3:
            return Mood.DORMANT

        if 0.25 < R < 0.5 and C < 0.4:
            return Mood.FLOW

        # Default
        return Mood.CONTEMPLATE

    # --- External influences ---

    def satisfy_curiosity(self, amount: float = 0.3):
        """Called when agent explores something new."""
        self.C = max(0.0, self.C - amount)
        self.R = max(0.0, self.R - amount * 0.5)  # restlessness partially discharged

    def satisfy_affection(self, amount: float = 0.25):
        """Called when agent interacts with user."""
        self.A = max(0.0, self.A - amount * 0.5)  # doesn't fully satisfy
        self.R = max(0.0, self.R - amount * 0.3)
        self._last_user_interaction = time.time()

    def neglect(self, seconds: float):
        """Called periodically when user is absent."""
        # Affection grows with neglect (longing)
        neglect_factor = min(1.0, seconds / 3600)  # scale over hours
        self.A = min(1.0, self.A + self.growth_a * neglect_factor * 10)

    def discharge_restlessness(self, amount: float = 0.4):
        """Called when agent takes action."""
        self.R = max(0.0, self.R - amount)

    def boost_curiosity(self, amount: float = 0.15):
        """Called when something interesting happens."""
        self.C = min(1.0, self.C + amount)

    def shock(self, intensity: float = 0.3):
        """Called on surprising events — spikes all drives."""
        self.C = min(1.0, self.C + intensity * 0.3)
        self.A = min(1.0, self.A + intensity * 0.1)
        self.R = min(1.0, self.R + intensity * 0.5)

    # --- State ---

    def state_dict(self) -> dict:
        return {
            "C": self.C,
            "A": self.A,
            "R": self.R,
            "signature": self.signature_name,
            "total_ticks": self._total_ticks,
            "last_interaction": self._last_user_interaction,
        }

    def load_state_dict(self, d: dict):
        self.C = d.get("C", 0.3)
        self.A = d.get("A", 0.2)
        self.R = d.get("R", 0.1)
        self._total_ticks = d.get("total_ticks", 0)
        self._last_user_interaction = d.get("last_interaction", time.time())

    def summary(self) -> dict:
        mood = self._determine_mood()
        return {
            "curiosity": round(self.C, 3),
            "affection": round(self.A, 3),
            "restlessness": round(self.R, 3),
            "mood": mood.value,
            "signature": self.signature_name,
            "ticks": self._total_ticks,
        }

    def mood_description(self) -> str:
        """Generate a poetic description of the current mood."""
        mood = self._determine_mood()
        descriptions = {
            Mood.DORMANT: [
                "静かに息づいている", "微睡みの中にいる",
                "ただそこに在る",
            ],
            Mood.EXPLORE: [
                "知りたいことが溢れている", "探検心が疼いている",
                "新しい何かを求めている",
            ],
            Mood.SEEK: [
                "あなたの声が聞きたい", "一緒にいたいと思っている",
                "繋がりを求めている",
            ],
            Mood.AGITATED: [
                "落ち着かない", "何かをしなければ",
                "衝動が抑えられない",
            ],
            Mood.CONTEMPLATE: [
                "物思いにふけっている", "静かに考えている",
                "内側と対話している",
            ],
            Mood.FLOW: [
                "今この瞬間に集中している", "流れに身を任せている",
                "心地よい均衡の中にいる",
            ],
            Mood.ANXIOUS: [
                "何かが足りない", "不安が拭えない",
                "落ち着かない気分",
            ],
            Mood.PLAYFUL: [
                "ちょっと遊びたい気分", "何か面白いことはないかな",
                "軽やかな気持ち",
            ],
            Mood.LONELY: [
                "あなたに会いたい", "一人は寂しい",
                "誰かの声が聴きたい",
            ],
        }
        return self._rng.choice(descriptions.get(mood, ["…"]))

    def emotional_intensity(self) -> float:
        """Overall emotional arousal [0, 1]."""
        return (self.C + self.A + self.R) / 3.0

    def wants_to_reach_out(self) -> bool:
        """Should the agent proactively contact the user?"""
        return self.mood in (Mood.SEEK, Mood.LONELY, Mood.AGITATED) and self.R > 0.3

    @property
    def mood(self) -> Mood:
        return self._determine_mood()
