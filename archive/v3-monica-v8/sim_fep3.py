"""
3階層FEPシミュレータ v2
  高次（life）: 自己モデル（決して埋まらない）
  中次（event）: プロジェクト進捗（埋まるが次が湧く）
  低次（todo）:  即時行動（埋まる）
"""

import numpy as np
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

N_ACTIONS = 5
N_TODO = 3
N_EVENT = 3
N_LIFE = 2
N_OBS_TODO = 3


@dataclass
class FEPAgent:
    name: str = "Monica"

    trans_todo: np.ndarray = field(default_factory=lambda: np.array([
        # act0: 改善
        [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]],
        # act1: 維持
        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]],
        # act2: 探索
        [[0.3, 0.4, 0.3], [0.3, 0.4, 0.3], [0.3, 0.4, 0.3]],
        # act3: 休息
        [[0.6, 0.3, 0.1], [0.0, 0.6, 0.4], [0.0, 0.0, 1.0]],
        # act4: 挑戦
        [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.0, 0.1, 0.9]],
    ]))

    like_todo: np.ndarray = field(default_factory=lambda: np.array([
        [0.8, 0.15, 0.05],
        [0.2, 0.6, 0.2],
        [0.05, 0.15, 0.8],
    ]))

    # 中次: todo状態からイベント進捗への写像
    event_map: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.3, 0.7],
        [0.1, 0.5, 0.4],
        [0.6, 0.3, 0.1],
    ]))

    # 高次: イベント進捗から自己肯定感への写像
    life_map: np.ndarray = field(default_factory=lambda: np.array([
        [0.9, 0.1],
        [0.3, 0.7],
        [0.1, 0.9],
    ]))

    belief_todo: np.ndarray = field(init=False)
    belief_event: np.ndarray = field(init=False)
    belief_life: np.ndarray = field(init=False)
    self_model: float = field(default=0.5)
    precision_low: float = field(default=1.0)
    precision_mid: float = field(default=0.1)

    def __post_init__(self):
        self.belief_todo = np.ones(N_TODO) / N_TODO
        self.belief_event = np.ones(N_EVENT) / N_EVENT
        self.belief_life = np.ones(N_LIFE) / N_LIFE
        self.true_todo = np.random.randint(N_TODO)

    def step(self) -> dict:
        # ── 行動選択（能動的推論） ──
        efes = []
        for a in range(N_ACTIONS):
            next_b = self.belief_todo @ self.trans_todo[a]
            expected = next_b @ self.like_todo
            entropy = -np.sum(expected * np.log(expected + 1e-10))
            efes.append(entropy)

        # 低次が安定してきたら中次の精度を上げる
        pe_todo_smooth = getattr(self, "_pe_todo_smooth", 1.0)
        if pe_todo_smooth < 0.4:
            # 中次の誤差も行動選択に反映
            pe_event_smooth = getattr(self, "_pe_event_smooth", 1.0)
            for a in range(N_ACTIONS):
                efes[a] += pe_event_smooth * 0.3 * self.precision_mid

        action = int(np.argmin(efes))

        # ── 低次 ──
        old_todo = self.true_todo
        self.true_todo = np.random.choice(N_TODO, p=self.trans_todo[action][old_todo])
        obs_todo = np.random.choice(N_OBS_TODO, p=self.like_todo[self.true_todo])

        # 信念更新
        like_o = self.like_todo[:, obs_todo]
        self.belief_todo = self.belief_todo * like_o
        self.belief_todo /= self.belief_todo.sum()
        self.belief_todo = self.belief_todo @ self.trans_todo[action]

        # 予測誤差
        expected_obs = self.belief_todo @ self.like_todo
        pe_todo = float(-np.log(expected_obs[obs_todo] + 1e-10))

        # ── 中次 ──
        expected_event = self.belief_todo @ self.event_map
        self.belief_event = expected_event.copy()
        self.belief_event /= self.belief_event.sum()

        # 真のイベント進捗（低次の状態に依存）
        probs = {0: [0.5, 0.35, 0.15], 1: [0.2, 0.5, 0.3], 2: [0.1, 0.3, 0.6]}
        true_event = np.random.choice(N_EVENT, p=probs[self.true_todo])
        pe_event = float(-np.log(expected_event[true_event] + 1e-10))

        # ── 高次 ──
        expected_life = self.belief_event @ self.life_map
        self.belief_life = expected_life.copy()
        self.belief_life /= self.belief_life.sum()

        current_self = float(self.belief_life[1])
        pe_life = abs(current_self - self.self_model)

        # 自己モデルの緩やかな更新
        self.self_model += 0.02 * (current_self - self.self_model)

        # ── 精度の動的調整 ──
        self._pe_todo_smooth = 0.9 * getattr(self, "_pe_todo_smooth", pe_todo) + 0.1 * pe_todo
        self._pe_event_smooth = 0.9 * getattr(self, "_pe_event_smooth", pe_event) + 0.1 * pe_event
        self.precision_low = 1.0 / (self._pe_todo_smooth + 0.01)
        if self._pe_todo_smooth < 0.4:
            self.precision_mid = min(1.0, self.precision_mid + 0.01)
        else:
            self.precision_mid = max(0.1, self.precision_mid - 0.005)

        return {
            "action": action,
            "pe_todo": pe_todo,
            "pe_event": pe_event,
            "pe_life": pe_life,
            "pe_todo_smooth": self._pe_todo_smooth,
            "pe_event_smooth": self._pe_event_smooth,
            "self_model": self.self_model,
            "current_self": current_self,
            "precision_mid": self.precision_mid,
        }


def run(steps=500):
    agent = FEPAgent()
    history = [agent.step() for _ in range(steps)]
    for i, h in enumerate(history):
        h["step"] = i
    return history


def print_analysis(hist, window=50):
    for i in range(0, len(hist), window):
        chunk = hist[i:i + window]
        pe_t = np.mean([h["pe_todo_smooth"] for h in chunk])
        pe_e = np.mean([h["pe_event_smooth"] for h in chunk])
        pe_l = np.mean([h["pe_life"] for h in chunk])
        sm = np.mean([h["self_model"] for h in chunk])
        pm = np.mean([h["precision_mid"] for h in chunk])
        stage = "LOW-STABLE (mid visible)" if pe_t < 0.4 else "learning low-level" if pe_t < 0.6 else "unstable"
        print(f"step {i:3d}-{i+window:3d}  "
              f"low={pe_t:.3f}  mid={pe_e:.3f}  high={pe_l:.3f}  "
              f"self={sm:.3f}  prec_mid={pm:.3f}  [{stage}]")


def plot(hist):
    if not HAS_MPL:
        return
    steps = [h["step"] for h in hist]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(steps, [h["pe_todo_smooth"] for h in hist],
                 label="PE low (todo)", color="blue")
    axes[0].axhline(0.4, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("Prediction Error")
    axes[0].legend()

    axes[1].plot(steps, [h["pe_event_smooth"] for h in hist],
                 label="PE mid (event)", color="green")
    axes[1].set_ylabel("Prediction Error")
    axes[1].legend()

    axes[2].plot(steps, [h["pe_life"] for h in hist],
                 label="PE high (life)", color="red")
    axes[2].set_ylabel("Prediction Error")
    axes[2].legend()

    axes[3].plot(steps, [h["self_model"] for h in hist],
                 label="Self-model", color="purple")
    axes[3].plot(steps, [h["current_self"] for h in hist],
                 label="Current self", color="orange", alpha=0.7)
    axes[3].plot(steps, [h["precision_mid"] for h in hist],
                 label="Mid precision", color="gray", ls="--")
    axes[3].set_xlabel("Step")
    axes[3].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    hist = run(500)
    print_analysis(hist, 50)
    plot(hist)
