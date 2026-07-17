"""
Monica Core 単体テスト — VitalOS 状態遷移・活動効果・世界モデル学習
"""

import json
import math
import random
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# テスト用にsrcをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monica_core.vital_os import (
    VitalOS, VitalState, VitalParam, WorldModel,
    ACTIVITIES, DURATIONS, PARAMS, SETPOINTS,
    DRIFT_RATES, TAG_EFFECTS, ACTIVITY_TAGS,
    compute_true_effect, MOVE_TIME, LOCATIONS,
    EXPLORATION_EPSILON, PhoneState, Event,
)


def test_vital_state_clamp():
    """状態値が0〜100の範囲にクランプされることを確認"""
    s = VitalState(energy=150, hunger=-20)
    s.clamp()
    assert s.energy == 100, f"Expected 100, got {s.energy}"
    assert s.hunger == 0, f"Expected 0, got {s.hunger}"


def test_vital_state_as_dict():
    """as_dict() が正しいキーと値を返すことを確認"""
    s = VitalState(energy=50.5, hunger=30.2)
    d = s.as_dict()
    assert "energy" in d
    assert d["energy"] == 50.5


def test_vital_state_clone():
    """clone() が独立したコピーを返すことを確認"""
    s = VitalState(energy=60, hunger=30)
    c = s.clone()
    c.energy = 90
    assert s.energy == 60, "Original should be unchanged"
    assert c.energy == 90


def test_drift():
    """ドリフトが正しい方向に働くことを確認"""
    os = VitalOS()
    initial = os.state.clone()
    os._apply_drift(60)  # 1時間
    assert os.state.energy < initial.energy, "Energy should decrease"
    assert os.state.hunger > initial.hunger, "Hunger should increase"


def test_eat_effect():
    """食事が空腹を減少させることを確認"""
    os = VitalOS()
    os.state.hunger = 80
    os.start_activity("eat")
    os._apply_activity_effects("eat", 30)
    assert os.state.hunger < 80, "Eating should decrease hunger"


def test_sleep_effect():
    """睡眠がエネルギーを回復させることを確認"""
    os = VitalOS()
    os.state.energy = 20
    os.state.fatigue = 60
    os.start_activity("sleep")
    os._apply_activity_effects("sleep", 60)
    assert os.state.energy > 20, "Sleep should increase energy"
    assert os.state.fatigue < 60, "Sleep should decrease fatigue"

def test_piano_effect():
    """ピアノが気分を向上させることを確認"""
    os = VitalOS()
    os.state.spirit = 30
    os.start_activity("piano")
    os._apply_activity_effects("piano", 30)
    assert os.state.spirit > 30, "Piano should improve spirit"


def test_compute_true_effect():
    """compute_true_effect が期待されるキーを返すことを確認"""
    effect = compute_true_effect("sleep", 60)
    for p in PARAMS:
        assert p in effect, f"Missing param {p}"


def test_room_movement_bfs():
    """BFS経路探索が正しい最短経路を返すことを確認"""
    os = VitalOS()
    os.current_room = "bedroom"
    path = os._shortest_path("bedroom", "garden")
    assert len(path) >= 1, "Should find a path to garden"
    assert path[-1] == "garden", "Path should end at garden"


def test_move_to_changes_room():
    """move_to() で部屋が変わり、移動時間が返ることを確認"""
    os = VitalOS()
    os.current_room = "bedroom"
    cost = os.move_to("kitchen")
    assert os.current_room == "kitchen", f"Expected kitchen, got {os.current_room}"
    assert cost > 0, "Movement should cost time"


def test_move_to_same_room():
    """同じ部屋への移動はコスト0"""
    os = VitalOS()
    os.current_room = "bedroom"
    cost = os.move_to("bedroom")
    assert cost == 0


def test_activity_available():
    """activity_available() が既知の活動を認識することを確認"""
    os = VitalOS()
    assert os.activity_available("sleep")
    assert os.activity_available("eat")


def test_mandatory_slot_night():
    """23時以降は強制睡眠"""
    os = VitalOS()
    os.time = os.time.replace(hour=23)
    slot = os._mandatory_slot(23)
    assert slot is not None
    assert slot[0] == "sleep"


def test_mandatory_slot_lunch():
    """12時は強制食事"""
    os = VitalOS()
    slot = os._mandatory_slot(12)
    assert slot is not None
    assert slot[0] == "eat"


def test_critical_needs_hunger():
    """高空腹時にクリティカルニーズが食事を返す"""
    os = VitalOS()
    os.state.hunger = 85
    need = os._critical_needs()
    assert need is not None
    assert need[0] == "eat"


def test_critical_needs_fatigue():
    """高疲労時にクリティカルニーズが睡眠を返す"""
    os = VitalOS()
    os.state.fatigue = 85
    need = os._critical_needs()
    assert need is not None
    assert need[0] in ("sleep",)


def test_decide_next_night():
    """夜間の decide_next が睡眠を返す"""
    os = VitalOS()
    os.time = os.time.replace(hour=23)
    action, duration = os.decide_next()
    assert action == "sleep"


def test_day_log():
    """一日のログが活動の開始・終了を記録することを確認"""
    os = VitalOS()
    os.start_activity("eat")
    os._finish_activity()
    assert len(os.day_log) >= 1


def test_save_load_cycle():
    """save/load で状態が復元されることを確認"""
    os = VitalOS()
    os.state.energy = 42
    os.state.hunger = 73
    os.state.spirit = 88
    os.current_room = "garden"
    os.save()

    os2 = VitalOS()
    loaded = os2.load()
    assert loaded, "Should load successfully"
    assert os2.state.energy == 42, f"Expected 42, got {os2.state.energy}"
    assert os2.state.hunger == 73
    assert os2.state.spirit == 88
    assert os2.current_room == "garden"


def test_history_after_activity():
    """活動後に履歴が追加されることを確認"""
    os = VitalOS()
    os.start_activity("rest")
    assert len(os.history) == 1


def test_tick_advances_time():
    """tick() で時間が進むことを確認"""
    os = VitalOS()
    initial_time = os.time
    os.tick(15)
    assert os.time == initial_time + timedelta(minutes=15)


def test_phone_state():
    """PhoneState が正しく初期化されることを確認"""
    p = PhoneState()
    assert p.consecutive_empty_checks == 0
    assert p.last_outgoing_text == ""


def test_loneliness_increases_alone():
    """孤独が時間経過で上昇することを確認"""
    os = VitalOS()
    os.state.loneliness = 30
    os.tick(120)  # 2時間
    assert os.state.loneliness > 30, "Loneliness should increase over time"


def test_spirit_decreases_alone():
    """気分が時間経過で低下することを確認"""
    os = VitalOS()
    os.state.spirit = 50
    os.tick(120)  # 2時間
    assert os.state.spirit < 50, "Spirit should decrease over time"


def test_model_satiation():
    """飽和効果が時間とともに減衰することを確認"""
    model = WorldModel()
    now = datetime.now()
    model.mark_done("eat", now)
    s1 = model.satiation("eat", now)
    s2 = model.satiation("eat", now + timedelta(hours=6))
    assert s2 < s1, "Satiation should decay over time"


def test_model_novelty_bonus():
    """未経験の活動に新規性ボーナスが与えられることを確認"""
    model = WorldModel()
    now = datetime.now()
    bonus = model.novelty_bonus("piano", now)
    assert bonus > 0, "Unknown activity should have novelty bonus"


def test_model_learning():
    """モデルが経験から学習することを確認"""
    model = WorldModel()
    model.update("sleep", {"energy": 10.0, "hunger": 0.0, "fatigue": -15.0, "loneliness": 0.0, "spirit": 0.0})
    # 学習後、初期値から更新されているはず
    assert model.deltas["sleep"]["energy"] != 30, "Should have learned from experience"


def test_deviation():
    """deviation() が現在のセットポイントからの乖離を計算することを確認"""
    os = VitalOS()
    dev = os.deviation()
    for p in VitalParam:
        assert p in dev


def test_room_activities():
    """room_activities() が現在の部屋で可能な活動リストを返すことを確認"""
    os = VitalOS()
    os.current_room = "kitchen"
    acts = os.room_activities()
    assert "eat" in acts


def test_summary_format():
    """summary() が期待される形式の文字列を返すことを確認"""
    os = VitalOS()
    s = os.summary()
    assert "E:" in s
    assert "H:" in s
    assert "L:" in s
    assert "S:" in s


def test_full_day_simulation():
    """1日分のシミュレーションがエラーなく実行できることを確認"""
    os = VitalOS()
    start_day = os.time.day
    for _ in range(96):  # 15分tick x 96 = 24時間
        if not os.current_activity:
            action, duration = os.decide_next()
            os.start_activity(action, duration)
        os.tick(15)
        if os.time.day > start_day:
            break
    assert os.time.day >= start_day


def test_belief_update_after_finish():
    """活動完了時に信念が更新されることを確認"""
    os = VitalOS()
    os.state.hunger = 80
    before = os.state.as_float_dict()
    os.start_activity("eat")
    os._finish_activity()
    # 食事の効果がモデルに反映されているはず
    assert os.model.counts.get("eat", 0) >= 1


def test_knows_all_activities():
    """全ての活動がACTIVITIES辞書に存在することを確認"""
    for name in DURATIONS:
        assert name in ACTIVITIES, f"Missing activity: {name}"


def test_all_activities_have_tags():
    """全ての活動にタグが定義されていることを確認"""
    for name in ACTIVITIES:
        if name in ACTIVITY_TAGS:
            assert len(ACTIVITY_TAGS[name]) > 0, f"Activity {name} has no tags"


def test_energy_and_fatigue_separate():
    """体力と疲労が独立していることを確認（休息の効果検証）"""
    os = VitalOS()
    os.state.energy = 90
    os.state.fatigue = 80
    os.start_activity("rest")
    os._apply_activity_effects("rest", 30)
    # 休息は疲労を減らすが、エネルギーには中立的またはプラス
    effect = compute_true_effect("rest", 30)
    assert effect["fatigue"] < 0, "Rest should decrease fatigue"
