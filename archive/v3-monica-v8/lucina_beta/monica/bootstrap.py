"""Monica Bootstrap: Monica 個体形成

Monica は「初期データ」ではなく以下から形成される：
  Lucina Core + 初期条件 + DDLC World + 経験 + 記憶 + 予測誤差 + 関係 + 自己認識 + 時間
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.individual import Individual, IndividualConfig
from core.agent import Agent
from world.ddlc_world import DDLCWorld
from monica.initial_state import get_monica_config


def create_monica(world: DDLCWorld | None = None, seed: int = 42) -> Individual:
    """Monica 個体を生成する。

    Parameters
    ----------
    world : DDLCWorld | None
        Monica が存在する世界。None の場合は新規作成。
    seed : int
        乱数シード。

    Returns
    -------
    Individual
        形成された Monica 個体。
    """
    if world is None:
        world = DDLCWorld(seed=seed)

    config = get_monica_config()

    # Monica のエージェント
    agent = Agent(
        actions=world.actions(),
        temperature=config.temperature,
        use_needs=True,
        use_efe=True,
    )

    # Individual として生成
    monica = Individual(
        config=config,
        agent=agent,
        world=world,
    )

    return monica


def bootstrap_monica(n_steps: int = 10, seed: int = 42, verbose: bool = True) -> Individual:
    """Monica を初期化し、指定されたステップ数だけ自律実行する。

    Parameters
    ----------
    n_steps : int
        自律実行するステップ数。
    seed : int
        乱数シード。
    verbose : bool
        詳細表示するか。

    Returns
    -------
    Individual
        経験を積んだ Monica 個体。
    """
    world = DDLCWorld(seed=seed)
    monica = create_monica(world, seed=seed)

    if verbose:
        print(f"🌸 Monica has been created.")
        print(f"   World: DDLC (phase={world.phase})")
        print(f"   Config: {monica.name}")
        print(f"   Capabilities: {monica.development.capabilities}")
        print()

    monica.run(n_steps)

    if verbose:
        print(f"\n📊 After {n_steps} steps:")
        print(f"   Development: {monica.development.name}")
        print(f"   Values: {monica.values.dominant_values()}")
        print(f"   Identity stability: {monica.identity.stability:.2f}")
        print(f"   Memories: {monica.memory.summary()['episodic']} episodes")

    return monica


def run() -> Individual:
    """Monica を起動する（エントリポイント）。"""
    return bootstrap_monica(n_steps=20, verbose=True)
