#!/usr/bin/env python3
"""モニカ — デスクトップに棲む自律会話エージェント

Usage:
    python -m monica              # デフォルトモデルで起動
    python -m monica --model qwen3.5:9b  # モデル指定
    python -m monica --help       # ヘルプ
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="🎀 モニカ — デスクトップに棲む文学少女",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python -m monica                     # デフォルトで起動
  python -m monica --model qwen3.5:9b  # モデルを指定
  python -m monica --model ""          # モデル一覧を表示
        """,
    )
    parser.add_argument(
        "--model", "-m",
        default="qwen3.6:35b",
        help="Ollamaモデル名 (default: qwen3.6:35b)",
    )
    args = parser.parse_args()

    # Ollamaの接続確認
    from monica.brain.ollama import OllamaBrain, OllamaConfig
    brain = OllamaBrain(OllamaConfig(model=args.model))

    if not brain.is_available():
        print()
        print("  ⚠️  Ollama に接続できません")
        print("  `ollama serve` が実行されているか確認してください")
        print()
        sys.exit(1)

    # モデルが存在するか確認
    if args.model:
        models = brain.list_models()
        if args.model not in models and not any(m.startswith(args.model) for m in models):
            print()
            print(f"  ⚠️  モデル '{args.model}' が見つかりません")
            print(f"  `ollama pull {args.model}` でダウンロードしてください")
            print(f"  利用可能: {', '.join(models[:5])}...")
            print()
            sys.exit(1)

    # モニカ起動
    from monica.core.monica import Monica
    from monica.cli.repl import REPL

    monica = Monica(model=args.model)
    repl = REPL(monica)

    try:
        repl.run()
    except Exception as e:
        print()
        print(f"  ❌ エラーが発生しました: {e}")
        print()


if __name__ == "__main__":
    main()
