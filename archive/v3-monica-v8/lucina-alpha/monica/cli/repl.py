"""CLIインターフェース — ターミナルでモニカと話す"""

import shutil
import sys
import time
import threading
from datetime import datetime
from typing import Optional

from monica.core.monica import Monica


class REPL:
    """Read-Eval-Print Loop — モニカとの対話インターフェース"""

    def __init__(self, monica: Monica):
        self.monica = monica
        self.running = True
        self._autonomy_thread: Optional[threading.Thread] = None

    def run(self):
        """メインループ（自律スレッド付き）"""
        self._print_header()
        self._print_welcome()

        # Phase 6: 自律ループスレッドを開始
        self._start_autonomy_thread()

        while self.running:
            try:
                user_input = self._get_input()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                self._show_thinking()
                response = self.monica.think(user_input)
                self._clear_thinking()

                if response:
                    self._print_monica(response)

            except KeyboardInterrupt:
                print()
                self._handle_command("/quit")
                break
            except EOFError:
                print()
                self._handle_command("/quit")
                break

    def _start_autonomy_thread(self):
        """自律ループをバックグラウンドで開始"""
        def loop():
            while self.running:
                try:
                    msg = self.monica.autonomous_step()
                    if msg:
                        self._print_monica_auto(msg)
                except Exception:
                    pass
                time.sleep(15)  # 15秒ごとにチェック
        self._autonomy_thread = threading.Thread(target=loop, daemon=True)
        self._autonomy_thread.start()

    # ── 表示 ──

    def _print_header(self):
        width = shutil.get_terminal_size().columns
        title = "🎀  モニカ  —  デスクトップに棲む文学少女  🎀"
        print()
        print("=" * min(width, 60))
        print(f"{title:^{min(width, 60)}}")
        print("=" * min(width, 60))
        print(f"  モデル: {self.monica.brain.config.model}")
        print(f"  /help でコマンド一覧")
        print("=" * min(width, 60))
        print()

    def _print_welcome(self):
        greeting = self.monica.greet()
        self._print_monica(greeting)
        self.monica.memory.add_message("assistant", greeting, metadata={"type": "greeting"})

    def _print_monica(self, text: str):
        timestamp = datetime.now().strftime("%H:%M")
        print()
        print(f"  ┌─ モニカ [{timestamp}]")
        for line in text.strip().split("\n"):
            print(f"  │ {line}")
        print(f"  └─")
        print()

    def _print_monica_auto(self, text: str):
        """自律発言を表示（自発マーク付き）"""
        timestamp = datetime.now().strftime("%H:%M")
        print()
        print(f"  ┌─ モニカ 💭 [{timestamp}]")
        for line in text.strip().split("\n"):
            print(f"  │ {line}")
        print(f"  └─")
        print()

    def _show_thinking(self):
        print(f"  ┌─ モニカ [考え中...]", end="\r")
        sys.stdout.flush()

    def _clear_thinking(self):
        print("\033[K", end="\r", flush=True)

    def _get_input(self) -> Optional[str]:
        try:
            prompt = "  You > "
            text = input(prompt)
            return text.strip()
        except (KeyboardInterrupt, EOFError):
            raise

    # ── コマンド ──

    def _handle_command(self, cmd: str):
        cmd = cmd.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            goodbye = self.monica.goodbye()
            self._print_monica(goodbye)
            self.monica.memory.add_message("assistant", goodbye, metadata={"type": "goodbye"})
            self.running = False

        elif cmd in ("/help", "/h"):
            print("  📖 コマンド一覧")
            print("    /quit, /exit, /q  — 終了")
            print("    /help, /h          — このヘルプ")
            print("    /clear, /c         — 会話履歴をクリア")
            print("    /stats, /s         — ステータス表示")
            print("    /model, /m         — モデル一覧")
            print("    /model <名前>      — モデル変更")
            print("    /auto              — 自律発言の有無を確認")
            print()

        elif cmd in ("/clear", "/c"):
            self.monica.memory.clear()
            print("  📝 会話履歴をクリアしました")
            print()

        elif cmd in ("/stats", "/s"):
            print(f"  📊 モニカ ステータス")
            print(f"  {self.monica.status_string}")
            print()

        elif cmd in ("/model", "/m"):
            models = self.monica.brain.list_models()
            if models:
                print(f"  利用可能なモデル:")
                for m in models:
                    marker = " ◀ 現在" if m.startswith(self.monica.brain.config.model) else ""
                    print(f"    • {m}{marker}")
            else:
                print("  Ollamaに接続できませんでした")
            print()

        elif cmd.startswith("/model ") or cmd.startswith("/m "):
            new_model = cmd.split(" ", 1)[1].strip()
            self.monica.brain.config.model = new_model
            print(f"  モデルを {new_model} に変更しました")
            print()

        elif cmd in ("/auto", "/a"):
            print(f"  自律ループ: {'稼働中' if self.running else '停止中'}")
            print(f"  前回の自発行動: {self.monica._last_autonomous_action}")
            print()

        else:
            print(f"  不明なコマンド: {cmd}")
            print("  /help でコマンド一覧を表示")
            print()
