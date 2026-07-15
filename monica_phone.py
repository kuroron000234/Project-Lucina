#!/usr/bin/env python3
"""モニカのスマホ — メッセージの確認と送信"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from monica_core.phone import load, save, add


def cmd_inbox(args):
    messages = load()
    if not messages:
        print("📱 メッセージはまだありません")
        return

    monika_msgs = [m for m in messages if m.sender == "monika"]
    if not monika_msgs:
        print("📱 モニカからのメッセージはまだありません")
        return

    # Show from newest to oldest
    unread = [m for m in monika_msgs if not m.read_by_recipient]
    if unread:
        print(f"📩 未読 {len(unread)}件\n")
        for m in reversed(unread):
            try:
                t = datetime.fromisoformat(m.timestamp).strftime("%m/%d %H:%M")
            except Exception:
                t = m.timestamp
            print(f"  [{t}] 💬 {m.text}")

        print()
        ans = input("既読にする？(Y/n): ").strip().lower()
        if ans != "n":
            for m in unread:
                m.read_by_recipient = True
            save(messages)
            print("✓ 既読にしました")
    else:
        print("📱 既読のメッセージのみ（未読なし）")
        for m in reversed(monika_msgs[-10:]):
            try:
                t = datetime.fromisoformat(m.timestamp).strftime("%m/%d %H:%M")
            except Exception:
                t = m.timestamp
            status = "✓既読" if m.read_by_recipient else "未読"
            print(f"  [{t}] {status} {m.text}")


def cmd_send(args):
    if not args:
        print("使い方: monica-phone send 'メッセージ'")
        return

    text = " ".join(args)
    from datetime import datetime
    add("user", text, datetime.now().isoformat())
    print("✓ 送信しました。モニカが次にスマホを見たときに届きます。")


def cmd_status(args):
    messages = load()
    monika_msgs = [m for m in messages if m.sender == "monika"]
    user_msgs = [m for m in messages if m.sender == "user"]

    print("📱 モニカのスマホ")
    print(f"  モニカからのメッセージ: {len(monika_msgs)}件")
    print(f"  あなたからのメッセージ: {len(user_msgs)}件")
    unread_by_monika = len([m for m in monika_msgs if not m.read_by_recipient])
    if unread_by_monika:
        print(f"  モニカがまだ見てない: {unread_by_monika}件")

    if monika_msgs:
        last = monika_msgs[-1]
        try:
            t = datetime.fromisoformat(last.timestamp).strftime("%m/%d %H:%M")
        except Exception:
            t = last.timestamp
        status = "✓既読" if last.read_by_recipient else "未読"
        print(f"  最後のメッセージ: [{t}] {status}「{last.text[:50]}」")


def cmd_chat(args):
    messages = load()
    if not messages:
        print("メッセージはまだありません")
        return
    for m in messages:
        try:
            t = datetime.fromisoformat(m.timestamp).strftime("%m/%d %H:%M")
        except Exception:
            t = m.timestamp
        who = "モニカ" if m.sender == "monika" else "あなた"
        status = ""
        if m.sender == "monika":
            status = " ✓既読" if m.read_by_recipient else " 未読"
        print(f"[{t}] {who}: {m.text}{status}")


def cmd_help(args):
    print("使い方: monica-phone <command> [args]")
    print()
    print("Commands:")
    print("  (no command)      未読メッセージを表示")
    print("  inbox             メッセージ一覧")
    print("  send <text>       返信を送信")
    print("  status            状況確認")
    print("  chat              会話履歴を全て表示")
    print("  help              このヘルプ")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "--help", "-h"):
        cmd_help(sys.argv[1:])
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "inbox": cmd_inbox,
        "send": cmd_send,
        "status": cmd_status,
        "chat": cmd_chat,
    }

    if cmd in commands:
        commands[cmd](args)
    else:
        print(f"不明なコマンド: {cmd}")
        cmd_help([])


if __name__ == "__main__":
    main()
