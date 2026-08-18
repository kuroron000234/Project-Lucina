"""
OpencodeBridge のセッション管理テスト (v4.1.2)

- セッションID発見 (_discover_session_id) が最新の lucina-daemon セッションを
  正しく選ぶこと
- 保存済みセッションが無効な場合にリトライされること
"""

import json

from core.agent.opencode_bridge import OpencodeBridge, PROJECT_ROOT


def _fake_session(sid: str, title: str, updated: int, directory: str = str(PROJECT_ROOT)) -> dict:
    return {
        "id": sid,
        "title": title,
        "updated": updated,
        "created": updated - 100,
        "projectId": "global",
        "directory": directory,
    }


class TestDiscoverSessionId:
    def test_picks_latest_lucina_daemon_session(self, monkeypatch):
        """最新の lucina-daemon セッションが選ばれる"""
        sessions = [
            _fake_session("ses_old1", "lucina-daemon", updated=100),
            _fake_session("ses_old2", "lucina-daemon", updated=200),
            _fake_session("ses_new", "lucina-daemon", updated=300),
        ]

        def fake_run(cmd, **kwargs):
            class R:
                returncode = 0
                stdout = json.dumps(sessions)
                stderr = ""
            return R()

        monkeypatch.setattr("core.agent.opencode_bridge.subprocess.run", fake_run)
        bridge = OpencodeBridge()
        sid = bridge._discover_session_id()
        assert sid == "ses_new"

    def test_ignores_other_titles_and_dirs(self, monkeypatch):
        """別タイトル・別ディレクトリのセッションは無視する"""
        sessions = [
            _fake_session("ses_other_title", "personal", updated=999),
            _fake_session("ses_other_dir", "lucina-daemon", updated=999,
                          directory="/other/project"),
            _fake_session("ses_mine", "lucina-daemon", updated=500),
        ]

        def fake_run(cmd, **kwargs):
            class R:
                returncode = 0
                stdout = json.dumps(sessions)
                stderr = ""
            return R()

        monkeypatch.setattr("core.agent.opencode_bridge.subprocess.run", fake_run)
        bridge = OpencodeBridge()
        sid = bridge._discover_session_id()
        assert sid == "ses_mine"

    def test_returns_none_on_empty(self, monkeypatch):
        """候補が無い場合は None を返す"""

        def fake_run(cmd, **kwargs):
            class R:
                returncode = 0
                stdout = json.dumps([])
                stderr = ""
            return R()

        monkeypatch.setattr("core.agent.opencode_bridge.subprocess.run", fake_run)
        bridge = OpencodeBridge()
        assert bridge._discover_session_id() is None

    def test_returns_none_on_error(self, monkeypatch):
        """コマンド失敗時は None を返す"""

        def fake_run(cmd, **kwargs):
            class R:
                returncode = 1
                stdout = ""
                stderr = "error"
            return R()

        monkeypatch.setattr("core.agent.opencode_bridge.subprocess.run", fake_run)
        bridge = OpencodeBridge()
        assert bridge._discover_session_id() is None


class TestInvalidSessionRetry:
    def test_build_cmd_includes_session(self):
        """セッションID保持時は --session が付与される"""
        bridge = OpencodeBridge()
        bridge._session_id = "ses_test123"
        cmd = bridge._build_cmd("hello")
        assert "--session" in cmd
        assert "ses_test123" in cmd

    def test_successful_run_saves_discovered_session(self, monkeypatch, tmp_path):
        """
        v4.1.2: 成功した run() は発見したセッションIDを保存し、
        次回の run() がそのセッションを再利用する（プロリファレーション防止）。
        """
        calls = {"n": 0}
        cmds = []

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            cmds.append(cmd)
            if cmd[:2] == ["opencode", "session"]:
                # セッション一覧を返す
                class R:
                    returncode = 0
                    stdout = json.dumps([
                        _fake_session("ses_found", "lucina-daemon", updated=500),
                    ])
                    stderr = ""
                return R()
            class R:
                returncode = 0
                stdout = "task done"
                stderr = ""
            return R()

        monkeypatch.setattr("core.agent.opencode_bridge.subprocess.run", fake_run)
        monkeypatch.setattr(
            OpencodeBridge, "health_check", lambda self, force=False: True,
        )
        monkeypatch.setattr(
            "core.agent.opencode_bridge.LUCINA_SESSION_ID_FILE",
            tmp_path / ".lucina_session",
        )
        bridge = OpencodeBridge()

        # 1回目: セッション無しで実行 → 発見して保存
        r1 = bridge.run("first task")
        assert r1["success"] is True
        assert bridge._session_id == "ses_found"
        assert (tmp_path / ".lucina_session").read_text().strip() == "ses_found"

        # 2回目: 保存済みセッションを再利用（--session 付き）
        calls["n"] = 0
        r2 = bridge.run("second task")
        assert r2["success"] is True
        assert calls["n"] == 1  # セッション一覧取得は走らない
        assert "--session" in cmds[-1]
        assert "ses_found" in cmds[-1]

    def test_retry_clears_invalid_session(self, monkeypatch, tmp_path):
        """
        保存済みセッションで失敗した場合、セッションをリセットして
        1回リトライする（run 本体は2回、--session 有無が切り替わる）。
        """
        calls = {"n": 0}
        cmds = []

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            cmds.append(cmd)
            has_session = "--session" in cmd
            class R:
                returncode = 0 if not has_session else 1
                stdout = "OK" if not has_session else ""
                stderr = "" if not has_session else "session not found"
            return R()

        monkeypatch.setattr("core.agent.opencode_bridge.subprocess.run", fake_run)
        # health_check と discover は run() 内部でも subprocess.run を呼ぶため
        # 別途パッチして run 本体の実行回数だけを数える。
        monkeypatch.setattr(
            OpencodeBridge, "health_check",
            lambda self, force=False: True,
        )
        monkeypatch.setattr(
            OpencodeBridge, "_discover_session_id",
            lambda self: None,
        )
        monkeypatch.setattr(
            "core.agent.opencode_bridge.LUCINA_SESSION_ID_FILE",
            tmp_path / ".lucina_session",
        )
        bridge = OpencodeBridge()
        bridge._session_id = "ses_dead"
        result = bridge.run("test task")
        # 1回目(失敗) + リトライ(成功) = run は2回
        assert calls["n"] == 2
        assert "--session" in cmds[0]      # 1回目は保存済みセッションを使う
        assert "--session" not in cmds[1]  # リトライは新規セッション
        assert result["success"] is True
        assert bridge._session_id is None  # discover は None のため未保存
