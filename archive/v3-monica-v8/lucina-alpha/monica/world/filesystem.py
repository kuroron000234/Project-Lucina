"""ファイルシステム操作モジュール — モニカがPC内を探索するためのツール"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class FileSystem:
    """ファイル・ディレクトリ操作"""

    def __init__(self, allowed_root: Optional[str] = None):
        self.allowed_root = Path(allowed_root).resolve() if allowed_root else None

    def read(self, path: str, max_size: int = 100_000) -> dict:
        """ファイルを読む"""
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"success": False, "error": "ファイルが見つかりません"}
            if p.is_dir():
                return {"success": False, "error": "ディレクトリです"}
            if p.stat().st_size > max_size:
                return {"success": False, "error": f"ファイルが大きすぎます（{p.stat().st_size} bytes）"}

            content = p.read_text(encoding="utf-8", errors="replace")
            return {"success": True, "content": content, "path": str(p)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write(self, path: str, content: str) -> dict:
        """ファイルに書き込む"""
        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {"success": True, "path": str(p), "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_dir(self, path: str = ".") -> dict:
        """ディレクトリの中身をリスト"""
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"success": False, "error": "パスが見つかりません"}
            if not p.is_dir():
                return {"success": False, "error": "ディレクトリではありません"}

            items = []
            for entry in sorted(p.iterdir()):
                items.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else 0,
                    "modified": entry.stat().st_mtime,
                })
            return {"success": True, "items": items, "path": str(p)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, pattern: str, root: str = ".") -> dict:
        """ファイル名で検索"""
        try:
            p = self._resolve(root)
            matches = []
            for f in p.rglob(pattern):
                matches.append({
                    "path": str(f.relative_to(p)),
                    "type": "dir" if f.is_dir() else "file",
                    "size": f.stat().st_size if f.is_file() else 0,
                })
                if len(matches) >= 20:
                    break
            return {"success": True, "matches": matches, "count": len(matches)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete(self, path: str) -> dict:
        """ファイル/ディレクトリを削除"""
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"success": False, "error": "見つかりません"}
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            return {"success": True, "deleted": str(p)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def info(self, path: str) -> dict:
        """ファイル情報を取得"""
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"success": False, "error": "見つかりません"}
            return {
                "success": True,
                "path": str(p),
                "type": "dir" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else 0,
                "modified": p.stat().st_mtime,
                "permissions": oct(p.stat().st_mode)[-3:],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _resolve(self, path: str) -> Path:
        p = Path(path).resolve()
        if self.allowed_root and not str(p).startswith(str(self.allowed_root)):
            # 許可されたルート外ならルートに制限
            return self.allowed_root / Path(path).name
        return p


class Shell:
    """シェル実行（安全のためタイムアウト付き）"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def run(self, command: str) -> dict:
        """コマンドを実行"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            return {
                "success": result.returncode == 0,
                "output": output.strip() or "(出力なし)",
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"タイムアウト（{self.timeout}秒）"}
        except Exception as e:
            return {"success": False, "error": str(e)}
