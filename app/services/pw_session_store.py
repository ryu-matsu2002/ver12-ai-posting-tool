# app/services/pw_session_store.py
"""
pw_session_store: Playwright のセッションID(sid)をキーに、
Livedoor サインアップで使う資格情報や補助情報を /tmp/captcha_sessions に保存・復元する小さなユーティリティ。
- JSON1ファイル/sid で管理
- 既存の storage_state と同じディレクトリを使う（疎結合）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

# storage_state と同じディレクトリを使う（pw_controller と合わせる）
_BASE = Path("/tmp/captcha_sessions")
_BASE.mkdir(parents=True, exist_ok=True)


def _path(sid: str) -> Path:
    """sid に対応する JSON のパス"""
    return _BASE / f"{sid}.meta.json"


def load(sid: str) -> Dict[str, Any]:
    """
    sid にひもづくメタ情報(JSON)を読み込む。
    例外やファイル未存在時は空 dict を返す。
    """
    p = _path(sid)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # 壊れていても呼び出し側でリカバリできるよう空で返す
        return {}


def save(sid: str, **kv) -> None:
    """
    部分更新：既存 JSON を読み込み、与えられたキーのみ上書きして保存。
    None は無視（上書きしない）。
    """
    data = load(sid)
    for k, v in kv.items():
        if v is not None:
            data[k] = v
    _path(sid).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_cred(sid: str) -> Optional[Dict[str, str]]:
    """
    サインアップに必要な資格情報セットを返す。
    3つとも揃っていなければ None。
    """
    data = load(sid)
    email = data.get("email")
    password = data.get("password")
    livedoor_id = data.get("livedoor_id")
    if email and password and livedoor_id:
        return {"email": email, "password": password, "livedoor_id": livedoor_id}
    return None


def clear(sid: str) -> None:
    """メタ情報 JSON を削除（後始末用。storage_state 本体は触らない）。"""
    p = _path(sid)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        # 失敗は致命ではないので握りつぶす
        pass
