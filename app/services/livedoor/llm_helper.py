"""
Playwright page.content() の HTML を GPT-4o に投げ
  [{"label": "...", "selector": "..."}] を返すユーティリティ
"""

from __future__ import annotations
from typing import List, Dict
import openai

MODEL = "gpt-4o-mini"

def extract_form_fields(html: str, max_tokens: int = 700) -> List[Dict[str, str]]:
    """
    Parameters
    ----------
    html : str
        Playwright page.content() の HTML
    Returns
    -------
    list[dict]
        e.g. [{"label":"メールアドレス","selector":"input[name='mail']"}, ...]
    """
    snippet = html[:3000]          # サイズ削減
    prompt = f"""
あなたは HTML フォーム解析エージェントです。
以下の HTML から「必須入力フィールド」の label と CSS selector を JSON リストで抽出してください。
出力例:
[
  {{"label":"メールアドレス","selector":"input[name='mail']"}},
  ...
]
HTML---
{snippet}
---HTML
"""
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.to_py()
