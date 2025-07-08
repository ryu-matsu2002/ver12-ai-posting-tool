"""
HTML フォームを GPT に渡して {label, selector} を抽出する
openai>=1.0 用の実装
"""

from __future__ import annotations
from typing import List, Dict
import os
from openai import OpenAI

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_form_fields(html: str, max_tokens: int = 700) -> List[Dict[str, str]]:
    snippet = html[:3000]
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
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content_to_dict()
