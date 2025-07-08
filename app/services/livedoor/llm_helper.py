"""
HTML フォームを GPT に渡し {label, selector} を抽出するユーティリティ
openai>=1.0 対応版
"""
from __future__ import annotations
from typing import List, Dict
import os, json
from openai import OpenAI

MODEL  = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_form_fields(html: str, max_tokens: int = 700) -> List[Dict[str, str]]:
    snippet = html[:3000]
    prompt  = f"""
あなたは HTML フォーム解析エージェントです。
以下 HTML から必須入力フィールドの label と CSS selector を
JSON リストで抽出してください。
-----
{snippet}
-----
例:
[{{"label":"メールアドレス","selector":"input[name='mail']"}}, …]
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    # ----★ ここが変更点 ★----
    content = resp.choices[0].message.content
    return json.loads(content)
