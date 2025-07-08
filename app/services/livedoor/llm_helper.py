"""
HTML フォームを GPT に渡し {label, selector} を抽出するユーティリティ
openai>=1.0 対応版（コードブロック・整形文字も安全にパース）
"""
from __future__ import annotations
from typing import List, Dict
import os, json, re
from openai import OpenAI

MODEL  = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

def _extract_json(text: str) -> List[Dict[str, str]]:
    """
    GPT 返答から最初の JSON リストを安全に抽出する
    """
    # 1) ```json ... ``` があれば中身を優先
    if (m := _CODEBLOCK_RE.search(text)):
        text = m.group(1)

    # 2) 余計な行を削除して { [, ] } の最初〜最後を取り出す
    if "[" in text and "]" in text:
        text = text[text.index("[") : text.rindex("]") + 1]

    # 3) JSON パース
    return json.loads(text)

def extract_form_fields(html: str, max_tokens: int = 700) -> List[Dict[str, str]]:
    snippet = html[:3000]
    prompt  = f"""
あなたは HTML フォーム解析エージェントです。
以下 HTML から必須入力フィールドの label と CSS selector を JSON リストで抽出してください。
必ず **JSON だけ** を返してください。
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
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    return _extract_json(raw)
