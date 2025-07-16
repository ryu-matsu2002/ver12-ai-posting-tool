# app/services/ai_executor.py

import os
import json
from openai import OpenAI

# ✅ OpenAI設定（.envから読み込み）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

async def ask_gpt_for_actions(html: str, goal: str, values: dict) -> list[dict]:
    """
    入力HTMLと目的(goal)をもとに、AIに対して「次にすべき操作」をJSON形式で聞く。
    values = {
        "email": "xxxx@example.com",
        "password": "abc123",
        "nickname": "ryu_test"
    }
    """

    # 値の説明をプロンプトに明示
    value_info = "\n".join([f"{k.upper()} = {v}" for k, v in values.items()])

    prompt = f"""
あなたはブラウザ操作用のAIアシスタントです。
以下のHTMLはあるWebページ（登録フォームなど）を表しています。
あなたの目標は「{goal}」を達成することです。

入力すべき項目や押すべきボタンを、次のJSON形式で出力してください：

[
  {{ "action": "fill", "selector": "#email", "value": "EMAIL" }},
  {{ "action": "fill", "selector": "#username", "value": "NICKNAME" }},
  {{ "action": "fill", "selector": "#password", "value": "PASSWORD" }},
  {{ "action": "click", "selector": "#submit" }}
]

※valueは EMAIL / PASSWORD / NICKNAME などのプレースホルダで書いてください。

--- 利用可能な値 ---
{value_info}

--- HTML START ---
{html}
--- HTML END ---
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)

    except Exception as e:
        raise RuntimeError(f"[AI-Executor] GPT呼び出し失敗: {str(e)}")
