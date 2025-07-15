# app/services/agent/field_extractor.py

import logging
# app/services/agent/field_extractor.py
from app.utils.openai_client import ask_gpt_json
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FIELD_PROMPT_TEMPLATE = """
以下のHTMLは、ユーザー登録ページです。

あなたはAIエージェントとして、人間が登録フォームを記入するように、
このHTMLの中から以下の項目を正確に特定してください：

1. メールアドレス入力欄（selectorと説明）
2. パスワード入力欄（2つある場合は確認用も）
3. ユーザー名 or ニックネーム入力欄（ある場合のみ）
4. CAPTCHA画像のセレクタ（ある場合のみ）
5. 登録ボタン or 確認ボタン（クリックに使用するセレクタ）

出力形式は次のようにしてください（JSON）:
{
  "email": "input[name='email']",
  "password": "input[name='password']",
  "password2": "input[name='password2']",
  "nickname": "input[name='livedoor_id']",
  "captcha": "img[src*='captcha']",
  "submit": "button[type='submit']"
}
入力できない項目は null にしてください。
"""

def extract_form_fields(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    short_html = soup.prettify()[:12000]  # トークン制限対策

    prompt = FIELD_PROMPT_TEMPLATE + "\n\nHTML:\n" + short_html

    try:
        result = ask_gpt_json(prompt)
        logger.info("✅ GPTによるフィールド推論結果: %s", result)
        return result
    except Exception as e:
        logger.error("❌ GPTフィールド推論に失敗: %s", str(e))
        return {}

