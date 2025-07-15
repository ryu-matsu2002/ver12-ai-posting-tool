# app/utils/openai_client.py

from app.article_generator import _chat  # ✅ 既存の信頼された関数を再利用
import json, logging

logger = logging.getLogger(__name__)

def ask_gpt_json(prompt: str, user_id: int = None) -> dict:
    """
    gpt-4o-mini にフォーム推論プロンプトを送信し、JSON形式で応答を受け取る。
    article_generator.py の _chat() を再利用。
    """
    try:
        raw = _chat(
            msgs=[
                {"role": "system", "content": "あなたはHTML構造を分析し、フォーム構成を特定するAIです。出力は必ずJSON形式で。"},
                {"role": "user", "content": prompt}
            ],
            max_t=800,  # 安全に制限（トークン管理も _chat が内部で行う）
            temp=0.2,
            user_id=user_id
        )
        return json.loads(raw)
    except Exception as e:
        logger.error(f"❌ GPTフォーム解析エラー: {e}")
        return {}
