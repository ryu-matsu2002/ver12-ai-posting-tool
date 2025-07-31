import re
from typing import List

def _links_from_html(html: str) -> List[str]:
    """
    HTMLから http/https のリンクをすべて抽出して返す（順序維持）
    """
    return re.findall(r'https?://[^\s"\'<>]+', html)
