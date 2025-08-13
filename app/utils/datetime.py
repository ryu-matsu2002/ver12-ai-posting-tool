# app/utils/datetime.py
from datetime import timezone, timedelta, datetime

JST = timezone(timedelta(hours=9))

def to_jst(dt_utc_naive: datetime | None) -> datetime | None:
    """
    DBにUTC naive（tzinfo=None）で保存された日時をJSTに変換して返す。
    NoneのときはNoneを返す。
    """
    if dt_utc_naive is None:
        return None
    # UTC naive を UTC aware に仮置きしてから JST へ
    return dt_utc_naive.replace(tzinfo=timezone.utc).astimezone(JST)
