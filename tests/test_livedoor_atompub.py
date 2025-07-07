# tests/test_livedoor_atompub.py
import os, pytest, uuid
from app.services.livedoor_atompub import LivedoorClient

@pytest.mark.skipif(
    not os.getenv("LD_BLOG") or not os.getenv("LD_ID"),
    reason="livedoor credentials not set"
)
def test_new_post():
    client = LivedoorClient(
        blog_name=os.environ["LD_BLOG"],
        username=os.environ["LD_ID"],
        apikey=os.environ["LD_APIKEY"]
    )
    url = client.new_post(
        title="テスト投稿 "+uuid.uuid4().hex[:6],
        html="<p>pytest からのテスト投稿</p>",
        categories=["テスト"]
    )
    assert url.startswith("http://")
