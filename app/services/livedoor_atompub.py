# app/services/livedoor_atompub.py
import base64, requests, uuid, datetime as dt
from xml.etree import ElementTree as ET
from slugify import slugify
from flask import current_app

# ---------- config ----------
API_BASE_FMT  = "https://livedoor.blogcms.jp/atompub/{blog}/"
HEADERS_ENTRY = {"Content-Type": "application/atom+xml;type=entry;charset=utf-8"}
HEADERS_IMAGE = {"Content-Type": "image/jpeg"}         # png/gif は呼び出し側で変える

class LivedoorClient:
    def __init__(self, blog_name: str, username: str, apikey: str):
        self.blog     = blog_name
        self.endpoint = API_BASE_FMT.format(blog=blog_name)
        self.session  = requests.Session()
        self.session.auth  = (username, apikey)
        self.timeout  = getattr(current_app, "config", {}).get(
            "LIVEDOOR_API_TIMEOUT", 15
        )

    # ---- public ----------------------------------------------------
    def new_post(self, title: str, html: str,
                 categories: list[str] | None = None, draft: bool = False) -> str:
        xml = self._build_entry_xml(title, html, categories, draft)
        url = self.endpoint + "article"
        res = self.session.post(url, data=xml.encode("utf-8"),
                                headers=HEADERS_ENTRY, timeout=self.timeout)
        res.raise_for_status()
        # 投稿 URL は <link rel="alternate" …> から取得
        root = ET.fromstring(res.text)
        link = root.find(".//{http://www.w3.org/2005/Atom}link[@rel='alternate']")
        return link.attrib["href"] if link is not None else ""

    def new_image(self, name: str, data: bytes, mime: str = "image/jpeg") -> str:
        url = self.endpoint + "image"
        headers = {"Content-Type": mime}
        res = self.session.post(url, data=data, headers=headers,
                                timeout=self.timeout)
        res.raise_for_status()
        root = ET.fromstring(res.text)
        link = root.find(".//{http://www.w3.org/2005/Atom}link[@rel='edit-media']")
        return link.attrib["href"] if link is not None else ""

    # ---- private ---------------------------------------------------
    def _build_entry_xml(self, title: str, html: str,
                         categories: list[str] | None, draft: bool) -> str:
        ns   = "http://www.w3.org/2005/Atom"
        app  = "http://purl.org/atom/app#"
        ET.register_namespace("", ns)
        ET.register_namespace("app", app)

        entry      = ET.Element(ET.QName(ns, "entry"))
        ET.SubElement(entry, ET.QName(ns, "title")).text = title
        ET.SubElement(entry, ET.QName(ns, "updated")).text = dt.datetime.utcnow().isoformat()+"Z"
        ET.SubElement(entry, ET.QName(ns, "content"),
                      {"type": "html"}).text = html
        if categories:
            for c in categories:
                ET.SubElement(entry, ET.QName(ns, "category"),
                              {"term": c})
        if draft:
            control = ET.SubElement(entry, ET.QName(app, "control"))
            ET.SubElement(control, ET.QName(app, "draft")).text = "yes"
        return ET.tostring(entry, encoding="utf-8", xml_declaration=True).decode()
