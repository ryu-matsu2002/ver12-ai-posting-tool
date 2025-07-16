# app/utils/html_utils.py

from bs4 import BeautifulSoup

def extract_hidden_inputs(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    data = {}
    for input_tag in soup.find_all("input", type="hidden"):
        name = input_tag.get("name")
        value = input_tag.get("value", "")
        if name:
            data[name] = value
    return data
