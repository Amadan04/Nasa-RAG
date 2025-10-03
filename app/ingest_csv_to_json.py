import os, re, json, time, requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
CSV_URL_COLUMN = os.getenv("CSV_URL_COLUMN", "URL")

IN_CSV  = os.path.join("data", "nasa_links.csv")
OUT_JSON = os.path.join("data", "nasa_bio.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (RAG SpaceBio Bot; +https://example.com)"
}

def clean_text(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def extract_article_text(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = (soup.title.string or "").strip() if soup.title else "Unknown"
    # grab main paragraphs
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = clean_text(" ".join(paras))
    return {"url": url, "title": title, "text": text}

def main(limit: int | None = None):
    df = pd.read_csv(IN_CSV)
    urls = df[CSV_URL_COLUMN].dropna().tolist()
    if limit:
        urls = urls[:limit]

    out = []
    for i, url in enumerate(urls, 1):
        try:
            art = extract_article_text(url)
            if len(art["text"]) < 300:
                print(f"[warn] very short text for {url}")
            out.append(art)
            print(f"[{i}/{len(urls)}] scraped: {url}")
            time.sleep(0.5)  # be gentle
        except Exception as e:
            print(f"[error] {url}: {e}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ saved {len(out)} docs → {OUT_JSON}")

if __name__ == "__main__":
    # tip: start with a small limit to test (e.g., 30)
    main(limit=None)
