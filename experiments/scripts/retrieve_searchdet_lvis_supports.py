"""Cache web supports for the fixed SearchDet LVIS pilot.

This is a thin batch wrapper around the released SearchDet retrieval policy:
headless Chrome searches Bing Images after Google CAPTCHA blocked automation and stores five positive and five
negative exemplars for each predeclared class.  It records every successful
file, making later inference independent of a live search-result ordering.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def query_text(label: str) -> str:
    return label.replace("_", " ").replace("(", " ").replace(")", " ").strip()


def build_driver(driver_path: str | None) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,1600")
    options.add_argument("--disable-gpu")
    service = Service(executable_path=driver_path) if driver_path else None
    return webdriver.Chrome(service=service, options=options)


def download_query(driver: webdriver.Chrome, term: str, directory: Path, count: int) -> list[str]:
    """Retrieve web supports via Bing Images after Google blocked automation."""
    directory.mkdir(parents=True, exist_ok=True)
    driver.get("https://www.bing.com/images/search?q=" + requests.utils.quote(term) + "&form=HDRSC3")
    time.sleep(3)
    for _ in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
    urls: list[str] = []
    for node in driver.find_elements(By.CSS_SELECTOR, "a.iusc"):
        try:
            metadata = json.loads(node.get_attribute("m"))
            source = metadata.get("murl") or metadata.get("turl")
            if source and source.startswith("http") and source not in urls:
                urls.append(source)
        except (TypeError, ValueError):
            continue
    written: list[str] = []
    for source in urls:
        if len(written) >= count:
            break
        file_path = directory / f"image_{len(written)+1:02d}.jpg"
        try:
            response = requests.get(source, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            file_path = directory / f"image_{len(written)+1:02d}.jpg"
            file_path.write_bytes(response.content)
            with Image.open(file_path) as image:
                image.convert("RGB").save(file_path, quality=95)
            written.append(str(file_path))
        except Exception:
            file_path.unlink(missing_ok=True)
    return written

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-query", type=int, default=5)
    parser.add_argument("--chromedriver", default=None)
    args = parser.parse_args()
    pilot = json.loads(Path(args.pilot_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source_manifest": str(Path(args.pilot_manifest).resolve()), "per_query": args.per_query, "retrieval_backend": "bing_images_adapter_after_google_captcha", "classes": []}
    driver = build_driver(args.chromedriver)
    try:
        for category in pilot["selected_categories"]:
            label = category["category_name"]
            category_dir = output_dir / safe_name(label)
            pos_dir, neg_dir = category_dir / "positive", category_dir / "negative"
            existing_pos = sorted(str(path) for path in pos_dir.glob("*.jpg"))
            existing_neg = sorted(str(path) for path in neg_dir.glob("*.jpg"))
            positive = existing_pos[:args.per_query] if len(existing_pos) >= args.per_query else download_query(driver, query_text(label), pos_dir, args.per_query)
            negative = existing_neg[:args.per_query] if len(existing_neg) >= args.per_query else download_query(driver, query_text(category["negative_keyword"]), neg_dir, args.per_query)
            record = {**category, "positive_files": positive, "negative_files": negative}
            report["classes"].append(record)
            print(f"{label}: positive={len(positive)} negative={len(negative)}", flush=True)
    finally:
        driver.quit()
    (output_dir / "support_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = [item["category_name"] for item in report["classes"] if len(item["positive_files"]) < args.per_query or len(item["negative_files"]) < args.per_query]
    print(f"completed classes={len(report['classes'])}; incomplete={missing}")


if __name__ == "__main__":
    main()
