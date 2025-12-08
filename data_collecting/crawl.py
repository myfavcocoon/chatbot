from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
import json


def crawl_law(url: str):
    # Cấu hình trình duyệt headless
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    driver.get(url)
    time.sleep(5)  # đợi JS load xong

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Lấy toàn bộ <p>
    paragraphs = soup.find_all("p")
    text_content = "\n".join(
        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
    )

    if not text_content:
        raise ValueError("Không tìm thấy nội dung văn bản luật")

    title = soup.title.string.strip()
    return title, text_content


def split_law_structure(text: str):
    # Chuẩn hoá xuống dòng
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Tìm tất cả tiêu đề điều
    matches = list(re.finditer(r'(?m)^(Điều\s+(\d+)\..*)$', text))
    law_structure = []

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        num = int(m.group(2))
        if num > 218:
            break
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Không tách Khoản tại đây, để preprocess xử lý
        clause_list = [{"clause_text": body}]
        law_structure.append({
            "article_id": num,  # dùng số điều làm id
            "article_title": title,
            "clauses": clause_list
        })
    return law_structure


def save_json(title, law_structure, filename="law.json", filename_jsonl="law.jsonl"):
    # JSON dạng phân cấp
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"law_title": title, "articles": law_structure},
            f,
            ensure_ascii=False,
            indent=2
        )

    # JSONL dạng phẳng (mỗi dòng 1 chunk)
    with open(filename_jsonl, "w", encoding="utf-8") as f:
        for art in law_structure:
            for clause in art["clauses"]:
                record = {
                    "law_title": title,
                    "article_id": art["article_id"],
                    "article_title": art["article_title"],
                    "clause_text": clause["clause_text"],
                    "points": clause.get("points", [])
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Đã lưu {filename} và {filename_jsonl}")


if __name__ == "__main__":
    TARGET_URL = "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Van-ban-hop-nhat-67-VBHN-VPQH-2025-Luat-Doanh-nghiep-671127.aspx"

    title, raw_text = crawl_law(TARGET_URL)
    print("Tiêu đề:", title[:100], "...")

    structure = split_law_structure(raw_text)
    print("Số điều:", len(structure))

    save_json(title, structure,
              filename="luat_doanh_nghiep_2025.json",
              filename_jsonl="luat_doanh_nghiep_2025.jsonl")
