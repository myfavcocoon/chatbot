from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
import json
import os


# ============================
# REMOVE CHƯƠNG / PHỤ LỤC / MỤC LỤC
# ============================
def remove_trash(text: str):
    """Xóa hoàn toàn CHƯƠNG / PHỤ LỤC / MỤC LỤC trong nội dung crawl."""
    if not text:
        return text

    patterns = [
        r"Chương\s+[IVXLCDM]+\b.*",
        r"CHƯƠNG\s+[IVXLCDM]+\b.*",
        r"Chương\s+\d+\b.*",
        r"CHƯƠNG\s+\d+\b.*",
        r"Phụ lục\b.*",
        r"PHỤ LỤC\b.*",
        r"Mục lục\b.*",
        r"MỤC LỤC\b.*",
    ]

    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    return text.strip()


# ============================
# CRAWL HTML
# ============================
def crawl_law(url: str):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    driver.get(url)
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    paragraphs = soup.find_all("p")

    text_content = "\n".join(
        re.sub(r"\s+", " ", p.get_text(separator=" ", strip=True))
        for p in paragraphs if p.get_text(strip=True)
    )

    if not text_content:
        raise ValueError(f"Không tìm thấy nội dung văn bản luật ở {url}")

    title = soup.title.string.strip()
    return title, text_content


# ============================
# TÁCH ĐIỀU – KHOẢN – ĐIỂM
# ============================
def split_law_structure(text: str, base_url: str):
    law_structure = []

    # Tách Điều thật (không lấy Điều trong câu)
    articles = re.split(r"(?:(?<=\n)|^)(?=Điều\s+\d+(?:\s*[-–]?\s*[A-Z]?)\b)", text, flags=re.S)
    articles = [a.strip() for a in articles if a.strip()]
    articles = [a for a in articles if re.match(r"^Điều\s+\d+", a)]

    for idx, article in enumerate(articles, start=1):

        # Tiêu đề Điều
        m = re.match(r"^(Điều\s+\d+(?:\s*[-–]?\s*[A-Z]?)?)", article)
        if m:
            full_title = m.group(1).strip()
            body = article[len(m.group(0)):].strip()
        else:
            full_title = f"Điều {idx}"
            body = article.strip()

        # Số Điều
        num = re.findall(r"\d+", full_title)
        article_num = num[0] if num else str(idx)
        article_title = f"Điều {article_num}"
        article_link = f"{base_url}#dieu_{article_num}"

        # XÓA CHƯƠNG trong phần thân điều
        body = remove_trash(body)

        # Tách Khoản
        clauses = re.split(
            r"(?:(?<=\n)|^)(Khoản\s+\d+(?:\s*[A-Z]?)\s.*?)(?=(?:(?<=\n)|^)Khoản\s+\d+|$)",
            body,
            flags=re.S
        )
        clauses = [c.strip() for c in clauses if c.strip()]

        clause_list = []
        if clauses:
            for c in clauses:
                c = remove_trash(c)

                # Tách điểm a), b), c) …
                points = re.split(r"(?:^|\n)([a-z]\))", c)
                points = [p.strip() for p in points if p.strip()]

                if len(points) > 1:
                    clause_list.append({
                        "clause_text": c,
                        "points": points
                    })
                else:
                    clause_list.append({"clause_text": c})
        else:
            clause_list.append({"clause_text": remove_trash(body)})

        law_structure.append({
            "article_id": idx,
            "article_title": article_title,
            "article_link": article_link,
            "clauses": clause_list
        })

    return law_structure


# ============================
# SAVE JSON / JSONL
# ============================
def save_json(title, law_structure, filename="law.json", filename_jsonl="law.jsonl"):

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"law_title": title, "articles": law_structure},
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(filename_jsonl, "w", encoding="utf-8") as f:
        for art in law_structure:
            for clause in art["clauses"]:
                record = {
                    "law_title": title,
                    "article_id": art["article_id"],
                    "article_title": art["article_title"],
                    "article_link": art["article_link"],
                    "clause_text": clause["clause_text"],
                    "points": clause.get("points", [])
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Đã lưu {filename} và {filename_jsonl}")


# ============================
# CRAWL NHIỀU LUẬT
# ============================
def crawl_multiple_laws(urls: list[str], output_dir="laws_output1"):
    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        print(f"Đang crawl: {url}")
        title, raw_text = crawl_law(url)
        structure = split_law_structure(raw_text, url)

        safe_title = re.sub(r"[^\w\d]+", "_", title)[:50]
        json_file = os.path.join(output_dir, f"{safe_title}.json")
        jsonl_file = os.path.join(output_dir, f"{safe_title}.jsonl")

        save_json(title, structure, filename=json_file, filename_jsonl=jsonl_file)


if __name__ == "__main__":
    urls = [
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Dau-tu-so-61-2020-QH14-321051.aspx",
        "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-gia-2012-142540.aspx",
        "https://thuvienphapluat.vn/van-ban/Cong-nghe-thong-tin/Luat-an-toan-thong-tin-mang-2015-298365.aspx",
        "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Luat-phi-va-le-phi-2015-298376.aspx",
        "https://thuvienphapluat.vn/van-ban/Cong-nghe-thong-tin/Luat-an-ninh-mang-2018-351416.aspx",
        "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Luat-quan-ly-thue-2019-387595.aspx",
        "https://thuvienphapluat.vn/van-ban/Tien-te-Ngan-hang/Luat-Cac-to-chuc-tin-dung-32-2024-QH15-577203.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Ho-tro-doanh-nghiep-nho-va-vua-2017-320905.aspx"
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Quan-ly-su-dung-von-Nha-nuoc-dau-tu-vao-san-xuat-kinh-doanh-tai-doanh-nghiep-2014-259731.aspx",
        "https://thuvienphapluat.vn/van-ban/Bao-hiem/Luat-Kinh-doanh-bao-hiem-2022-465916.aspx",
        "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-Can-cuoc-26-2023-QH15-552422.aspx",
        "https://thuvienphapluat.vn/van-ban/Cong-nghe-thong-tin/Luat-Giao-dich-dien-tu-2023-20-2023-QH15-513347.aspx",
        "https://thuvienphapluat.vn/van-ban/Bao-hiem/Luat-Bao-hiem-xa-hoi-2024-557190.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Van-ban-hop-nhat-67-VBHN-VPQH-2025-Luat-Doanh-nghiep-671127.aspx",
        "https://thuvienphapluat.vn/van-ban/Ke-toan-Kiem-toan/Law-No-88-2015-QH13-on-accounting-299767.aspx?tab=3",
        "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-Luat-lao-dong-2019-333670.aspx",
        "https://thuvienphapluat.vn/van-ban/Chung-khoan/Luat-Chung-khoan-nam-2019-399763.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-canh-tranh-345182.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Pha-san-2014-238641.aspx",
        "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-Bao-ve-quyen-loi-nguoi-tieu-dung-2023-19-2023-QH15-500102.aspx",
        "https://thuvienphapluat.vn/van-ban/Tien-te-Ngan-hang/Luat-14-2022-QH15-Phong-chong-rua-tien-519327.aspx",
        "https://thuvienphapluat.vn/van-ban/So-huu-tri-tue/Luat-So-huu-tri-tue-2005-50-2005-QH11-7022.aspx",
        "https://thuvienphapluat.vn/van-ban/Tai-nguyen-Moi-truong/Luat-so-72-2020-QH14-Bao-ve-moi-truong-2020-431147.aspx",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Dat-dai-2024-31-2024-QH15-523642.aspx",
        "https://thuvienphapluat.vn/van-ban/Xay-dung-Do-thi/Luat-Xay-dung-2014-238644.aspx",
        "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Van-ban-hop-nhat-03-VBHN-VPQH-2024-Luat-Thue-thu-nhap-ca-nhan-629167.aspx", 
        "https://thuvienphapluat.vn/van-ban/Thuong-mai/Nghi-dinh-121-2021-ND-CP-kinh-doanh-tro-choi-dien-tu-co-thuong-cho-nguoi-nuoc-ngoai-499046.aspx" ,
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-80-2021-ND-CP-huong-dan-Luat-Ho-tro-doanh-nghiep-nho-va-vua-486147.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-01-2021-ND-CP-dang-ky-doanh-nghiep-283247.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-210-2025-ND-CP-sua-doi-Nghi-dinh-38-2018-ND-CP-630031.aspx"
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-23-2021-ND-CP-huong-dan-Luat-Viec-lam-ve-doanh-nghiep-hoat-dong-dich-vu-viec-lam-468265.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-07-2022-TT-BTC-ban-giao-xu-ly-khoan-no-khi-chuyen-doi-so-huu-doanh-nghiep-502987.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-05-2022-TT-BTC-huong-dan-tai-co-cau-doanh-nghiep-khong-du-dieu-kien-co-phan-hoa-502983.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-124-2021-TT-BTC-huong-dan-co-che-tai-chinh-kem-theo-Quyet-dinh-1804-QD-TTg-318693.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-20-2021-TT-BGTVT-Quy-che-phoi-hop-giua-doanh-nghiep-duoc-giao-quan-ly-ha-tang-hang-khong-488878.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-78-2025-TT-BTC-bai-bo-Thong-tu-01-2020-TT-BKHDT-huong-dan-phan-loai-hop-tac-xa-668204.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-168-2025-ND-CP-dang-ky-doanh-nghiep-623074.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-76-2025-TT-BTC-bai-bo-thong-tu-tu-vay-tu-tra-doanh-nghiep-nha-nuoc-so-huu-tren-50-von-666402.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-38-2018-ND-CP-quy-dinh-chi-tiet-dau-tu-cho-doanh-nghiep-nho-va-vua-khoi-nghiep-sang-tao-377302.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Thue-thu-nhap-doanh-nghiep-2025-so-67-2025-QH15-580594.aspx"
    ]

    crawl_multiple_laws(urls)
