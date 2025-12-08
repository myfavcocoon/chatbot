import json
import re
from tqdm import tqdm

NOISE_PATTERNS = [
    r"THƯ\s*VIỆN\s*PHÁP\s*LUẬT",
    r"\bĐăng\s*nhập\b|\bĐăng\s*xuất\b|\bĐăng\s*ký\b",
    r"\bThành\s*Viên\b|\bBasic\b|\bPro\b",
    r"\bTải\s+Văn\s*bản\b|\bTải\s+về\b|\bfile\s+(PDF|Word)\b",
    r"\bLược\s*đồ\b|\btiện\s*ích\b",
    r"\bQuy\s*chế\b|\bThỏa thuận\b|\bBảo vệ Dữ liệu\b",
    r"www\.thuvienphapluat\.vn", r"Click trái|Tắt so sánh",
    r"Trang cá nhân", r"Xin chào Quý khách", r"Trân trọng"
]
noise_re = re.compile("|".join(NOISE_PATTERNS), re.I)

def parse_article_no(title: str):
    m = re.search(r"Điều\s+(\d+)\.", title)
    return int(m.group(1)) if m else None

def chunk_by_structure(article_text: str):
    parts = re.split(r'(?m)^\s*(\d+)\.\s+', article_text)
    chunks = []
    if len(parts) > 1:
        it = iter(parts[1:])
        for khoan_no, body in zip(it, it):
            pts = re.split(r'(?m)^\s*([a-z])\)\s+', body)
            if len(pts) > 1:
                it2 = iter(pts[1:])
                for diem_letter, sub in zip(it2, it2):
                    chunks.append({
                        "khoan_no": int(khoan_no),
                        "diem_letter": diem_letter,
                        "text": re.sub(r'\s+', ' ', sub).strip()
                    })
            else:
                chunks.append({
                    "khoan_no": int(khoan_no),
                    "diem_letter": None,
                    "text": re.sub(r'\s+', ' ', body).strip()
                })
    else:
        chunks.append({
            "khoan_no": None, "diem_letter": None,
            "text": re.sub(r'\s+', ' ', article_text).strip()
        })
    return chunks

def preprocess_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            art_no = parse_article_no(record.get("article_title", ""))
            if not art_no or art_no > 218:
                continue

            # Lấy text từ field 'clause_text'
            text = (record.get("clause_text") or "").strip()
            if not text or noise_re.search(text):
                continue

            sub_chunks = chunk_by_structure(text)
            for i, c in enumerate(sub_chunks, 1):
                new_record = {
                    "law_title": record.get("law_title", ""),
                    "article_id": record.get("article_id"),
                    "article_title": record.get("article_title", ""),
                    "article_no": art_no,
                    "chunk_id": i,
                    "khoan_no": c["khoan_no"],
                    "diem_letter": c["diem_letter"],
                    "text": c["text"],
                    "points": record.get("points", [])
                }
                f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    preprocess_jsonl(
        input_path="luat_doanh_nghiep_2025.jsonl",
        output_path="law_chunks_2025.jsonl"
    )
