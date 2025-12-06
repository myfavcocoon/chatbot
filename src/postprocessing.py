# ===============================
# File: postprocessing.py
# ===============================

import re
from rapidfuzz import fuzz


# =====================================================================
# 1. Cắt tất cả từ 'Câu hỏi:' trở xuống
# =====================================================================
def cut_after_question(text: str) -> str:
    match = re.search(r"câu\s*hỏi\s*:", text, flags=re.IGNORECASE)
    if match:
        return text[:match.start()].strip()
    return text


# =====================================================================
# 2. Tách câu nhưng GIỮ newline để preserve format
# =====================================================================
def split_sentences_with_newline(text: str):
    # Tách thành list: [sentence, newline, sentence, newline, ...]
    parts = re.split(r'(?<=[.!?])(\s+|\n+)', text)

    items = []
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        if not sent:
            continue
        newline = parts[i + 1] if i + 1 < len(parts) else ""
        items.append({"text": sent, "newline": newline})
    return items


# =====================================================================
# 3. Loại câu chứa dấu ngoặc ()
# =====================================================================
def remove_parenthesis_items(items):
    output = []
    for item in items:
        if "(" in item["text"] or ")" in item["text"]:
            continue
        output.append(item)
    return output


# =====================================================================
# 4. Loại trùng exact + fuzzy (so sánh theo TEXT, giữ lại newline)
# =====================================================================
def dedupe_items(items, threshold=92):
    out = []
    seen = []

    for item in items:
        s = item["text"]

        if not seen:
            seen.append(s)
            out.append(item)
            continue

        # fuzzy với ALL câu đã thấy
        sim = max(fuzz.ratio(s, prev) for prev in seen)

        if sim < threshold:
            seen.append(s)
            out.append(item)

    return out


# =====================================================================
# 5. Hàm chính
# =====================================================================
def clean_text(raw_text: str) -> str:
    # 1. Cắt phần từ “Câu hỏi:” trở xuống
    text = cut_after_question(raw_text)

    # 2. Tách câu + preserve newline
    items = split_sentences_with_newline(text)

    # 3. Remove câu chứa ngoặc ()
    items = remove_parenthesis_items(items)

    # 4. Dedupe exact + fuzzy
    items = dedupe_items(items)

    # 5. Ghép lại, giữ format gốc
    return "".join([item["text"] + item["newline"] for item in items])


if __name__ == "__main__":
    TEST_INPUT = """
Công ty là doanh nghiệp. Công ty là doanh nghiệp.  

Theo Khoản 1, Điều 4, Luật Doanh nghiệp, doanh nghiệp là tổ chức có tên riêng, có tài sản, có trụ sở giao dịch. 
Doanh nghiệp là tổ chức có tên riêng, có tài sản, có trụ sở giao dịch!  

(Trợ lý pháp lý AI)

Nếu bạn cần thêm thông tin, vui lòng cung cấp thêm ngữ cảnh.
Nếu bạn cần thêm thông tin, vui lòng cung cấp thêm ngữ cảnh.
Nếu bạn cần thêm thông tin vui lòng cung cấp thêm ngữ cảnh.  

Cảm ơn bạn đã sử dụng dịch vụ của tôi.  
Cảm ơn bạn đã sử dụng dịch vụ của tôi!  

Câu hỏi:
Công ty TNHH có phải là doanh nghiệp không?
"""

    print("\n====== CLEANED OUTPUT ======\n")
    print(clean_text(TEST_INPUT))
