# postprocessing.py
import re

def clean_text(raw_text: str) -> str:
    # Tìm vị trí của các cụm cần loại bỏ, bao gồm [] hoặc ()
    match = re.search(
        r"(câu\s*hỏi\s*:|cấu\s*trúc\s*trả\s*lời\s*:|trợ\s∗lý\s∗pháp\s∗lý\s∗AItrợ\s*lý\s*pháp\s*lý\s*AI|\(trợ\s*lý\s*pháp\s*lý\s*AI\))",
        raw_text,
        flags=re.IGNORECASE
    )
    if match:
        return raw_text[:match.start()].strip()
    return raw_text.strip()


if __name__ == "__main__":
    TEST = """
Đây là phần trả lời hợp lệ.

Câu hỏi:
Đây là phần rác model sinh thêm → cần xóa.

Cấu trúc trả lời:
Đây cũng là rác.

[Trợ lý pháp lý AI] Thông tin thêm không cần thiết.

(Trợ lý pháp lý AI) Thêm nữa.
"""

    print("\n====== CLEANED OUTPUT ======\n")
    print(clean_text(TEST))
