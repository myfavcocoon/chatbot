#postprocessing.py
import re


def clean_text(raw_text: str) -> str:
    match = re.search(r"câu\s*hỏi\s*:", raw_text, flags=re.IGNORECASE)
    if match:
        return raw_text[:match.start()]
    return raw_text


if __name__ == "__main__":
    TEST = """
Đây là phần trả lời hợp lệ.

Câu hỏi:
Đây là phần rác model sinh thêm → cần xóa.

"""

    print("\n====== CLEANED OUTPUT ======\n")
    print(clean_text(TEST))
