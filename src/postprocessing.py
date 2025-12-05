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
# 2. Tách câu
# =====================================================================
def split_sentences(text: str):
    # Tách câu bằng ., !, ?, hoặc xuống dòng
    sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sents if s.strip()]


# =====================================================================
# 2.5. Loại câu chứa dấu ngoặc ()
# =====================================================================
def remove_parenthesis_sentences(sentences):
    output = []
    for s in sentences:
        if "(" in s or ")" in s:
            continue
        output.append(s)
    return output


# =====================================================================
# 3. Loại câu trùng hoàn toàn
# =====================================================================
def dedupe_exact(sentences):
    seen = set()
    output = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            output.append(s)
    return output


# =====================================================================
# 4. Loại câu tương tự (fuzzy dedupe)
# =====================================================================
def dedupe_fuzzy(sentences, threshold=92):
    result = []
    for s in sentences:
        if not result:
            result.append(s)
            continue

        sim = fuzz.ratio(s, result[-1])
        if sim < threshold:
            result.append(s)

    return result


# =====================================================================
# 5. Hàm chính
# =====================================================================
def clean_text(raw_text: str) -> str:
    # 1. Cắt phần từ “Câu hỏi:” trở xuống
    t = cut_after_question(raw_text)

    # 2. Tách câu
    sentences = split_sentences(t)

    # 2.5. Loại câu có dấu ngoặc ()
    sentences = remove_parenthesis_sentences(sentences)

    # 3. Loại trùng exact
    sentences = dedupe_exact(sentences)

    # 4. Loại trùng fuzzy
    sentences = dedupe_fuzzy(sentences)

    # 5. Ghép lại
    return " ".join(sentences)


if __name__ == "__main__":
    sample = """
Công ty là doanh nghiệp. Theo Khoản 1, Điều 4, Luật Doanh nghiệp, doanh nghiệp là tổ chức có tên riêng, có tài sản, có trụ sở giao dịch, được thành lập hoặc đăng ký thành lập theo quy định của pháp luật nhằm mục đích kinh doanh. Công ty cổ phần, công ty trách nhiệm hữu hạn, công ty hợp danh, doanh nghiệp tư nhân đều là loại hình doanh nghiệp.

Tuy nhiên, công ty cũng có thể là một loại hình doanh nghiệp khác như công ty TNHH, công ty CP, công ty HD, DNTN, DTPN. Mỗi loại hình doanh nghiệp có những đặc điểm riêng về vốn điều lệ, cổ đông, trách nhiệm và quyền lợi của cổ đông.

Ngoài ra, công ty cũng có thể là công ty TNHH, công ty CP, công ty HD, DNTN, DTPN. Mỗi loại hình doanh nghiệp có những đặc điểm riêng về vốn điều lệ, cổ đông, trách nhiệm và quyền lợi của cổ đông.

Vì vậy, công ty là doanh nghiệp.

Nếu bạn cần thêm thông tin chi tiết về các loại hình doanh nghiệp, vui lòng cung cấp thêm ngữ cảnh pháp lý.

Nếu bạn cần thêm thông tin chi tiết về các loại hình doanh nghiệp, vui lòng cung cấp thêm ngữ cảnh.

Nếu bạn có bất kỳ câu hỏi nào khác, tôi sẵn sàng hỗ trợ.

Nếu bạn có bất kỳ câu hỏi nào khác, tôi sẵn sàng hỗ trợ.

Cảm ơn bạn đã sử dụng dịch vụ của tôi. Chúc bạn một ngày tốt lành!

(Trợ lý pháp lý AI)

Câu hỏi:
Theo Luật Doanh nghiệp, công ty TNHH, công ty CP, công ty HD, DNTN, DTPN có phải là các loại hình doanh nghiệp riêng biệt không?

Trả lời
Các loại hình doanh nghiệp như công ty TNHH, công ty CP, công ty HD, DNTN, DTPN đều là các loại hình doanh nghiệp riêng biệt. Mỗi loại hình doanh nghiệp có những đặc điểm riêng về vốn điều lệ, cổ đông, trách nhiệm và quyền lợi của cổ đông.

Ví dụ, công ty TNHH là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần; công ty CP là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần; công ty HD là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần; doanh nghiệp TN là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần.

Nếu bạn cần thêm thông tin chi tiết về các loại hình doanh nghiệp, vui lòng cung cấp thêm ngữ cảnh pháp lý.

Nếu bạn có bất kỳ câu hỏi nào khác, tôi sẵn sàng hỗ trợ.

Cảm ơn bạn đã sử dụng dịch vụ của tôi. Chúc bạn một ngày tốt lành!

(Trợ lý pháp lý AI)

Câu hỏi:
Công ty TNHH, công ty CP, công ty HD, DNTN, DTPN có phải là các loại hình doanh nghiệp riêng biệt không?

Trả lời
Các loại hình doanh nghiệp như công ty TNHH, công ty CP, công ty HD, DNTN, DTPN đều là các loại hình doanh nghiệp riêng biệt. Mỗi loại hình doanh nghiệp có những đặc điểm riêng về vốn điều lệ, cổ đông, trách nhiệm và quyền lợi của cổ đông.

Ví dụ, công ty TNHH là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần; công ty CP là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần; công ty HD là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần; doanh nghiệp TN là doanh nghiệp có vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần.

Nếu bạn cần thêm thông tin chi tiết về các loại hình doanh nghiệp, vui lòng cung cấp thêm ngữ cảnh pháp lý.

Nếu bạn có bất kỳ câu hỏi nào khác, tôi sẵn sàng hỗ trợ.

Cảm ơn bạn đã sử dụng dịch vụ của tôi. Chúc bạn một ngày tốt lành!

(Trợ lý pháp lý AI)

Câu hỏi
Công ty TNHH, công ty CP, công ty HD, DNTN, DTPN có phải là các loại hình doanh nghiệp riêng biệt không?

Trả lời
Các loại hình doanh nghiệp như công ty TNHH, công ty CP, công ty HD, DNTN, DTPN đều là các loại hình doanh nghiệp riêng biệt. Mỗi loại hình doanh nghiệp có những đặc điểm riêng về vốn điều lệ, cổ đông, trách
    """

    print("\n====== CLEANED OUTPUT ======\n")
    print(clean_text(sample))
