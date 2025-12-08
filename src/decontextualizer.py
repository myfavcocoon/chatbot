# src/decontextualizer.py

import time
from typing import List
import google.generativeai as genai
from .config import GEMINI_API_KEY


def decontextualize_conversation(context: List[str], question: str, DEBUG: bool = False) -> str:
    """
    Rewrite a user question to be fully decontextualized.
    If DEBUG=True and Gemini API fails at any step, print a warning and return original question.
    """
    import time
    import google.generativeai as genai

    ctx_text = "\n".join(context)
    prompt = f"""
    Bạn là trợ lý AI pháp lý, nhiệm vụ của bạn là **chỉ làm rõ câu hỏi nếu nó thiếu thông tin cần thiết để hiểu độc lập**, bằng **tiếng Việt**, thật ngắn gọn, đúng luật, không thêm từ ngữ thừa.

    Hướng dẫn:

    - Nếu câu hỏi đã đủ ý, độc lập, và rõ ràng → **giữ nguyên văn**.
    - Nếu câu hỏi thiếu thông tin tham chiếu trong hội thoại trước đó → bổ sung thông tin cần thiết để câu hỏi có thể hiểu được độc lập.
    - Nếu câu hỏi không liên quan đến luật (ví dụ small talk, hỏi chuyện ngoài luật) → giữ nguyên câu hỏi, không sửa.
    - Không lặp lại câu hỏi cũ.
    - Không thêm lời mở đầu, kết thúc, hay giải thích gì. Chỉ trả về câu hỏi cuối cùng đã chỉnh sửa nếu cần.

    Hội thoại trước đó:
    {ctx_text}

    Câu hỏi mới: {question}

    Viết lại câu hỏi (hoặc giữ nguyên nếu đủ ý):
    """


    start = time.time()
    try:
        # ============= Configure API and define model =============
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        output = response.text.strip()
    except Exception as e:
        if DEBUG:
            print(f"[⚠️ WARNING]: {e}")
        output = question  # fallback: trả về câu hỏi gốc
    end = time.time()

    if DEBUG:
        print(f"[⏱️ Processing time: {end - start:.3f} seconds]")
        print("Rewrite:", output)
    return output

if __name__ == "__main__":
    # Context ban đầu có 2 cặp Q&A
    test_context = [
        "Q: Doanh nghiệp có cần giấy phép để xả thải ra môi trường không?",
        "A: Có, doanh nghiệp phải có giấy phép theo quy định pháp luật.",
        "Q: Nếu không có giấy phép thì có bị xử phạt không?",
        "A: Có, doanh nghiệp sẽ bị xử phạt hành chính.",
    ]

    # Danh sách các câu hỏi mới và câu trả lời mẫu tương ứng
    test_qas = [
        ("cách thành lập doanh nghiệp", 
         "bạn có thể đăng ký qua cổng thông tin điên tử"),        
        ("link là gì?", 
         "Bạn có thể truy cập Cổng thông tin quốc gia về đăng ký doanh nghiệp tại https://dangkykinhdoanh.gov.vn"),
        ("Doanh nghiệp cần làm gì nếu vi phạm luật lao động?", 
         "Doanh nghiệp phải thực hiện các biện pháp khắc phục và có thể bị xử phạt theo quy định."),
        ("Nếu vi phạm thì phạt bao nhiêu", 
         "Người đại diện pháp luật của doanh nghiệp chịu trách nhiệm chính."),
        ("Mức phạt tối đa đối với hành vi xả thải trái phép là bao nhiêu?", 
         "Mức phạt có thể lên tới 500 triệu đồng tùy theo loại vi phạm."),
        ("Điều 5 khoản 3 luật doanh nghiệp quy định gì?", 
         "Mức phạt có thể lên tới 500 triệu đồng tùy theo loại vi phạm."),
        ("Ok cảm ơn bạn", 
         "kcj.")
    ]

    print("=== Running batch tests with Q&A ===\n")

    for i, (q, a) in enumerate(test_qas, 1):
        print(f"--- Test {i} ---")
        rewrite = decontextualize_conversation(test_context, q, DEBUG=False)
        print("Original question:", q)
        print("Rewritten question:", rewrite)
        print("Sample answer:", a, "\n")
        
        # Cập nhật conversation: thêm Q&A vừa rewrite
        test_context.append(f"Q: {rewrite}")
        test_context.append(f"A: {a}")
        
        # Giữ context 4 turn gần nhất để mô phỏng hội thoại liên tục
        test_context = test_context[-4:]
