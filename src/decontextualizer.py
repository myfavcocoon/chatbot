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
Bạn là một trợ lý AI thông minh, có nhiệm vụ viết lại câu hỏi của người dùng sao cho đầy đủ thông tin, độc lập, và dễ hiểu **bằng tiếng Việt**.

Hướng dẫn:
- Nếu câu hỏi phụ thuộc vào hội thoại trước đó, hãy giải quyết tất cả **tham chiếu (coreferences)** và đảm bảo câu hỏi có thể hiểu được **một mình**.
- Không bỏ sót bất kỳ thông tin quan trọng nào.
- Giữ nguyên ý nghĩa gốc của câu hỏi.
- Không lặp lại những câu hỏi đã xuất hiện trong đoạn hội thoại trước.
- Nếu câu hỏi không liên quan đến ngữ cảnh trước, không cần sửa, giữ nguyên câu hỏi.

Hội thoại trước đó:
{ctx_text}

Câu hỏi mới: {question}

Viết lại câu hỏi:
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
        ("đường link đăng nhập vào cổng thông tin quốc gia là gì?", 
         "Bạn có thể truy cập Cổng thông tin quốc gia về đăng ký doanh nghiệp tại https://dangkykinhdoanh.gov.vn"),
        ("Doanh nghiệp cần làm gì nếu vi phạm luật lao động?", 
         "Doanh nghiệp phải thực hiện các biện pháp khắc phục và có thể bị xử phạt theo quy định."),
        ("Nếu vi phạm thì phạt bao nhiêu", 
         "Người đại diện pháp luật của doanh nghiệp chịu trách nhiệm chính."),
        ("Mức phạt tối đa đối với hành vi xả thải trái phép là bao nhiêu?", 
         "Mức phạt có thể lên tới 500 triệu đồng tùy theo loại vi phạm."),
        ("Thời hạn để nộp hồ sơ đăng ký doanh nghiệp là bao lâu?", 
         "Thời hạn nộp hồ sơ đăng ký doanh nghiệp là 3 ngày làm việc kể từ khi chuẩn bị đầy đủ hồ sơ.")
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
