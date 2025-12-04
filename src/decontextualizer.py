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
You are a helpful assistant that rewrites user questions so they become fully decontextualized.

Rewrite the user's question so that:
- All coreferences are resolved.
- No information is missing.
- The question stands alone without any context.
- The original meaning stays the same.
- Do not duplicate earlier questions mentioned in the context.

Conversation:
{ctx_text}

Question: {question}
Rewrite:
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

    return output

if __name__ == "__main__":
    test_context = [
        "Q: Doanh nghiệp có cần giấy phép để xả thải ra môi trường không?",
        "A: Có, doanh nghiệp phải có giấy phép theo quy định pháp luật.",
        "Q: Nếu không có giấy phép thì có bị xử phạt không?",
        "A: Có, doanh nghiệp sẽ bị xử phạt hành chính.",
    ]

    test_question = "Cách thành lập doanh nghiệp"

    print("=== Running test with 5 conversation pairs ===")
    rewrite = decontextualize_conversation(test_context, test_question)
    print("Rewrite:", rewrite)
