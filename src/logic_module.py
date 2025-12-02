# logic_module.py
import os
import uuid
from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from transformers import pipeline, AutoTokenizer
import torch

from .config import *  # BASE_DIR, MONGO_URI, MONGO_DB, MONGO_COLLECTION, RRF_K, COSINE_THRESHOLD, BM25_TOPK, PINECONE_TOPK, EMBED_MODEL_NAME, HF_TOKEN
from .bm25_manager import BM25Retriever
from .pinecone_manager import search_pinecone
from .ensemble_retriever import build_context, extract_text, ensemble_rrf, embed_text
from .model_loader import load_model_with_adapter, build_pipeline

# from config import *  # BASE_DIR, MONGO_URI, MONGO_DB, MONGO_COLLECTION, RRF_K, COSINE_THRESHOLD, BM25_TOPK, PINECONE_TOPK, EMBED_MODEL_NAME, HF_TOKEN
# from bm25_manager import BM25Retriever
# from pinecone_manager import search_pinecone
# from ensemble_retriever import build_context, extract_text, ensemble_rrf, embed_text
# from model_loader import load_model_with_adapter, build_pipeline
# -------------------- 1️⃣ MongoDB --------------------
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
sessions_col = db[MONGO_COLLECTION]
sessions_col.create_index([("updated_at", DESCENDING)])

def create_session(name=None):
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = {
        "session_id": sid,
        "name": name or f"Phiên {now.strftime('%H:%M %d/%m')}",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "summary_memory": ""
    }
    sessions_col.insert_one(doc)
    return sid

def list_sessions(limit=100):
    docs = sessions_col.find({}, {"session_id":1, "name":1, "updated_at":1}).sort("updated_at", -1).limit(limit)
    out = []
    for d in docs:
        label = f"{d.get('name')} — {d.get('updated_at').strftime('%Y-%m-%d %H:%M:%S')}"
        out.append((label, d.get('session_id')))
    return out

def load_session_doc(session_id):
    return sessions_col.find_one({"session_id": session_id})

def save_message(session_id, role, text):
    now = datetime.now(timezone.utc)
    sessions_col.update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"role": role, "text": text, "ts": now}},
         "$set": {"updated_at": now}}
    )

def save_summary_buffer(session_id, buffer_str):
    sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {"summary_memory": buffer_str, "updated_at": datetime.now(timezone.utc)}}
    )

def delete_session(session_id):
    sessions_col.delete_one({"session_id": session_id})

# -------------------- 2️⃣ BM25 + Embedding --------------------
input_path = os.path.join(BASE_DIR, "data/keywords_db.jsonl")
bm25_retriever = BM25Retriever(jsonl_path=input_path)
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

# -------------------- 3️⃣ Load model pipeline --------------------
model_key = "qwen2-3b"  # hoặc các key khác trong MODEL_OPTIONS

# Thay vì load model + tokenizer riêng, dùng build_pipeline
gen_pipe, tokenizer = build_pipeline(
    model_key=model_key,
    max_new_tokens=1024,
    temperature=0.2
)

# -------------------- 4️⃣ Summary update --------------------
def update_summary(old_summary, new_lines, max_chars=MAX_MEMORY_CHARS):
    combined = (old_summary + "\n" + new_lines).strip() if old_summary else new_lines
    return combined[-max_chars:]

# -------------------- 5️⃣ Chat function --------------------
custom_template = """
Bạn là trợ lý pháp lý AI. Luôn trả lời chính xác theo luật Việt Nam, trình bày rõ ràng và có cấu trúc, nếu thiếu thông tin thì hỏi lại. Không suy đoán sai luật.
 
### Ngữ cảnh pháp lý:
{context}

### Lịch sử hội thoại:
{history}

### Câu hỏi:
{input}

### Trả lời:"""


def chat_fn(session_id, gr_history, user_input, summary_buffer, retrieval_cache, top_k=5):
    if not session_id:
        session_id = create_session("Phiên mới")
    if not user_input.strip():
        return session_id, gr_history or [], summary_buffer, retrieval_cache, "<div style='color:red'>Vui lòng nhập câu hỏi.</div>"

    # Build context + retrieval
    context, refs, retrieval_cache, cache_hit, cosine_score = build_context(user_input, retrieval_cache, bm25_retriever=bm25_retriever, embedding_model=embedding_model)
    history_summary = summary_buffer or "Không có hội thoại trước."

    # Chuẩn bị prompt cho LLM
    full_prompt = custom_template.format(context=context, history=history_summary, input=user_input)
    ans_full = gen_pipe(full_prompt)[0]["generated_text"]
    split_token = "### Trả lời:"
    ans = ans_full.split(split_token, 1)[1].strip() if split_token in ans_full else ans_full.strip()

    # Update Gradio history + Mongo
    gr_history = gr_history or []
    gr_history.append((user_input, ans))
    save_message(session_id, "user", user_input)
    save_message(session_id, "assistant", ans)

    # Update summary memory
    new_lines = f"Q: {user_input}\nA: {ans}"
    updated_summary = update_summary(history_summary, new_lines)
    save_summary_buffer(session_id, updated_summary)

    refs_html = """
    <style>
    .ref-card {
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 10px;
        background: #f7f9fc;
        border: 1px solid #e3e8ef;
        transition: all 0.2s ease;
    }
    .ref-card:hover {
        background: #eef3f9;
        border-color: #c9d6e4;
    }
    .ref-link {
        text-decoration: none;
        color: #1a73e8;
        font-weight: 600;
    }
    .ref-meta {
        font-size: 13px;
        color: #555;
        margin-top: 3px;
    }
    </style>
    """

    for ref in refs[:top_k]:
        data = ref.get("bm25") or ref.get("pinecone")

        law = data.get("law_title", "Không xác định luật")
        title = data.get("article_title", "Điều không xác định")
        clause = data.get("clause_no", "")
        link = data.get("article_link", "#")

        display = f"{law} - {title}"
        if clause:
            display += f" - Khoản {clause}"

        refs_html += f"""
        <div class="ref-card">
            <a class="ref-link" href="{link}" target="_blank">📘 {display}</a>
            <div class="ref-meta">Nguồn: {law}</div>
        </div>
        """


      
    return session_id, gr_history, updated_summary, retrieval_cache, refs_html

# -------------------- 6️⃣ Session handlers --------------------
def create_session_handler(name):
    sid = create_session(name)
    return sid, [], "", [], "<div>Created new session.</div>"

def load_session_handler(session_label):
    if not session_label: return None, [], "", [], ""
    sessions = list_sessions()
    sid = next((s_id for label,s_id in sessions if label==session_label), None)
    if not sid: return None, [], "", [], ""
    doc = load_session_doc(sid)
    gr_history = [(m["text"], "") for m in doc.get("messages",[])]
    sm = doc.get("summary_memory","")
    return sid, gr_history, sm, [], ""

def delete_session_handler(session_label):
    if not session_label: return None, [], "", [], "<div>No session selected.</div>"
    sessions = list_sessions()
    sid = next((s_id for label,s_id in sessions if label==session_label), None)
    if not sid: return None, [], "", [], "<div>Session not found.</div>"
    delete_session(sid)
    new_sid = create_session("Phiên mới")
    return new_sid, [], "", [], "<div>Deleted session and created a new one.</div>"
if __name__ == "__main__":
    print("=== Test loading LLM pipeline ===")
    try:
        # Test prompt
        test_prompt = "Viết một câu hỏi pháp lý mẫu về hợp đồng:"
        output = gen_pipe(test_prompt, max_new_tokens=20, do_sample=False)
        print("Pipeline loaded successfully!")
        print("Test prompt output:", output[0]["generated_text"])
        print("Tokenizer vocab size:", len(tokenizer))
        print("CUDA available:", torch.cuda.is_available())
    except Exception as e:
        print("Error loading model or running pipeline:", e)
