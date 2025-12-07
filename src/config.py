# src/config.py
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# BASE_DIR = parent folder của src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ----------------- MongoDB -----------------
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION")

# ----------------- Pinecone -----------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_TOP_K = 5
# ----------------- HuggingFace -----------------
HF_TOKEN = os.environ.get("HF_TOKEN")

# ----------------- Gemini -----------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ----------------- BM25 params -----------------
BM25_K1 = 1.0
BM25_B = 0.2
BM25_TOPK = 5

# ----------------- RRF -----------------
RRF_K = 60

# ----------------- Memory -----------------
MAX_MEMORY_CHARS = 2000
COSINE_THRESHOLD = 0.75

# ----------------- Embedding model -----------------
EMBED_MODEL_NAME = "BAAI/bge-m3"

# ----------------- Law short names -----------------
LAW_SHORT_NAMES = [
    "luật lao động",
    "luật bảo hiểm xã hội",
    "luật bảo vệ môi trường",
    "luật bảo vệ quyền lợi người tiêu dùng",
    "luật chứng khoán",
    "luật căn cước",
    "luật cạnh tranh",
    "luật giao dịch điện tử",
    "luật kinh doanh bảo hiểm",
    "luật kế toán",
    "luật phá sản",
    "luật phòng chống rửa tiền",
    "luật quản lý thuế",
    "luật sở hữu trí tuệ",
    "luật xây dựng",
    "luật đất đai",
    "luật đầu tư",
    "nghị định 121 2021",
    "nghị định 168 2025",
    "nghị định 23 2021",
    "nghị định 80 2021",
    "thông tư 05 2022",
    "thông tư 07 2022",
    "nghị định 124 2021",
    "thông tư 20 2022",
    "luật thuế thu nhập cá nhân",
    "luật doanh nghiệp"
    "luật thuế doanh nghiệp"
]

MODEL_OPTIONS = {
    "qwen2-3b": {
        "base_model": "AITeamVN/Vi-Qwen2-3B-RAG",
        "adapter_dir": os.path.join(BASE_DIR, "models", "qwen2-3b"),
    },
    "qwen2-7b": {
        "base_model": "AITeamVN/Vi-Qwen2-7B-RAG",
        "adapter_dir": os.path.join(BASE_DIR, "models", "qwen2-7b"),
    },
    "llama-3b": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "adapter_dir": os.path.join(BASE_DIR, "models", "llama-3b"),
    }

}

# # ----------------- Stopwords -----------------

STOPWORDS_FILE = os.path.join(BASE_DIR, r"data\vietnamese-stopwords.txt")
MODEL_KEY = "qwen2-3b"
if __name__ == "__main__":
    print("===== DEBUG CONFIG =====")

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STOPWORDS_FILE: {STOPWORDS_FILE}")

    print(f"MONGO_URI: {MONGO_URI}")
    print(f"MONGO_DB: {MONGO_DB}")
    print(f"MONGO_COLLECTION: {MONGO_COLLECTION}")

    print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
    print(f"PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")

    print(f"HF_TOKEN: {HF_TOKEN}")
    print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")

    print(f"BM25_K1: {BM25_K1}, BM25_B: {BM25_B}, BM25_TOPK: {BM25_TOPK}")
    print(f"RRF_K: {RRF_K}")
    print(f"MAX_MEMORY_CHARS: {MAX_MEMORY_CHARS}, COSINE_THRESHOLD: {COSINE_THRESHOLD}")
    print(f"EMBED_MODEL_NAME: {EMBED_MODEL_NAME}")


    for model_name, cfg in MODEL_OPTIONS.items():
        print(f"{model_name} adapter_dir: {cfg['adapter_dir']}")
        
    print("=== DEBUG CONFIG END ===")