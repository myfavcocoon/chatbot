
import os
import json
import string
import re
from rank_bm25 import BM25Okapi
from unidecode import unidecode
from .config import (
    STOPWORDS_FILE, BASE_DIR, LAW_SHORT_NAMES,
    BM25_K1, BM25_B, BM25_TOPK
)

# ============================================================
# Load stopwords
# ============================================================
def load_stopwords(file_path=STOPWORDS_FILE):
    if not os.path.isabs(file_path):
        file_path = os.path.join(BASE_DIR, file_path)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return set([line.strip().lower() for line in f if line.strip()])
    return set()

STOPWORDS = load_stopwords()

# ============================================================
# Text preprocessing
# ============================================================
def clean_text(text):
    text = re.sub(r'<.*?>', '', text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_text(text):
    listpunct = string.punctuation.replace('_','')
    for p in listpunct:
        text = text.replace(p, ' ')
    return text.lower()

def custom_tokenizer(text):
    text = clean_text(text)
    text = normalize_text(text)

    law_token = ""
    for name in LAW_SHORT_NAMES:
        if name in text:
            law_token = "_".join(unidecode(name).split())
            text = text.replace(name, "")
            break

    text = re.sub(r"điều\s*(\d+)", r"dieu_\1", text, flags=re.IGNORECASE)
    text = re.sub(r"khoản\s*(\d+)", r"khoan_\1", text, flags=re.IGNORECASE)

    if law_token:
        text = law_token + " " + text.strip()

    words = text.split()
    return [w for w in words if w not in STOPWORDS]

# ============================================================
# BM25 Retriever Class
# ============================================================
class BM25Retriever:
    def __init__(self, jsonl_path=None):
        self.index = None
        self.metadata = None
        if jsonl_path is not None:
            self.init_index(jsonl_path)

    def init_index(self, jsonl_path):
        if not os.path.isabs(jsonl_path):
            jsonl_path = os.path.join(BASE_DIR, jsonl_path)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"BM25 JSONL file not found: {jsonl_path}")

        docs, meta = [], []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                docs.append(item["clause_text"])
                meta.append(item)

        tokenized = [custom_tokenizer(doc) for doc in docs]
        self.index = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)
        self.metadata = meta
        print(f"[BM25] Loaded {len(docs)} documents from {jsonl_path}")

    def search(self, text, top_k=BM25_TOPK):
        if self.index is None:
            raise RuntimeError("BM25 index not initialized. Call init_index() first or provide jsonl_path at class init.")
        
        query_tokens = custom_tokenizer(text)
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            meta = self.metadata[idx]
            results.append({
                "id": meta["id"],
                "score": float(score),
                "law_title": meta.get("law_title", ""),
                "article_title": meta.get("article_title", ""),
                "clause_no": meta.get("clause_no", ""),
                "article_link": meta.get("article_link", ""), 
                "text": meta.get("clause_text", "")
            })

        return results

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    input_path = os.path.join(BASE_DIR, "data", "keywords_db.jsonl")
    retriever = BM25Retriever(jsonl_path=input_path)

    query = "Theo luật lao động Điều 6 Khoản 2, người sử dụng lao động cần làm gì?"
    results = retriever.search(query, top_k=5)

    print(f"\n🔎 Query: {query}")
    print("\nTop 5 BM25 Results:\n")
    for r in results:
        print(f"ID: {r['id']} | Score: {r['score']:.4f} | Điều {r['clause_no']} | Link {r['article_link']}")
        print("TEXT:", r["text"][:100], "...\n")
