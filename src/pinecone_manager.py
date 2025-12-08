import os
from sentence_transformers import SentenceTransformer
import torch
from config import (
    BASE_DIR, PINECONE_API_KEY, PINECONE_INDEX_NAME,
    EMBED_MODEL_NAME, PINECONE_TOP_K
)
from pinecone import Pinecone

# -----------------------
# Pinecone init
# -----------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_pinecone = pc.Index(PINECONE_INDEX_NAME)

# -----------------------
# Embedding model init
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

def embed_text(text: str):
    emb = embedding_model.encode(text)
    return emb.tolist()

# -----------------------
# Query Pinecone (text hoặc vector)
# -----------------------
def search_pinecone(query_input, top_k=PINECONE_TOP_K, is_vector=True):
    """
    query_input:
        - nếu is_vector=False → query_input là TEXT, sẽ embed
        - nếu is_vector=True  → query_input là VECTOR, dùng trực tiếp

    Example:
        search_pinecone("tìm điều khoản")  --> auto embed
        search_pinecone(vector, is_vector=True)
    """

    if is_vector:
        vector = query_input
    else:
        vector = embed_text(query_input)
    matches = index_pinecone.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )["matches"]

    # 3) chuẩn hóa kết quả
    results = []
    for m in matches:
        metadata = m.get("metadata", {})
        results.append({
            "id": m["id"],
            "score": float(m["score"]),
            "law_title": metadata.get("law_title", ""),
            "article_title": metadata.get("article_title", ""),
            "clause_no": metadata.get("clause_no", ""),
            "article_link": metadata.get("article_link", ""),
            "text": metadata.get("clause_text", ""),
        })

    return results


# -----------------------
# Debug mode
# -----------------------
if __name__ == "__main__":
    print("=== DEBUG Pinecone Module ===")

    test_text = "Theo luật Lao động Điều 6 Khoản 2, người sử dụng lao động cần làm gì?"

    # Test embed
    vec = embed_text(test_text)
    print(f"Embedding length: {len(vec)}")

    # Test query with TEXT
    print("\n--- Query bằng TEXT ---")
    res1 = search_pinecone(test_text,is_vector=False)
    print(res1[:2])

    # Test query with VECTOR
    print("\n--- Query bằng VECTOR (không embed lại) ---")
    res2 = search_pinecone(vec, is_vector=True)
    print(res2[:2])
