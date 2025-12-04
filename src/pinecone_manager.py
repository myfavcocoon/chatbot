import os
from sentence_transformers import SentenceTransformer
import torch
from .config import BASE_DIR, PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBED_MODEL_NAME, PINECONE_TOP_K
from pinecone import Pinecone

# -----------------------
# Pinecone init
# -----------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_pinecone = pc.Index(PINECONE_INDEX_NAME)

# -----------------------
# Embedding model
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

def embed_text(text: str):
    emb = embedding_model.encode(text)
    return emb.tolist()

# -----------------------
# Query Pinecone
# -----------------------
def search_pinecone(text, top_k=PINECONE_TOP_K):
    """Truy xuất từ Pinecone, trả về danh sách doc với các trường chuẩn."""
    emb = embed_text(text)
    matches = index_pinecone.query(vector=emb, top_k=top_k, include_metadata=True)["matches"]
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
            "text": metadata.get("clause_text", "")  
        })
    return results


if __name__ == "__main__":
    print("=== DEBUG Pinecone Module ===")
    
    # Test embedding
    test_text = "Theo luật Lao động Điều 6 Khoản 2, người sử dụng lao động cần làm gì?"
    emb = embed_text(test_text)
    print(f"Embedding vector length: {len(emb)}")
    print(f"First 5 values: {emb[:5]}")

    # Test Pinecone query
    try:
        results = search_pinecone(test_text)
        print("\nTop 5 Pinecone results:")
        for r in results:  # chỉ lấy 3 kết quả đầu
            print(
                f"ID: {r['id']}, "
                f"Score: {r['score']}, "
                f"Law: {r['law_title']}, "
                f"Article: {r['article_title']}, "
                f"Clause: {r['clause_no']}, "
                f"Link: {r['article_link']}"                
                f"Text snippet: {r['text'][:50]}..."
            )
    except Exception as e:
        print("Error querying Pinecone:", e)