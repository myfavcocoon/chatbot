from .config import RRF_K, COSINE_THRESHOLD, BM25_TOPK, PINECONE_TOP_K, EMBED_MODEL_NAME
from .bm25_manager import BM25Retriever
from .pinecone_manager import search_pinecone, embed_text
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# ============================================================
# RRF Ensemble
# ============================================================
def ensemble_rrf(bm25_results, pinecone_results, k=RRF_K, pinecone_weight=1.0, top_k=6, debug_print=False):
    scores = {}
    debug = {}

    # ----------------------
    # BM25
    # ----------------------
    for rank, r in enumerate(bm25_results, start=1):
        doc_id = r["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank)

        if doc_id not in debug:
            debug[doc_id] = {"id": doc_id}

        debug[doc_id]["bm25"] = {
            "rank": rank,
            "score": r.get("score", 0),
            "law_title": r.get("law_title", ""),
            "article_title": r.get("article_title", ""),
            "clause_no": r.get("clause_no", ""),
            "article_link": r.get("article_link", ""),
            "text": r.get("text", "")
        }

    # ----------------------
    # Pinecone
    # ----------------------
    for rank, r in enumerate(pinecone_results, start=1):
        doc_id = r["id"]
        scores[doc_id] = scores.get(doc_id, 0) + pinecone_weight * (1/(k + rank))

        if doc_id not in debug:
            debug[doc_id] = {"id": doc_id}

        debug[doc_id]["pinecone"] = {
            "rank": rank,
            "score": r.get("score", 0),
            "law_title": r.get("law_title", ""),
            "article_title": r.get("article_title", ""),
            "clause_no": r.get("clause_no", ""),
            "article_link": r.get("article_link", ""),
            "text": r.get("text", "")
        }

    # ----------------------
    # Sort by RRF score
    # ----------------------
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ----------------------
    # Lấy top-k
    # ----------------------
    ranked = ranked[:top_k]

    # ----------------------
    # Format output
    # ----------------------
    results = []
    for doc_id, rrf_score in ranked:
        entry = debug[doc_id]
        entry["rrf_score"] = rrf_score
        results.append(entry)

    # ----------------------
    # Debug print
    # ----------------------
    if debug_print:
        print("\n===== DEBUG RRF =====")
        for r in results:
            print(f"ID: {r['id']}, RRF score: {r['rrf_score']:.4f}")
            print("  BM25:", r.get("bm25"))
            print("  Pinecone:", r.get("pinecone"))
        print("===== END DEBUG RRF =====\n")

    return results


# ============================================================
# Extract text
# ============================================================
def extract_text(doc):
    if "bm25" in doc and doc["bm25"].get("text"):
        return doc["bm25"]["text"]
    if "pinecone" in doc and doc["pinecone"].get("text"):
        return doc["pinecone"]["text"]
    return ""

# ============================================================
# Build context
# ============================================================
def build_context(query, retrieval_cache, bm25_retriever=None, embedding_model=None, top_k=6, pinecone_weight=1.5):
    query_vec = embed_text(query) if embedding_model else None
    cache_hit = False
    cosine_score = None

    # Check cache
    for item in retrieval_cache:
        if query_vec:
            sim = 1 - cosine(query_vec, item["query_vec"])
            if sim >= COSINE_THRESHOLD:
                cache_hit = True
                cosine_score = sim
                context_text = "\n\n".join([extract_text(d) for d in item["docs"][:top_k]])
                return context_text, item["docs"][:top_k], retrieval_cache, cache_hit, cosine_score

    if bm25_retriever is None:
        raise ValueError("bm25_retriever must be provided and initialized")

    bm25_docs = bm25_retriever.search(query, BM25_TOPK)
    pine_docs = search_pinecone(query_vec, PINECONE_TOP_K)
    final_docs = ensemble_rrf(bm25_docs, pine_docs, k=RRF_K, pinecone_weight=pinecone_weight)

    # Cắt top_k doc sau RRF
    final_docs = final_docs[:top_k]

    # Update cache
    retrieval_cache.append({
        "query_text": query,
        "query_vec": query_vec,
        "docs": final_docs
    })

    context_text = "\n\n".join([extract_text(d) for d in final_docs])
    return context_text, final_docs, retrieval_cache, cache_hit, cosine_score

# ============================================================
# Test main
# ============================================================
if __name__ == "__main__":
    query = "Theo luật Lao động Điều 6 Khoản 2, người sử dụng lao động cần làm gì?"

    retrieval_cache = []
    bm25_retriever = BM25Retriever(jsonl_path="data/keywords_db.jsonl")
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

    context_text, final_docs, retrieval_cache, cache_hit, cosine_score = build_context(
        query,
        retrieval_cache,
        bm25_retriever=bm25_retriever,
        embedding_model=embedding_model,
        top_k=3,
    )

    print("\n===== Context Preview =====")
    print(context_text)
    print("Cache hit:", cache_hit, "Cosine score:", cosine_score)
