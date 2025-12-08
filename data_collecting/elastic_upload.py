import os
import json
import re
import unicodedata
from tqdm import tqdm
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from elasticsearch import Elasticsearch, helpers

# ===============================
# CONFIG
# ===============================

JSONL_PATH = "processed_laws1/all_laws_merged_clean_split.jsonl"

PC_INDEX = "raglaw-final"
MODEL_NAME = "BAAI/bge-m3"   # BGE-M3 (1024-dim)

ES_INDEX = "rag_law"
MAX_METADATA_CHARS = 4000
BATCH_SIZE = 32

EMBED_JSONL_PATH = "embedded_laws.jsonl"  # <--- file lưu embedding

# ===============================
# Helpers
# ===============================

def normalize_id(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^A-Za-z0-9_\-]', '_', text)

def truncate_text(text: str, max_len: int = MAX_METADATA_CHARS) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text

# ===============================
# Connect ES
# ===============================

es = Elasticsearch(
    "https://6ef39ac35d084c4ea45c7943bfd0f447.us-central1.gcp.cloud.es.io:443",
    api_key="YXdWaW9ab0JaVUtqUUlabjQyT3g6UkZuX09OMm10QlhlOExQRnlmUV9zUQ=="
)

# Delete old ES index
if es.indices.exists(index=ES_INDEX):
    print(f"Deleting existing Elasticsearch index '{ES_INDEX}'...")
    es.indices.delete(index=ES_INDEX)

print(f"Creating fresh Elasticsearch index '{ES_INDEX}'...")
es.indices.create(index=ES_INDEX)
print("Elasticsearch OK.")

def upload_es_batch(batch):
    actions = []
    for d in batch:
        actions.append({
            "_index": ES_INDEX,
            "_source": {
                "text": d["clause_text"],
                "law_title": d.get("law_title", ""),
                "article_title": d.get("article_title", ""),
                "article_id": d.get("article_id", ""),
                "clause_no": d.get("clause_no", ""),
                "article_link": d.get("article_link", "")
            }
        })
    if actions:
        helpers.bulk(es, actions)

# ===============================
# Connect Pinecone
# ===============================

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "pcsk_2oNNtt_2ti3W7tcQYiTvyio2iFxi9Dbfy54h9uV4neYt2tkAffBFjszzj5Gvqaie2g7Zat"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete + recreate Pinecone index
if PC_INDEX in [i.name for i in pc.list_indexes()]:
    print(f"Deleting existing Pinecone index '{PC_INDEX}'...")
    pc.delete_index(PC_INDEX)

print(f"Creating fresh Pinecone index '{PC_INDEX}'...")
pc.create_index(
    name=PC_INDEX,
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

pindex = pc.Index(PC_INDEX)
print("Pinecone OK.")

# ===============================
# Load JSONL
# ===============================

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(docs)} cleaned law clauses.")

# ===============================
# Embedding Model (BGE-M3)
# ===============================

print("Loading BGE-M3 embedding model...")
embedder = SentenceTransformer(MODEL_NAME)

# ===============================
# MAIN LOOP
# ===============================

embedded_docs = []

for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Processing batches"):

    batch = docs[i:i+BATCH_SIZE]

    # --- Upload to Elasticsearch ---
    upload_es_batch(batch)

    # --- Prepare texts ---
    texts = [d["clause_text"] for d in batch]
    ids = [normalize_id(f'{d["law_title"]}_{d["article_id"]}_{d["clause_no"]}') for d in batch]

    # --- Embedding with BGE-M3 ---
    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # --- Prepare vectors & save for JSONL ---
    vectors = []
    for j, d in enumerate(batch):
        vector_item = {
            "id": ids[j],
            "values": embeddings[j].tolist(),
            "metadata": {
                "text": truncate_text(d["clause_text"]),
                "law_title": d.get("law_title", ""),
                "article_title": d.get("article_title", ""),
                "article_id": str(d.get("article_id", "")),
                "clause_no": str(d.get("clause_no", "")),
                "link": d.get("article_link", "")
            }
        }
        vectors.append(vector_item)
        embedded_docs.append(vector_item)

    # --- Upload to Pinecone ---
    pindex.upsert(vectors=vectors)

# ===============================
# Save embedded JSONL
# ===============================
with open(EMBED_JSONL_PATH, "w", encoding="utf-8") as f:
    for item in embedded_docs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nDONE! Embedded data saved to: {EMBED_JSONL_PATH}")
print("Fresh data uploaded to Elasticsearch + Pinecone.")
