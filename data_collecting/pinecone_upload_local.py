import os
import json
import re
import unicodedata
from tqdm import tqdm
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec

# --- CONFIG ---
INDEX_NAME = "raglaw-final"
DOCS_PATH = "processed_laws_merged/all_laws_merged_clean_split.jsonl"
MODEL_NAME = "BAAI/bge-m3"  # 384-dim multilingual
MAX_METADATA_CHARS = 4000  # truncate long law text

# --- Load API key ---
load_dotenv()
PINECONE_API_KEY = "pcsk_2oNNtt_2ti3W7tcQYiTvyio2iFxi9Dbfy54h9uV4neYt2tkAffBFjszzj5Gvqaie2g7Zat"
if not PINECONE_API_KEY:
    raise ValueError("Missing Pinecone API key. Please set PINECONE_API_KEY in your .env file")

# --- Helpers ---
def normalize_id(text: str) -> str:
    """Convert Vietnamese ID to ASCII-safe string"""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^A-Za-z0-9_\-]', '_', text)
    return text

def truncate_text(text: str, max_len: int = MAX_METADATA_CHARS) -> str:
    """Ensure metadata text is within Pinecone's 40KB limit"""
    return text[:max_len] + "..." if len(text) > max_len else text

# --- Connect to Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"Creating index '{INDEX_NAME}' ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# --- Load data ---
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(docs)} law chunks")

# --- Embedding model ---
print(f"Loading embedding model: {MODEL_NAME}")
embedder = TextEmbedding(MODEL_NAME)
batch_size = 64

# --- Upload to Pinecone ---
for i in tqdm(range(0, len(docs), batch_size), desc="Uploading to Pinecone"):
    batch = docs[i:i+batch_size]
    texts = [d["text"] for d in batch]
    ids = [normalize_id(d["id"]) for d in batch]
    embs = list(embedder.embed(texts))

    vectors = []
    for j, d in enumerate(batch):
        vectors.append({
            "id": ids[j],
            "values": embs[j],
            "metadata": {
                "title": d.get("title", ""),
                "text": truncate_text(d["text"]),
                "law": d["meta"].get("law", ""),
                "article_id": str(d["meta"].get("article_id", "")),
            }
        })

    # Upsert batch
    index.upsert(vectors=vectors)

print("Upload complete! All law chunks are now safely stored in Pinecone.")
