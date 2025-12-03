import json
import os
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import BASE_DIR, EMBED_MODEL_NAME, PINECONE_API_KEY, PINECONE_INDEX_NAME
from tqdm import tqdm  # tiến trình

# -----------------------
# Data directory
# -----------------------
DATA_DIR = os.path.join(BASE_DIR, "data")

# -----------------------
# Pinecone init
# -----------------------
load_dotenv()  # load env cho Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_pinecone = pc.Index(PINECONE_INDEX_NAME)

# -----------------------
# Embedding model
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✔ Sử dụng device: {device}")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

def embed_text(text: str):
    return embedding_model.encode(text).tolist()

# -----------------------
# Upload lên Pinecone
# -----------------------
def update_pinecone(jsonl_filename: str):
    """
    Đọc file JSONL từ DATA_DIR, embed clause_text, upsert lên Pinecone
    """
    jsonl_path = os.path.join(DATA_DIR, jsonl_filename)

    if not os.path.exists(jsonl_path):
        print(f"❌ File {jsonl_filename} không tồn tại!")
        return

    vectors = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"✔ Đang upload {len(lines)} vectors từ {jsonl_filename} lên Pinecone...")

    for i, line in enumerate(tqdm(lines, desc="Embedding & chuẩn bị dữ liệu")):
        obj = json.loads(line)

        vector_id = obj.get("id")
        if not vector_id:
            print(f"⚠ Bỏ qua dòng {i} vì thiếu 'id'")
            continue

        clause_text = obj.get("clause_text", "")

        # Embedding
        emb = embed_text(clause_text)

        # Metadata: giữ toàn bộ trừ clause_id
        metadata = {k: v for k, v in obj.items() if k != "clause_id"}

        vectors.append({
            "id": vector_id,
            "values": emb,
            "metadata": metadata
        })

        if (i + 1) % 50 == 0:
            print(f"  -> Đã xử lý {i + 1}/{len(lines)} vectors")

    if vectors:
        index_pinecone.upsert(vectors=vectors)
        print(f"✔ Hoàn tất upload {len(vectors)} vectors từ {jsonl_filename} lên Pinecone")
    else:
        print("⚠ Không có vectors nào để upload!")

# -----------------------
# Update keywords_db
# -----------------------
def update_keywords_db(source_filename: str, target_filename: str):
    """
    Append dữ liệu từ source JSONL vào target JSONL trong DATA_DIR
    """
    source_path = os.path.join(DATA_DIR, source_filename)
    target_path = os.path.join(DATA_DIR, target_filename)

    if not os.path.exists(source_path):
        print(f"❌ File nguồn {source_filename} không tồn tại!")
        return

    mode = "a" 

    with open(source_path, "r", encoding="utf-8") as infile, \
         open(target_path, mode, encoding="utf-8") as outfile:

        lines = infile.readlines()
        print(f"✔ Đang thêm {len(lines)} dòng từ {source_filename} vào {target_filename}...")

        for i, line in enumerate(tqdm(lines, desc="Append keywords")):
            obj = json.loads(line)
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  -> Đã ghi {i + 1}/{len(lines)} dòng")

    print(f"✔ Hoàn tất cập nhật {source_filename} vào {target_filename}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    update_pinecone("updated_pc.jsonl")
    update_keywords_db("updated_kw.jsonl", "keywords_db.jsonl")
