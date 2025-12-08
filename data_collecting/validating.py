#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_data.py
Automatic data validation + visualization for embedded_laws.jsonl
Generates:
    - Histogram + boxplot of vector norms
    - Cosine similarity heatmap for duplicate detection
    - Pie chart + bar chart distribution of law_title
    - HTML Dashboard
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# CONFIG
# ================================
INPUT_JSONL = "embedded_laws.jsonl"
OUT_DIR = "validation_report"
os.makedirs(OUT_DIR, exist_ok=True)


# ================================
# LOAD DATA
# ================================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


print("Loading JSONL...")
data = load_jsonl(INPUT_JSONL)

df = pd.DataFrame([{
    "id": d["id"],
    "text": d["metadata"]["text"],
    "law_title": d["metadata"].get("law_title", ""),
    "embedding": np.array(d["values"], dtype=np.float32)
} for d in data])

print(f"→ Loaded {len(df)} entries.")


# ================================
# 1. VECTOR STATISTICS
# ================================
print("Computing vector norms...")
df["norm"] = df["embedding"].apply(lambda x: np.linalg.norm(x))

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df["norm"], bins=50)
plt.title("Embedding Vector Norm Distribution")
plt.xlabel("Norm")
plt.ylabel("Count")
hist_path = f"{OUT_DIR}/hist_embeddings.png"
plt.savefig(hist_path, dpi=150)
plt.close()

# Boxplot
plt.figure(figsize=(6,4))
plt.boxplot(df["norm"])
plt.title("Boxplot of Vector Norms")
box_path = f"{OUT_DIR}/box_embeddings.png"
plt.savefig(box_path, dpi=150)
plt.close()


# ================================
# 2. COSINE SIMILARITY HEATMAP (duplicate detection)
# ================================
print("Computing cosine similarity matrix (may take time)...")

# sample up to 200 rows to avoid giant matrix
sample_df = df.sample(min(200, len(df)), random_state=42)
emb_matrix = np.vstack(sample_df["embedding"].to_numpy())
sim_matrix = cosine_similarity(emb_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(sim_matrix, cmap="coolwarm")
plt.title("Cosine Similarity Heatmap (Duplicate Detection)")
heatmap_path = f"{OUT_DIR}/heatmap_similarity.png"
plt.savefig(heatmap_path, dpi=150)
plt.close()


# ================================
# 3. DISTRIBUTION OF LAW TITLES
# ================================
print("Plotting law_title distributions...")

law_counts = df["law_title"].value_counts()

# Pie Chart
plt.figure(figsize=(8,8))
plt.pie(law_counts, labels=law_counts.index, autopct="%1.1f%%")
plt.title("Distribution of Law Titles")
pie_path = f"{OUT_DIR}/pie_law_title.png"
plt.savefig(pie_path, dpi=150)
plt.close()

# Bar Chart
plt.figure(figsize=(10,6))
law_counts.plot(kind="bar")
plt.title("Number of Clauses per Law")
plt.xlabel("Law Title")
plt.ylabel("Count")
bar_path = f"{OUT_DIR}/bar_law_title.png"
plt.tight_layout()
plt.savefig(bar_path, dpi=150)
plt.close()


# ================================
# 4. SAVE BASIC STATS
# ================================
stats = {
    "total_items": len(df),
    "avg_norm": float(df["norm"].mean()),
    "min_norm": float(df["norm"].min()),
    "max_norm": float(df["norm"].max()),
    "law_title_counts": law_counts.to_dict()
}

with open(f"{OUT_DIR}/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)


# ================================
# 5. GENERATE HTML DASHBOARD
# ================================
HTML_TEMPLATE = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Data Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        h1 {{
            color: #333;
        }}
        img {{
            max-width: 600px;
            margin: 20px 0;
        }}
    </style>
</head>

<body>
<h1>Data Validation Dashboard</h1>

<h2>1. Embedding Vector Norm Distribution</h2>
<img src="hist_embeddings.png" />
<img src="box_embeddings.png" />

<h2>2. Cosine Similarity Heatmap (Duplicate Detection)</h2>
<img src="heatmap_similarity.png" />

<h2>3. Law Title Distribution</h2>
<h3>Pie Chart</h3>
<img src="pie_law_title.png" />
<h3>Bar Chart</h3>
<img src="bar_law_title.png" />

<h2>4. Summary Stats</h2>
<pre>{json.dumps(stats, indent=2, ensure_ascii=False)}</pre>

</body>
</html>
"""

with open(f"{OUT_DIR}/index.html", "w", encoding="utf-8") as f:
    f.write(HTML_TEMPLATE)

print("Validation report generated at:", OUT_DIR)
