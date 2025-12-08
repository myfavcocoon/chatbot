import os
import json
import re
from collections import defaultdict

input_folder = "laws_output1"
output_folder = "processed_laws1"
os.makedirs(output_folder, exist_ok=True)

merged_data = []
valid_files = 0
skipped_files = 0
seen_keys = set()
law_articles_map = defaultdict(list)

# ========================
# CLEAN TEXT LOGIC
# ========================
def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'^\.\s*', '', text)

    replacements = {
        r'\s*;\s*': '; ',
        r'\s*,\s*': ', ',
        r'\s*\.\s*': '. ',
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'\.\.+', '.', text)

    return text.strip()


# ========================
# SPLIT KHOẢN (1., 2., 3...)
# ========================
def split_clauses(text: str):
    if not text:
        return []

    # Giữ nguyên text, KHÔNG convert newline
    t = text

    # Remove  "." ở đầu nếu có
    t = re.sub(r'^\s*\.\s*', '', t)

    # Regex tách khoản trong 1 dòng luôn
    # Match "1. ", "2. ", "10. "...
    matches = list(re.finditer(r'(\d+)\.\s', t))
    results = []

    if not matches:
        # Không có khoản → cả điều là clause 0
        main_clause = t.strip()
        if main_clause:
            results.append(("0", main_clause))
        return results

    # Lấy phần clause chính (trước 1.)
    first_start = matches[0].start()
    main_clause = t[:first_start].strip()
    if main_clause:
        results.append(("0", main_clause))

    # Tách từng khoản
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        chunk = t[start:end].strip()
        clause_no = m.group(1)
        results.append((clause_no, chunk))

    return results

# =======================
# READ & MERGE JSONL FILES
# =======================
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".jsonl"):
        continue

    file_path = os.path.join(input_folder, filename)
    law_title = os.path.splitext(filename)[0]

    try:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            buffer = ""
            for line in f:
                line = line.strip()
                if not line:
                    continue

                buffer += line
                try:
                    item = json.loads(buffer)
                    data.append(item)
                    buffer = ""
                except json.JSONDecodeError:
                    buffer += " "
                    continue

        if not data:
            skipped_files += 1
            print(f"Skipped {filename} (empty or invalid)")
            continue

        # Remove first & last line
        if len(data) > 2:
            data = data[1:-1]

        clean_data = []

        for article in data:
            clause_text = article.get("clause_text", "")
            article_title = article.get("article_title", "")

            # Only clean, do not remove articles
            article["clause_text"] = clean_text(clause_text)
            article["article_title"] = clean_text(article_title)

            clean_data.append(article)

        if not clean_data:
            skipped_files += 1
            print(f"Skipped {filename} (no valid content)")
            continue

        # Add to map
        for idx, article in enumerate(clean_data):
            key = f"{law_title}_{idx}"
            if key not in seen_keys:
                seen_keys.add(key)
                law_articles_map[law_title].append(article)

        valid_files += 1
        print(f"Processed {filename} ({len(clean_data)} cleaned articles)")

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        skipped_files += 1


# =======================
# RE-NUMBER article_id
# AND SPLIT CLAUSES
# =======================
final_output = []
global_id = 1

for law_title, articles in law_articles_map.items():
    for article in articles:
        article_id = global_id
        global_id += 1

        # Split clause_text into clause 0,1,2,3...
        splits = split_clauses(article.get("clause_text", ""))

        for clause_no, text in splits:
            cleaned_clause = clean_text(text)

            final_output.append({
                "law_title": article.get("law_title"),
                "article_id": article_id,
                "article_title": article.get("article_title"),
                "article_link": article.get("article_link"),
                "clause_no": int(clause_no) if clause_no.isdigit() else clause_no,
                "clause_text": cleaned_clause
            })


# =======================
# WRITE FINAL OUTPUT JSONL
# =======================
output_path = os.path.join(output_folder, "all_laws_merged_clean_split.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for item in final_output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n==============================")
print("DONE! MERGE + CLEAN + SPLIT hoàn tất.")
print(f"{valid_files} files processed, {skipped_files} skipped.")
print(f"Tổng số khoản (clause) output: {len(final_output)}")
print(f"Output saved to: {output_path}")
print("==============================")
