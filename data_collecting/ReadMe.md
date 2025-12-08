# RAG_LAW_2025

Hệ thống Tìm kiếm & Hỏi đáp Luật Việt Nam sử dụng RAG + Embedding

RAG_LAW_2025 là dự án xây dựng nền tảng tìm kiếm và hỏi đáp dựa trên văn bản luật Việt Nam.
Dự án gồm các bước: thu thập dữ liệu luật, xử lý – chuẩn hóa văn bản, tách điều khoản, sinh embedding, lưu trữ vector database và cuối cùng là xây dựng pipeline RAG trả lời câu hỏi có trích dẫn.

---

## Mục tiêu chính

* Thu thập và chuẩn hóa dữ liệu văn bản luật.
* Tách điều, khoản, điểm bằng regex và logic phân tích văn bản.
* Gắn link văn bản gốc cho từng điều luật.
* Chunk dữ liệu tối ưu cho RAG.
* Sinh embedding bằng mô hình mạnh (bge-m3 / VinaLaw / OpenAI).
* Lưu vào VectorStore (FAISS / Chroma).
* Tạo pipeline RAG trả lời câu hỏi có trích dẫn theo đúng điều luật.

---

## Cấu trúc thư mục dự án

```
RAG_LAW_2025/
│── laws_output11t/           # Dữ liệu crawl thô
│── processed_laws_merged/    # Dữ liệu đã xử lý & tách điều luật
│── crawl.py                  # Script crawl đơn luồng
│── crawl_multi.py            # Script crawl đa luồng
│── elastic_upload.py         # Upload dữ liệu lên ElasticSearch
│── pinecone_upload_local.py  # Upload embedding lên Pinecone
│── preprocess.py             # Xử lý văn bản phiên bản đơn
│── preprocess_multi.py       # Xử lý văn bản phiên bản đa luồng
│── requirements.txt          # Thư viện yêu cầu
│── ReadMe.md                 # Tài liệu dự án
```

---

## Cài đặt

### 1. Clone dự án

```
git clone https://github.com/.../RAG_LAW_2025.git
cd RAG_LAW_2025
```

### 2. Cài đặt thư viện

```
pip install -r requirements.txt
```

---

## Quy trình chạy hệ thống

### **1. Crawl dữ liệu luật**

```
python crawl_multi.py
```

> Thu thập file văn bản (txt/html) từ nguồn hoặc thư mục local.

---

### **2. Xử lý – làm sạch văn bản**

```
python process_text.py
```

* Chuẩn hóa text
* Xóa ký tự thừa
* Chuẩn hóa tiêu đề, số điều

---

### **3. Tách điều – khoản – điểm**

```
processed_laws_merged/{law_name}.json
```

Mỗi điều gồm:

* `article_number`
* `text`
* `law_name`
* `link`

---

### **6. Chạy hệ thống RAG để hỏi đáp**

Hệ thống sẽ trả về:

* Câu trả lời
* Điều khoản liên quan
* Link văn bản gốc
* Trích dẫn đầy đủ

---

## Công nghệ sử dụng

* Python 3.10+
* BeautifulSoup4, lxml (crawl & parse HTML)
* Regex-based text parsing
* SentenceTransformers BGE-m3 
* Pinecone & Elasticsearch

---


## Tác giả

Dự án RAG_LAW_2025 – 2025
