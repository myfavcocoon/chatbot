# Vietnamese Business Law AI Assistant

A research and prototype repository for building a Vietnamese business law assistant with retrieval-augmented generation (RAG), ensemble retrieval, and evaluation tooling. The repo includes data preparation notebooks, model training/evaluation, and Gradio app prototypes for serving the assistant.

## Key Capabilities

- RAG with BM25 + vector stores (e.g., Pinecone) via `ensemble_retriever.py`
- Decontextualization and logic modules for safer answers
- Data generation and cleaning notebooks
- Automated evaluation for groundedness and quality (BERTScore, etc.)
- Gradio app notebooks for quick demos and a simple server variant

## Quick Start

- Requirements: Python 3.10+, Windows PowerShell 5.1 (default), Git
- Recommended: virtual environment (venv or conda)

```powershell
# From the repository root
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# If you maintain a requirements file, add it here (example):
# pip install -r requirements.txt

# Common scientific stack (adjust as needed)
pip install gradio pandas numpy scikit-learn matplotlib seaborn jupyter
pip install torch transformers sentencepiece accelerate
pip install rank_bm25 pinecone-client python-dotenv tqdm
pip install bert-score
```

If you use conda, adapt activation and package steps accordingly.

## Project Structure

```text
app_gradio_server.ipynb      # Minimal server-style Gradio demo
app_gradio.ipynb             # Interactive Gradio app for the assistant
data_viz.ipynb               # Visualizations and sanity checks
model_evaluation.ipynb       # General model eval (quality metrics)
rag_evaluation.ipynb         # RAG-specific evaluation (groundedness, etc.)
train_models.ipynb           # Fine-tuning / training workflows
README.md                    # Project overview and usage

data/                        # Prepared datasets and artifacts
data_generating/             # Notebooks to generate/clean data
eval_result/                 # Saved evaluation outputs

src/
	__init__.py
	bm25_manager.py            # BM25 retrieval setup and queries
	config.py                  # Central configuration (paths, keys, params)
	decontextualizer.py        # Context cleaning / decontextualization routines
	ensemble_retriever.py      # Combine BM25 and vector retrieval
	logic_module.py            # Guardrails / reasoning helpers
	model_loader.py            # Load local/HF models, embeddings
	pinecone_manager.py        # Pinecone index helpers
	update_db.py               # Build/update keyword/paragraph DBs
```

## Configuration

- Edit `src/config.py` to set paths, model names, and retrieval parameters.
- For Pinecone or other services, set environment variables (e.g., `PINECONE_API_KEY`) or use a `.env` file.

```powershell
# Example .env creation
"PINECONE_API_KEY=your_key_here" | Out-File -Encoding utf8 .env
"PINECONE_ENV=your_env_here" | Add-Content .env
```

## Data

-
- Core datasets are under `data/`. Notable files:
  - `keywords_db.jsonl`, `updated_kw.jsonl`, `updated_pc.jsonl`: keyword and paragraph DBs
  - `rag_questions*.jsonl`: RAG queries and contexts
  - `eval_data.csv`, `rag_answer.csv`, `rag_final_answer.csv`: evaluation inputs/outputs
  - `vietnamese-stopwords.txt`: stopword list for retrieval
- Use `data_generating/*.ipynb` to produce or refresh derived datasets.

## Run the Gradio App

Two options exist depending on your preference:

1) Interactive notebook: open `app_gradio.ipynb` and run all cells.

2) Server-style notebook: open `app_gradio_server.ipynb` and run the serving cell. If needed, export the app to a Python script.

Typical pattern inside the notebooks is to initialize retrievers and model, then launch a Gradio interface. If you convert to a script, you can run:

```powershell
python app_gradio.py
```

## Retrieval & RAG Pipeline

- `bm25_manager.py`: builds BM25 indices using Vietnamese stopwords and tokenization heuristics.
- `pinecone_manager.py`: utilities to create/update a Pinecone index for vector retrieval.
- `ensemble_retriever.py`: merges BM25 and vector retrieval, ranks/filters results, and returns contexts.
- `decontextualizer.py`: cleans and normalizes retrieved text for better grounding.
- `logic_module.py`: applies lightweight checks (e.g., refusal on illegal/harmful prompts) and answer shaping.

## Evaluation

- `model_evaluation.ipynb`: generic evaluation using metrics like BERTScore.
- `rag_evaluation.ipynb`: evaluates groundedness, context adherence, and response quality specifically for RAG.
- Outputs are saved to `eval_result/` (e.g., `*_bertscore.csv`, `*_groundedness.csv`).

To run evaluations:

```powershell
# Launch Jupyter and open the evaluation notebooks
python -m jupyter lab
# or
python -m notebook
```

## Training

- `train_models.ipynb` provides a starting point for fine-tuning or instruction-tuning.
- `model_loader.py` centralizes model and tokenizer loading across experiments.

## Common Workflows


- Update keyword DBs:

```powershell
python -m src.update_db
```

- Index content to Pinecone (inside a notebook or script):
  - Set `PINECONE_API_KEY` and environment.
  - Use helpers in `src/pinecone_manager.py` to upsert embeddings.

- Build BM25 index:
  - Use utilities in `src/bm25_manager.py` with `vietnamese-stopwords.txt`.

## Notes on Vietnamese Language Handling

- Tokenization and stopword filtering are critical for BM25; adjust stopwords and tokenization rules for your corpus.
- Ensure Unicode normalization when reading/writing text files.
- For transformers-based models, verify Vietnamese support (e.g., Qwen, LLaMA variants).

## Safety and Scope

- This assistant targets Vietnamese business law topics. It is not a substitute for professional legal counsel.
- The `logic_module.py` and evaluation notebooks include basic guardrails; extend them before production use.

## Troubleshooting

- Missing packages: run `pip install` for any import errors.
- Pinecone auth errors: check `.env` and environment variables.
- Encoding issues: use UTF-8; on Windows PowerShell, prefer `Out-File -Encoding utf8` when creating files.

## License

See `LICENSE` in the repository root.
