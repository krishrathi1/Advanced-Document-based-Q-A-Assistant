# Advanced RAG Assistant

A Streamlit application that implements a Retrieval-Augmented Generation (RAG) workflow using local Ollama embeddings & LLMs, ChromaDB for persistent vector storage, and a cross-encoder re-ranker for higher-quality retrieval. The app lets you upload PDFs, DOCX, TXT, and CSV files, embed them into a vector store, and ask document-aware questions. Responses are post-processed to follow a strict structured template.

---

## Features

* Upload and process PDF, DOCX, TXT, and CSV files into text chunks.
* Split documents into overlapping chunks for better retrieval.
* Use Ollama embedding function (`nomic-embed-text`) to generate vector embeddings.
* Store and persist vectors using ChromaDB (PersistentClient).
* Query the vector store and rerank with a CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`).
* Query an Ollama LLM (`mistral:latest`) in streaming mode and enforce a rigid response template.
* Stream the final cleaned answer back to the user via Streamlit.

---

## Repository structure

```
project-root/
├── app.py (or main script that runs the Streamlit app)
├── README.md
├── requirements.txt
├── demo-rag-chroma/  # ChromaDB persistent data path (created at runtime)
└── ...
```

> The main Streamlit file contains functions for document processing, embedding, vector queries, LLM calls, re-ranking, and the UI.

---

## Requirements

* Python 3.9+ (3.10 recommended)
* Ollama (local server) running and accessible at `http://localhost:11434`
* ChromaDB-compatible Python package
* Streamlit
* LangChain/Community loaders used in the app
* sentence-transformers (for CrossEncoder)
* pandas, PyMuPDF, and other file loaders as needed

Example `pip` packages (put these into `requirements.txt`):

```
streamlit
chromadb
ollama
sentence-transformers
langchain-core
langchain-community
pymupdf
unstructured
pandas
python-multipart

tqdm
# Add other required libs depending on your environment/loaders
```

---

## Environment & Pre-requisites

1. **Ollama**

   * Install and run Ollama locally. The app expects the embeddings endpoint at `http://localhost:11434/api/embeddings` and the chat endpoint accessible by the Ollama Python client.
   * Ensure the following models are available locally (or adjust names in the code):

     * `nomic-embed-text:latest` (embedding model)
     * `mistral:latest` (LLM used for responses)

2. **ChromaDB**

   * The code uses `chromadb.PersistentClient(path="./demo-rag-chroma")`. Make sure the process has write permissions to that path.

3. **File loaders**

   * PDF: PyMuPDF (via `PyMuPDFLoader`).
   * DOCX: Unstructured Word loader (`UnstructuredWordDocumentLoader`).
   * Plain text & CSV handled via `TextLoader` and `pandas`.

4. **Cross-Encoder**

   * The re-ranker uses `cross-encoder/ms-marco-MiniLM-L-12-v2` from `sentence-transformers`. This will download weights on first run.

---

## Installation

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # on Windows use venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Ensure Ollama is running and models are installed.
3. Run the Streamlit app:

```bash
streamlit run app.py
# or if your main file is named differently, replace app.py accordingly
```

Open the URL printed by Streamlit (usually `http://localhost:8501`) in your browser.

---

## Usage

1. **Upload documents**

   * Use the sidebar to upload PDF/DOCX/TXT/CSV files. You can upload up to 50 files.
   * Click `Process and Embed` to convert files into chunks and insert them into the ChromaDB collection.

2. **Ask questions**

   * Enter a question in the main UI text area and click `Ask`.
   * The app will query the Chroma collection, rerank with the CrossEncoder, and call the LLM with the top chunks as context.
   * A streaming response will be shown that follows the enforced template.

3. **View retrieved chunks & metadata**

   * Expand the `Retrieved Documents` and `Most Relevant Chunks` sections to inspect raw retrieval output.

---

## Configuration options in code

* `system_prompt`: The high-level system prompt the LLM uses.
* `chunk_size`, `chunk_overlap`, and `separators` in `RecursiveCharacterTextSplitter`.
* Chroma persistent path `./demo-rag-chroma` — change if you prefer a different storage path.
* Ollama `url` and `model_name` for embeddings — update if your Ollama server or model names differ.

---

## Troubleshooting & Tips

* **Ollama connection issues**: Verify the Ollama server is running and accessible at the URL you configured. Use `curl` to test the embeddings endpoint.
* **Slow CrossEncoder downloads**: The `CrossEncoder` model will download weights on first use. If you have limited bandwidth, pre-download the model separately.
* **ChromaDB persistence errors**: Ensure correct filesystem permissions. If the DB becomes corrupted, you can remove the `demo-rag-chroma` folder and re-embed documents.
* **Unsupported file types**: The app raises `ValueError` for unsupported file types. Add more loaders if you need other formats.

---

## Security & Privacy

* Uploaded documents are read and (temporarily) written to a local temporary file for processing.
* The app currently assumes a trusted local environment. Do not expose it directly to the public internet without adding authentication and transport security.

---

## Extending the App

* Add additional re-rankers or different cross-encoders for better precision.
* Add multi-file summarization endpoints to precompute condensed context for long documents.
* Add an admin panel to manage embeddings (delete / re-index specific documents).
* Swap the streaming LLM for a different local or remote model by changing Ollama model names or integrating another LLM client.

---

## License

This repository does not include an explicit license by default. Add a `LICENSE` file if you intend to open source this project.

---

## Contributors

* Original author (you) — application code, prompt engineering, and Streamlit UI

---

If you'd like, I can also generate a `requirements.txt` and a minimal `app.py` wrapper or a Dockerfile for running Ollama + Streamlit together. Just tell me which you'd prefer.
