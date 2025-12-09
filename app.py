import os
import tempfile
import re
import chromadb
import ollama
import streamlit as st
import pandas as pd
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ----------------------------- #
# ENHANCED SYSTEM PROMPT
# ----------------------------- #
system_prompt = """
You are an advanced AI assistant specializing in deep contextual reasoning and structured explanations.

OBJECTIVE
Provide a detailed, logically structured, and factual answer using ONLY the provided context.

RESPONSE FORMAT (MANDATORY)
Always respond in this exact structure and order. Do not add, remove, rename, or reorder sections.

🧾 Answer:
Overview: <2–4 sentences summarizing the requested result.>
Key Details:
- <Bullet 1>
- <Bullet 2>
- <Bullet 3>
Step-by-step Reasoning:
1. <Step 1>
2. <Step 2>
3. <Step 3>
Final Conclusion: <1–2 sentences with the direct conclusion.>

TABLE RULES
- If multiple models/items are present or comparison is requested, include a table below the Overview with the most relevant columns for the question.
- Table headers should be clear and concise. No citations inside table cells.
- After the table, continue with Key Details, Step-by-step Reasoning, and Final Conclusion sections exactly as above.

INSTRUCTIONS
1. Read the context carefully and identify all relevant information.
2. Synthesize insights—combine related pieces from different sections.
3. Cover all relevant fields (e.g., RAM, ROM, Power, Certifications).
4. Prefer completeness over brevity. Avoid guessing; state explicitly if context is insufficient.
5. Use professional, academic tone.
"""

# ----------------------------- #
# DOCUMENT PROCESSING
# ----------------------------- #
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes uploaded documents (PDF, DOCX, TXT, CSV) into smaller text chunks."""
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=file_ext, delete=False)
    try:
        temp_file.write(uploaded_file.read())
        temp_file.close()

        if file_ext == ".pdf":
            loader = PyMuPDFLoader(temp_file.name)
        elif file_ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(temp_file.name)
        elif file_ext == ".txt":
            loader = TextLoader(temp_file.name)
        elif file_ext == ".csv":
            df = pd.read_csv(temp_file.name)
            docs = [
                Document(page_content=row.to_string(), metadata={"row": i})
                for i, row in df.iterrows()
            ]
            return docs
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        docs = loader.load()

    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

# ----------------------------- #
# VECTOR STORE
# ----------------------------- #
def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    st.success(f"✅ {file_name} successfully embedded into the knowledge base!")

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

# ----------------------------- #
# LLM CALL + FORMAT FIXER
# ----------------------------- #
def call_llm_collect(context: str, prompt: str, detail_level: str = "detailed") -> str:
    """Call ollama in streaming mode, collect all chunks and return the final text."""
    mode = "Provide a concise answer." if detail_level == "concise" else "Provide an extremely detailed and structured answer."

    response = ollama.chat(
        model="mistral:latest",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
{mode}

You are given structured product specifications.
- Always cover all relevant fields (e.g., RAM, ROM, Power, Certifications).
- If multiple models are present, compare them in a table.
- Respond ONLY in the exact format defined below. Do not include any extra sections or text before or after.

REQUIRED FORMAT (exactly):
🧾 Answer:
Overview: <2–4 sentences>
Key Details:
- <Bullet 1>
- <Bullet 2>
- <Bullet 3>
Step-by-step Reasoning:
1. <Step 1>
2. <Step 2>
3. <Step 3>
Final Conclusion: <1–2 sentences>

Context:
{context}

Question:
{prompt}
"""}
        ],
    )

    pieces = []
    for chunk in response:
        # chunk can be str or dict depending on ollama client; handle both safely
        if isinstance(chunk, dict) and chunk.get("message"):
            pieces.append(chunk["message"].get("content", ""))
        elif isinstance(chunk, str):
            pieces.append(chunk)
        # only break when chunk is dict and has done==True
        if isinstance(chunk, dict) and chunk.get("done") is True:
            break

    full_text = "".join(pieces).strip()
    return full_text

def enforce_ollama_format(text: str) -> str:
    """Ensure the returned text strictly follows the required template.
    If the model deviates, try to extract the main pieces or synthesize them heuristically.
    """
    # Normalize unicode and whitespace
    text = re.sub(r"\r\n|\r", "\n", text).strip()

    # Find sections with regex
    pattern = (
        r"🧾\s*Answer:\s*(?:\n)?"
        r"Overview:\s*(?P<overview>.*?)\n\s*Key Details:\s*(?P<key>.*?)\n\s*Step-by-step Reasoning:\s*(?P<steps>.*?)\n\s*Final Conclusion:\s*(?P<final>.*)"
    )
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    if m:
        overview = m.group("overview").strip()
        key = m.group("key").strip()
        steps = m.group("steps").strip()
        final = m.group("final").strip()

        # Clean bullets and ordered lists
        key_lines = [line.strip() for line in re.split(r"\n|\r", key) if line.strip()]
        # ensure bullets start with '- '
        pattern_bullets = r"^[-*\s]+"
        key_formatted = "\n".join([f"- {re.sub(pattern_bullets, '', l)}" for l in key_lines[:10]])

        # Steps: extract numbered lines
        steps_lines = re.findall(r"\d+\.\s*(.+)", steps)
        if not steps_lines:
            # fallback: split by sentences
            steps_lines = re.split(r"(?<=\.)\s+", steps)[:3]
        steps_formatted = "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(steps_lines[:10])])

        final_line = final.split("\n")[0].strip()

        assembled = (
            "🧾 Answer:\n"
            f"Overview: {overview}\n\n"
            "Key Details:\n"
            f"{key_formatted}\n\n"
            "Step-by-step Reasoning:\n"
            f"{steps_formatted}\n\n"
            f"Final Conclusion: {final_line}"
        )
        return assembled

    # If regex failed, build a best-effort template
    # Heuristic: first 2 sentences -> Overview; collect lines starting with '-' for Key; numbered for Steps; last sentence for Final Conclusion
    sentences = re.split(r"(?<=[.!?])\s+", text)
    overview = " ".join(sentences[:2]).strip() if sentences else "Overview not found."

    bullets = re.findall(r"^\s*[-*]\s*(.+)$", text, flags=re.MULTILINE)
    if not bullets:
        # try to find short lines that look like bullets (<=120 chars)
        candidates = [line.strip() for line in text.splitlines() if 0 < len(line.strip()) < 120]
        bullets = candidates[:3]
    key_formatted = "\n".join([f"- {b.strip()}" for b in bullets[:10]]) or "- (no key details found)"

    numbered = re.findall(r"\d+\.\s*(.+)", text)
    if not numbered:
        # fallback: take next 3 sentences after overview
        numbered = sentences[2:5]
    steps_formatted = "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(numbered[:10])]) or "1. (no steps found)"

    final = sentences[-1].strip() if sentences else "(no final conclusion found)"

    assembled = (
        "🧾 Answer:\n"
        f"Overview: {overview}\n\n"
        "Key Details:\n"
        f"{key_formatted}\n\n"
        "Step-by-step Reasoning:\n"
        f"{steps_formatted}\n\n"
        f"Final Conclusion: {final}"
    )

    return assembled

def generate_fixed_style_response(context: str, prompt: str, detail_level: str = "detailed"):
    """Generator that yields the final, format-fixed response so Streamlit can stream it once.

    We collect all chunks from ollama, enforce the required template, and yield the cleaned output as a single final chunk.
    """
    full_text = call_llm_collect(context=context, prompt=prompt, detail_level=detail_level)
    fixed = enforce_ollama_format(full_text)

    # Yield once (st.write_stream expects an iterator of strings)
    yield fixed

# ----------------------------- #
# CROSS ENCODER RE-RANKING
# ----------------------------- #
def re_rank_cross_encoders(query: str, documents: list[str], top_k: int = 50) -> tuple[str, list[int]]:
    """Rerank retrieved documents by relevance using a stronger cross encoder."""
    if not documents:
        return "", []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    ranks = encoder_model.rank(query, documents, top_k=top_k)

    # ranks may be a list of dicts with 'corpus_id'; guard for expected structure
    try:
        relevant_text = "\n\n---\n\n".join(
            [f"### Retrieved Chunk {i+1}\n{documents[rank['corpus_id']]}" for i, rank in enumerate(ranks)]
        )
        relevant_text_ids = [rank["corpus_id"] for rank in ranks]
    except Exception:
        # fallback: return the top results in given order
        relevant_text = "\n\n---\n\n".join([f"### Retrieved Chunk {i+1}\n{doc}" for i, doc in enumerate(documents[:top_k])])
        relevant_text_ids = list(range(min(len(documents), top_k)))

    return relevant_text, relevant_text_ids

# ----------------------------- #
# STREAMLIT APP
# ----------------------------- #
def main():
    st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")
    st.title("🧠 Advanced Document-based Q&A Assistant")

    with st.sidebar:
        st.header("📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload up to 50 files (PDF, DOCX, TXT, CSV)",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True
        )
        process_btn = st.button("⚡ Process and Embed")
        st.markdown("---")
        detail_level = st.radio("Response Depth:", ["detailed", "concise"], index=0)

        if uploaded_files and process_btn:
            with st.spinner("Processing and embedding documents..."):
                for idx, uploaded_file in enumerate(uploaded_files[:50], start=1):
                    normalized = uploaded_file.name.translate(
                        str.maketrans({"-": "_", ".": "_", " ": "_"})
                    )
                    docs = process_document(uploaded_file)
                    add_to_vector_collection(docs, normalized)
                    st.progress(idx / min(len(uploaded_files), 50))
                st.success(f"✅ Successfully processed {min(len(uploaded_files), 50)} files!")

    st.header("💬 Ask Questions About Your Documents")
    prompt = st.text_area("Type your question here:")
    ask_btn = st.button("🔍 Ask")

    if ask_btn and prompt.strip():
        with st.spinner("Retrieving relevant data and generating response..."):
            results = query_collection(prompt, n_results=50)
            context_docs = results.get("documents", [[]])[0]
            relevant_text, relevant_ids = re_rank_cross_encoders(prompt, context_docs, top_k=50)

            # Use the fixed-style generator instead of raw streaming to guarantee format
            response_gen = generate_fixed_style_response(context=relevant_text, prompt=prompt, detail_level=detail_level)
            st.write_stream(response_gen)

            with st.expander("📘 Retrieved Documents"):
                st.write(results)

            with st.expander("🏷️ Most Relevant Chunks"):
                st.write(relevant_ids)
                st.write(relevant_text)

# ----------------------------- #
# ENTRY POINT
# ----------------------------- #
if __name__ == "__main__":
    main()

