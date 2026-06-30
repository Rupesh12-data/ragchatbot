

import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from llama_parse import LlamaParse
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

LLAMA_PARSE_KEY = os.getenv("LLAMA_PARSE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"   # change if you want a different Groq model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"     # small, fast, local embeddings
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks
TOP_K = 4               # number of chunks to retrieve per question


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env file.")
        st.stop()
    return Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def parse_pdf_to_markdown(file_path: str, filename: str) -> str:
    """Parse a single PDF into markdown text using LlamaParse."""
    if not LLAMA_PARSE_KEY:
        st.error("LLAMA_PARSE_KEY not found in .env file.")
        st.stop()

    parser = LlamaParse(
        api_key=LLAMA_PARSE_KEY,
        result_type="markdown",
        auto_mode=True,
        auto_mode_trigger_on_table_in_page=True,
        skip_diagonal_text=True,
        disable_ocr=True,
        disable_image_extraction=True,
        do_not_cache=True,
        verbose=False,
    )
    documents = parser.load_data(file_path)
    markdown_text = "\n".join(doc.text for doc in documents)
    return markdown_text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple character-based chunking with overlap. Returns list of dicts."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Vector store (FAISS, in-memory, session-scoped)
# ---------------------------------------------------------------------------

def build_faiss_index(chunks: list[dict]):
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity since vectors are normalized
    index.add(embeddings)
    return index, chunks


def retrieve_chunks(query: str, index, chunks: list[dict], top_k: int = TOP_K):
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, min(top_k, len(chunks)))
    results = []
    for i, score in zip(idxs[0], scores[0]):
        if i == -1:
            continue
        results.append({**chunks[i], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def answer_question(question: str, retrieved_chunks: list[dict], chat_history: list[dict]):
    client = get_groq_client()

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided document excerpts. If the answer isn't in the excerpts, "
        "say you don't know based on the uploaded documents. Cite the source "
        "filename when relevant."
    )

    user_prompt = f"Document excerpts:\n{context}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": system_prompt}]
    # include prior turns for conversational context (keep last few turns only)
    for turn in chat_history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
    st.title("📄 PDF RAG Chatbot")
    st.caption("Upload PDFs, then ask questions about them.")

    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.chunks = None
        st.session_state.processed_files = set()
        st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF file(s)", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Process PDFs", type="primary"):
            all_chunks = []
            with st.spinner("Parsing and chunking PDFs..."):
                for uploaded_file in uploaded_files:
                    file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
                    if file_hash in st.session_state.processed_files:
                        st.info(f"Skipping {uploaded_file.name} (already processed)")
                        continue

                    # Save to a temp file since LlamaParse needs a path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    try:
                        st.write(f"Parsing **{uploaded_file.name}**...")
                        markdown_text = parse_pdf_to_markdown(tmp_path, uploaded_file.name)
                        file_chunks = chunk_text(markdown_text, source=uploaded_file.name)
                        all_chunks.extend(file_chunks)
                        st.session_state.processed_files.add(file_hash)
                        st.success(f"Processed {uploaded_file.name}: {len(file_chunks)} chunks")
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

            if all_chunks:
                with st.spinner("Building search index..."):
                    if st.session_state.chunks:
                        # append to existing index data
                        combined_chunks = st.session_state.chunks + all_chunks
                    else:
                        combined_chunks = all_chunks
                    index, chunks = build_faiss_index(combined_chunks)
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                st.success("Index ready. You can now ask questions!")

        if st.session_state.chunks:
            st.markdown("---")
            st.write(f"**Indexed chunks:** {len(st.session_state.chunks)}")
            sources = sorted(set(c["source"] for c in st.session_state.chunks))
            st.write("**Files:**")
            for s in sources:
                st.write(f"- {s}")

        if st.button("Clear all"):
            st.session_state.index = None
            st.session_state.chunks = None
            st.session_state.processed_files = set()
            st.session_state.chat_history = []
            st.rerun()

    # --- Chat area ---
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        if st.session_state.index is None:
            st.warning("Please upload and process at least one PDF first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retrieved = retrieve_chunks(question, st.session_state.index, st.session_state.chunks)
                    answer = answer_question(question, retrieved, st.session_state.chat_history)
                    st.markdown(answer)

                    with st.expander("Sources used"):
                        for c in retrieved:
                            st.markdown(f"**{c['source']}** (score: {c['score']:.2f})")
                            st.text(c["text"][:300] + "...")

            st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()