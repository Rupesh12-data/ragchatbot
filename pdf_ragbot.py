

import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from llama_parse import LlamaParse

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

LLAMA_PARSE_KEY = os.getenv("LLAMA_PARSE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

import os
import tempfile
import streamlit as st

from create_llamaparse import parse_s3_pdf_to_markdown_table

st.markdown(
    """
    <h1 style="text-align:center;">
          Parsed PDF
    </h1>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

def chunk_text(markdown: str):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return text_splitter.split_text(markdown)

if "markdown" not in st.session_state:
    st.session_state.markdown = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:

        # Parse only once
        if st.session_state.markdown is None:

            markdown = parse_s3_pdf_to_markdown_table(tmp_path)

            if markdown:

                st.session_state.markdown = markdown
                st.session_state.chunks = chunk_text(markdown)

                st.success(" Successfully!")

            else:

                st.error("❌ Failed to parse PDF.")

    finally:

        os.unlink(tmp_path)

# ---------------------------------
# Buttons
# ---------------------------------
if st.session_state.markdown is not None:

    col1, col2 = st.columns(2)

    with col1:

     if st.button(" Parsed PDF"):

        if st.session_state.get("show") == "parse":
            st.session_state.show = None      # Close it
        else:
            st.session_state.show = "parse"   # Open it


    with col2:

     if st.button(" Chunks"):

        if st.session_state.get("show") == "chunks":
            st.session_state.show = None      # Close it
        else:
            st.session_state.show = "chunks"  # Open it
    if st.session_state.get("show") == "parse":

        st.markdown("##  Parsed PDF")

        st.text_area(
            "",
            st.session_state.markdown,
            height=600,
            label_visibility="collapsed"
        )

    if st.session_state.get("show") == "chunks":

        st.markdown("##  All Chunks")

        st.write(f"Total Chunks : {len(st.session_state.chunks)}")

        for i, chunk in enumerate(st.session_state.chunks, start=1):

            st.markdown(f"### Chunk {i}")

            st.text_area(
                "",
                chunk,
                height=180,
                key=f"chunk_{i}",
                label_visibility="collapsed"
            )

            st.divider()
# Chat Box (always at the bottom)
col1, col2 = st.columns([8, 1])

with col1:
    question = st.text_input(
        "",
        placeholder="Ask anything about the PDF...",
        label_visibility="collapsed"
    )

with col2:
    ask = st.button("⬆")



# -----------------------
# Load Embedding Model
# -----------------------

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model

embedding_model = load_embedding_model()

# -----------------------
# Create Vector DB from PDF Chunks
# -----------------------

@st.cache_resource
def create_vector_db(chunks):

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, embeddings




if "vector_db" not in st.session_state and st.session_state.chunks is not None:

    vector_db, embeddings = create_vector_db(st.session_state.chunks)

    st.session_state.vector_db = vector_db

# -----------------------
# Retrieve Top K chunks
# -----------------------

def retrieve_chunks(query, vector_db, chunks, k=3):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = vector_db.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]

    return "\n\n".join(results)

if ask and question.strip() and st.session_state.get("vector_db"):

    with st.spinner("Searching..."):

        context = retrieve_chunks(
            question,
            st.session_state.vector_db,
            st.session_state.chunks,
            k=3
        )


if ask and question.strip() and st.session_state.get("vector_db"):

    with st.spinner("Searching..."):

        context = retrieve_chunks(
            question,
            st.session_state.vector_db,
            st.session_state.chunks,
            k=3
        )

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.
If the answer is not found, say "I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner("Generating answer..."):

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

    answer = response.text

    st.subheader("Answer")
    st.write(answer)

