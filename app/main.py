from pathlib import Path

import streamlit as st

from app.generator import generate_answer
from app.retriever import retrieve


INDEX_FILES = [
    Path("data/index/vectorizer.pkl"),
    Path("data/index/matrix.npz"),
    Path("data/index/chunks.jsonl"),
]

DEMO_QUESTIONS = [
    "How are stress and anxiety related to social media use and sleep?",
    "Which profiles show high stress and high depression risk?",
    "What patterns are visible for teenagers with low stress?",
    "What is the capital of France?",
]


def index_exists():
    return all(path.exists() for path in INDEX_FILES)


st.set_page_config(page_title="Homework RAG Demo", layout="wide")

st.title("Homework RAG Demo")
st.write("Ask a question about the demo teen mental health dataset.")

if not index_exists():
    st.warning("Index is not built yet. Please run: uv run python scripts/build_index.py")
    st.stop()

st.subheader("Demo questions")
for q in DEMO_QUESTIONS:
    st.write(f"- {q}")

query = st.text_input("Your question", value=DEMO_QUESTIONS[0])

if st.button("Ask"):
    answer_result = generate_answer(query, top_k=3, min_score=0.1)
    retrieved = retrieve(query, top_k=3)

    st.subheader("Answer")
    if answer_result["refused"]:
        st.error(answer_result["answer"])
    else:
        st.success(answer_result["answer"])

    st.subheader("Sources")
    if answer_result["sources"]:
        for src in answer_result["sources"]:
            st.write(
                f"- doc_id={src['doc_id']} | name={src['name']} | score={src['score']:.4f}"
            )
    else:
        st.write("No sources used.")

    st.subheader("Top-k fragments")
    for i, item in enumerate(retrieved, start=1):
        with st.expander(
            f"{i}. {item['doc_id']} | score={item['score']:.4f} | {item['name']}"
        ):
            st.write(item["text"])
