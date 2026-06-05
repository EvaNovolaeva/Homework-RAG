import json
import pickle
from pathlib import Path

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


INDEX_DIR = Path("data/index")
VECTORIZER_PATH = INDEX_DIR / "vectorizer.pkl"
MATRIX_PATH = INDEX_DIR / "matrix.npz"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"


def load_chunks(chunks_path: Path = CHUNKS_PATH):
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_index():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    matrix = load_npz(MATRIX_PATH)
    chunks = load_chunks()

    return vectorizer, matrix, chunks


def retrieve(query: str, top_k: int = 3):
    vectorizer, matrix, chunks = load_index()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "doc_id": chunk["doc_id"],
            "name": chunk.get("name", ""),
            "text": chunk["text"],
            "score": float(scores[idx]),
        })

    return results
