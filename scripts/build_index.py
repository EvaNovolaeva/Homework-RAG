import json
import pickle
import shutil
from pathlib import Path

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from app.chunker import build_chunks
from scripts.ingest import main as ingest_main


INDEX_DIR = Path("data/index")
PROCESSED_DIR = Path("data/processed")
CHUNKS_PROCESSED_PATH = PROCESSED_DIR / "chunks.jsonl"
CHUNKS_INDEX_PATH = INDEX_DIR / "chunks.jsonl"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.pkl"
MATRIX_PATH = INDEX_DIR / "matrix.npz"


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    ingest_main()
    build_chunks()

    texts = []

    with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    save_npz(MATRIX_PATH, matrix)
    shutil.copyfile(CHUNKS_PROCESSED_PATH, CHUNKS_INDEX_PATH)

    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved matrix to {MATRIX_PATH}")
    print(f"Saved chunks to {CHUNKS_INDEX_PATH}")
    print(f"Matrix shape: {matrix.shape}")


if __name__ == "__main__":
    main()
