import json
from pathlib import Path

from app.chunker import chunk_text, build_chunks


def test_chunk_text_returns_list():
    text = "Paragraph one.\n\nParagraph two."
    chunks = chunk_text(text, max_chars=50, overlap=10)

    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_text_respects_max_chars_for_normal_input():
    text = "Short paragraph one.\n\nShort paragraph two.\n\nShort paragraph three."
    chunks = chunk_text(text, max_chars=40, overlap=10)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 52


def test_build_chunks_creates_file(tmp_path):
    docs_path = tmp_path / "documents.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"

    docs = [
        {
            "doc_id": "doc1",
            "name": "Doc 1",
            "source_file": "source.json",
            "text": "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        }
    ]

    with open(docs_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total = build_chunks(
        documents_path=str(docs_path),
        chunks_path=str(chunks_path),
        max_chars=30,
        overlap=5,
    )

    assert total >= 1
    assert chunks_path.exists()

    lines = chunks_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == total

    first = json.loads(lines[0])
    assert "doc_id" in first
    assert first["doc_id"] == "doc1"
