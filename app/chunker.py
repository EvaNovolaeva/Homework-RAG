import json
from pathlib import Path


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs
    text = text.strip()
    return [text] if text else []


def chunk_text(text: str, max_chars: int = 300, overlap: int = 50) -> list[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else current + "\n\n" + paragraph

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)

                if overlap > 0:
                    tail = current[-overlap:]
                    current = tail + "\n\n" + paragraph
                else:
                    current = paragraph
            else:
                start = 0
                step = max_chars - overlap if max_chars > overlap else max_chars
                while start < len(paragraph):
                    chunk = paragraph[start:start + max_chars]
                    chunks.append(chunk)
                    start += step
                current = ""

    if current:
        chunks.append(current)

    return chunks


def build_chunks(
    documents_path: str = "data/processed/documents.jsonl",
    chunks_path: str = "data/processed/chunks.jsonl",
    max_chars: int = 300,
    overlap: int = 50,
) -> int:
    in_path = Path(documents_path)
    out_path = Path(chunks_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin, start=1):
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            text = doc.get("text", "")
            name = doc.get("name", f"Document {line_idx}")
            source_file = doc.get("source_file", "")

            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

            for chunk_idx, chunk in enumerate(chunks):
                row = {
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "doc_id": doc_id,
                    "name": name,
                    "source_file": source_file,
                    "text": chunk,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_chunks += 1

    return total_chunks


if __name__ == "__main__":
    total = build_chunks()
    print(f"Created data/processed/chunks.jsonl with {total} chunks")
