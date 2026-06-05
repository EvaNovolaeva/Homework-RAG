import json
from pathlib import Path

RAW_PATH = Path("data/raw/datasets.json")
OUT_PATH = Path("data/processed/documents.jsonl")


def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = data.get("datasets", [])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i, item in enumerate(datasets, start=1):
            doc = {
                "doc_id": item.get("id", f"doc_{i:03d}"),
                "name": item.get("title", f"Document {i}"),
                "source_file": str(RAW_PATH),
                "text": item.get("text", "")
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Created {OUT_PATH} with {len(datasets)} documents")


if __name__ == "__main__":
    main()
