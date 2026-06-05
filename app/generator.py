from app.retriever import retrieve


def generate_answer(query: str, top_k: int = 3, min_score: float = 0.1):
    results = retrieve(query, top_k=top_k)

    relevant = [r for r in results if r["score"] > min_score]

    if not relevant:
        return {
            "answer": "I cannot answer this question from the available context.",
            "sources": [],
            "refused": True,
        }

    answer_parts = []
    sources = []

    for item in relevant:
        answer_parts.append(item["text"])
        sources.append({
            "doc_id": item["doc_id"],
            "name": item["name"],
            "score": item["score"],
        })

    answer = "Based on the retrieved context: " + " ".join(answer_parts)

    return {
        "answer": answer,
        "sources": sources,
        "refused": False,
    }
