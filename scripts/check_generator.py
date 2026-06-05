from app.generator import generate_answer


def print_result(title, result):
    print(f"\n=== {title} ===")
    print("refused:", result["refused"])
    print("answer:", result["answer"])
    print("sources:")
    for src in result["sources"]:
        print(f"  - doc_id={src['doc_id']} name={src['name']} score={src['score']:.4f}")


def main():
    relevant_query = "How are stress and anxiety related to social media use and sleep?"
    irrelevant_query = "What is the capital of France?"

    relevant_result = generate_answer(relevant_query, top_k=3, min_score=0.1)
    irrelevant_result = generate_answer(irrelevant_query, top_k=3, min_score=0.1)

    print_result("Relevant query", relevant_result)
    print_result("Irrelevant query", irrelevant_result)


if __name__ == "__main__":
    main()
