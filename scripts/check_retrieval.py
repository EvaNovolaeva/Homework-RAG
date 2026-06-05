from app.retriever import retrieve


def print_results(title, results):
    print(f"\n=== {title} ===")
    for i, item in enumerate(results, start=1):
        print(f"{i}. doc_id={item['doc_id']} score={item['score']:.4f} name={item['name']}")
        print(f"   text={item['text']}")
        print()


def main():
    relevant_query = "high stress social media sleep anxiety"
    irrelevant_query = "quantum physics black holes"

    relevant_results = retrieve(relevant_query, top_k=3)
    irrelevant_results = retrieve(irrelevant_query, top_k=3)

    print_results("Relevant query", relevant_results)
    print_results("Irrelevant query", irrelevant_results)


if __name__ == "__main__":
    main()
