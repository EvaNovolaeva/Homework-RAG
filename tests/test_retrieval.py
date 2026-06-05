from app.retriever import retrieve


def test_retrieve_returns_top_k_results():
    results = retrieve("high stress anxiety sleep", top_k=3)

    assert isinstance(results, list)
    assert len(results) == 3


def test_retrieve_has_required_fields():
    results = retrieve("social media stress", top_k=2)

    assert len(results) == 2

    first = results[0]
    assert "doc_id" in first
    assert "name" in first
    assert "text" in first
    assert "score" in first


def test_relevant_query_has_positive_score():
    results = retrieve("high stress anxiety depression risk", top_k=3)

    assert any(item["score"] > 0 for item in results)


def test_irrelevant_query_can_return_zero_scores():
    results = retrieve("quantum physics black holes", top_k=3)

    assert len(results) == 3
    assert all(item["score"] >= 0 for item in results)
