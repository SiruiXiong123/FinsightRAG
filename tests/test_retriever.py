from finsightrag.retriever import MultiFinRetriever


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def search(self, document_id, query, modality, include_below_threshold=False):
        self.calls.append(
            {
                "document_id": document_id,
                "query": query,
                "modality": modality,
                "include_below_threshold": include_below_threshold,
            }
        )
        if not include_below_threshold:
            return []
        return [
            {
                "id": f"{modality}_1",
                "document_id": document_id,
                "modality": modality,
                "rank": 1,
                "score": 0.42,
                "threshold": 0.7,
                "passed_threshold": False,
                "content": f"{modality} raw candidate",
            }
        ]


def test_retriever_returns_raw_candidates_when_threshold_filters_everything():
    store = FakeVectorStore()
    retriever = MultiFinRetriever(
        store,
        {
            "min_text_chunks": 6,
            "max_table_chunks": 4,
            "max_image_chunks": 3,
        },
    )

    result = retriever.retrieve(query="what car design is it", document_id="doc-1")

    assert result["retrieval_mode"] == "raw_fallback"
    assert result["fallback_triggered"] is True
    assert result["trace"] == {
        "num_text": 1,
        "num_table": 1,
        "num_image": 1,
        "total": 3,
        "threshold_fallback": True,
        "num_below_threshold": 3,
    }
    assert [context["modality"] for context in result["contexts"]] == ["text", "table", "image"]
    assert all(context["passed_threshold"] is False for context in result["contexts"])
    assert any(call["include_below_threshold"] for call in store.calls)
