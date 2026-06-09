class MultiFinRetriever:
    def __init__(self, vector_store, cfg: dict):
        self.vector_store = vector_store
        self.cfg = cfg

    def retrieve(self, query: str, document_id: str) -> dict:
        text_hits = self.vector_store.search(
            document_id=document_id,
            query=query,
            modality="text",
        )

        if len(text_hits) >= self.cfg["min_text_chunks"]:
            return self._pack(
                query=query,
                document_id=document_id,
                mode="text_only",
                text_hits=text_hits,
                table_hits=[],
                image_hits=[],
            )

        table_hits = self.vector_store.search(
            document_id=document_id,
            query=query,
            modality="table",
        )[: self.cfg["max_table_chunks"]]

        image_hits = self.vector_store.search(
            document_id=document_id,
            query=query,
            modality="image",
        )[: self.cfg["max_image_chunks"]]

        if not (text_hits or table_hits or image_hits):
            raw_text_hits = self.vector_store.search(
                document_id=document_id,
                query=query,
                modality="text",
                include_below_threshold=True,
            )
            raw_table_hits = self.vector_store.search(
                document_id=document_id,
                query=query,
                modality="table",
                include_below_threshold=True,
            )
            raw_image_hits = self.vector_store.search(
                document_id=document_id,
                query=query,
                modality="image",
                include_below_threshold=True,
            )
            if raw_text_hits or raw_table_hits or raw_image_hits:
                return self._pack(
                    query=query,
                    document_id=document_id,
                    mode="raw_fallback",
                    text_hits=raw_text_hits,
                    table_hits=raw_table_hits,
                    image_hits=raw_image_hits,
                    threshold_fallback=True,
                )

        return self._pack(
            query=query,
            document_id=document_id,
            mode="multimodal",
            text_hits=text_hits,
            table_hits=table_hits,
            image_hits=image_hits,
        )

    def _pack(self, query, document_id, mode, text_hits, table_hits, image_hits, threshold_fallback=False):
        hits = text_hits + table_hits + image_hits
        return {
            "query": query,
            "document_id": document_id,
            "retrieval_mode": mode,
            "fallback_triggered": mode != "text_only",
            "contexts": [self._normalize_hit(hit) for hit in hits],
            "trace": {
                "num_text": len(text_hits),
                "num_table": len(table_hits),
                "num_image": len(image_hits),
                "total": len(hits),
                "threshold_fallback": bool(threshold_fallback),
                "num_below_threshold": sum(1 for hit in hits if hit.get("passed_threshold") is False),
            },
        }

    def _normalize_hit(self, hit: dict) -> dict:
        return {
            "id": hit.get("id"),
            "document_id": hit.get("document_id"),
            "modality": hit.get("modality"),
            "rank": hit.get("rank"),
            "score": hit.get("score"),
            "threshold": hit.get("threshold"),
            "passed_threshold": hit.get("passed_threshold"),
            "page": hit.get("page"),
            "page_span": hit.get("page_span"),
            "bbox": hit.get("bbox"),
            "title": hit.get("title"),
            "content": hit.get("content"),
            "summary": hit.get("summary"),
            "json": hit.get("json"),
            "crop_path": hit.get("crop_path") or hit.get("asset_path"),
            "asset_path": hit.get("asset_path"),
        }
