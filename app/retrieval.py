from typing import List, Dict

from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

from app.config import settings


class HybridRetriever:

    def __init__(self):

        # Vector DB (Dense retrieval)
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

        # Keyword search (Sparse retrieval)
        self.es = Elasticsearch(
            settings.ELASTICSEARCH_URL
        )

        self.collection = settings.QDRANT_COLLECTION
        self.index = settings.ELASTICSEARCH_INDEX

    

    def dense_search(self, embedding, top_k=5) -> List[Dict]:

        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k,
        )

        return [
            {
                "content": r.payload.get("content", ""),
                "score": r.score,
                "source": "dense",
            }
            for r in results
        ]



    def bm25_search(self, query: str, top_k=5) -> List[Dict]:

        response = self.es.search(
            index=self.index,
            query={
                "match": {
                    "content": query
                }
            },
            size=top_k,
        )

        hits = response["hits"]["hits"]

        return [
            {
                "content": h["_source"]["content"],
                "score": h["_score"],
                "source": "bm25",
            }
            for h in hits
        ]


    def reciprocal_rank_fusion(self, dense, bm25, k=60):

        """
        RRF formula:
        score = Σ 1 / (k + rank)
        """

        scores = {}

        # Dense results
        for rank, doc in enumerate(dense):
            key = doc["content"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        # BM25 results
        for rank, doc in enumerate(bm25):
            key = doc["content"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        # Sort by score
        fused = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in fused]



    def retrieve(self, question: str, embedding: List[float]):

        """
        Full Hybrid Retrieval Pipeline
        """

        dense_results = self.dense_search(
            embedding=embedding,
            top_k=5
        )

        bm25_results = self.bm25_search(
            query=question,
            top_k=5
        )

        fused_docs = self.reciprocal_rank_fusion(
            dense_results,
            bm25_results
        )

        # return top 5 final contexts
        return fused_docs[:5]