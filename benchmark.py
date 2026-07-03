import time
import statistics

from app.pipeline import HybridRAGPipeline
from app.cache import Cache


QUESTION = "What is the refund policy?"
RUNS = 20


def average(values):
    return statistics.mean(values)


def ms(seconds):
    return seconds * 1000


def main():

    pipeline = HybridRAGPipeline()
    cache = Cache()

    embedding_times = []
    dense_times = []
    bm25_times = []
    hybrid_times = []
    reranker_times = []
    generation_times = []
    total_times = []

    print("=" * 70)
    print("HYBRID RAG PERFORMANCE BENCHMARK")
    print("=" * 70)

    for i in range(RUNS):

        print(f"Run {i+1}/{RUNS}")

        total_start = time.perf_counter()


        start = time.perf_counter()

        embedding = pipeline.embed_query(QUESTION)

        embedding_times.append(
            time.perf_counter() - start
        )

  

        start = time.perf_counter()

        dense_docs = pipeline.retriever.dense_search(
            embedding,
            top_k=5
        )

        dense_times.append(
            time.perf_counter() - start
        )



        start = time.perf_counter()

        sparse_docs = pipeline.retriever.bm25_search(
            QUESTION,
            top_k=5
        )

        bm25_times.append(
            time.perf_counter() - start
        )


        start = time.perf_counter()

        docs = pipeline.retriever.retrieve(
            QUESTION,
            embedding
        )

        hybrid_times.append(
            time.perf_counter() - start
        )

 

        start = time.perf_counter()

        docs = pipeline.reranker.rerank(
            QUESTION,
            docs
        )

        docs = docs[:5]

        reranker_times.append(
            time.perf_counter() - start
        )


        start = time.perf_counter()

        answer = pipeline.generator.generate(
            QUESTION,
            docs
        )

        generation_times.append(
            time.perf_counter() - start
        )

        total_times.append(
            time.perf_counter() - total_start
        )


    cache.set(QUESTION, answer)

    N = 1000

    start = time.perf_counter()

    for _ in range(N):
        cache.get(QUESTION)

    cache_latency = (time.perf_counter() - start) / N



    print("\n")
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"Embedding Time            : {ms(average(embedding_times)):.2f} ms")
    print(f"Dense Retrieval           : {ms(average(dense_times)):.2f} ms")
    print(f"BM25 Retrieval            : {ms(average(bm25_times)):.2f} ms")
    print(f"Hybrid Retrieval          : {ms(average(hybrid_times)):.2f} ms")
    print(f"BGE Reranker              : {ms(average(reranker_times)):.2f} ms")
    print(f"LLM Generation            : {average(generation_times):.2f} sec")
    print(f"End-to-End Pipeline       : {average(total_times):.2f} sec")
    print(f"Redis Cache Lookup        : {ms(cache_latency):.3f} ms")
    print(f"Throughput                : {1 / average(total_times):.2f} req/sec")

    print("=" * 70)

    print("\nSample Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()