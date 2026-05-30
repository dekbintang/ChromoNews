from collections import defaultdict


def reciprocal_rank_fusion(bm25_results, semantic_results, k=60, top_k=10):
    """
    Menggabungkan ranking dari BM25 dan Semantic Search
    menggunakan Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        bm25_results: Hasil BM25 [(doc_index, score), ...]
        semantic_results: Hasil Semantic [(doc_index, score), ...]
        k: Smoothing constant (default: 60)
        top_k: Jumlah dokumen final
    Returns:
        List of (doc_index, rrf_score) tuples, sorted descending
    """
    rrf_scores = defaultdict(float)

    # Hitung RRF score dari BM25 results
    for rank, (doc_idx, _) in enumerate(bm25_results):
        rrf_scores[doc_idx] += 1.0 / (k + rank + 1)  # rank dimulai dari 1

    # Hitung RRF score dari Semantic results
    for rank, (doc_idx, _) in enumerate(semantic_results):
        rrf_scores[doc_idx] += 1.0 / (k + rank + 1)

    # Sort berdasarkan RRF score tertinggi
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Return top-k
    return sorted_results[:top_k]


# --- Untuk testing mandiri ---
if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess_text
    from bm25_search import build_bm25_index, search_bm25
    from semantic_search import load_embedding_model, encode_corpus, search_semantic

    # Load data
    print("Memuat dataset...")
    df = pd.read_csv('preprocessed_news_sample.csv')

    # --- BM25 ---
    print("\n[1/3] Membangun BM25 index...")
    tokenized_corpus = [str(doc).split() for doc in df['processed_content']]
    bm25_index = build_bm25_index(tokenized_corpus)

    # --- Semantic ---
    print("[2/3] Memuat model semantic search...")
    model = load_embedding_model()
    corpus_embeddings = encode_corpus(model, df['content'].tolist())

    # --- Hybrid Search ---
    query_raw = "perkembangan kasus Rafael Alun pajak"
    query_processed = preprocess_text(query_raw)
    print(f"\n[3/3] Menjalankan Hybrid Search...")
    print(f"Query asli     : {query_raw}")
    print(f"Query processed: {query_processed}")

    # Dapatkan hasil dari masing-masing metode
    bm25_results = search_bm25(query_processed, bm25_index, top_k=20)
    semantic_results = search_semantic(query_raw, model, corpus_embeddings, top_k=20)

    # Gabungkan dengan RRF
    hybrid_results = reciprocal_rank_fusion(bm25_results, semantic_results, top_k=10)

    print(f"\n--- TOP 10 HASIL HYBRID SEARCH (RRF) ---")
    for rank, (idx, score) in enumerate(hybrid_results, 1):
        print(f"\n[{rank}] RRF Score: {score:.6f}")
        print(f"    Judul  : {df['title'].iloc[idx]}")
        print(f"    Tanggal: {df['date'].iloc[idx]}")
        print(f"    Snippet: {str(df['content'].iloc[idx])[:120]}...")

    # Tampilkan perbandingan
    print(f"\n--- PERBANDINGAN ---")
    print(f"BM25 menemukan    : {len(bm25_results)} dokumen")
    print(f"Semantic menemukan: {len(semantic_results)} dokumen")
    
    bm25_ids = {idx for idx, _ in bm25_results}
    sem_ids = {idx for idx, _ in semantic_results}
    overlap = bm25_ids & sem_ids
    print(f"Overlap (ada di keduanya): {len(overlap)} dokumen")
    print(f"Unik dari BM25    : {len(bm25_ids - sem_ids)} dokumen")
    print(f"Unik dari Semantic : {len(sem_ids - bm25_ids)} dokumen")
