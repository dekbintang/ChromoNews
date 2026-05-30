import numpy as np
from rank_bm25 import BM25Okapi


def build_bm25_index(tokenized_corpus):
    """
    Membangun index BM25 dari corpus yang sudah di-tokenisasi.

    Args:
        tokenized_corpus: List of tokenized documents
                          (setiap dokumen = list of kata setelah preprocessing)
    Returns:
        BM25Okapi object
    """
    return BM25Okapi(tokenized_corpus)


def search_bm25(query, bm25_index, top_k=10):
    """
    Melakukan pencarian BM25.

    Args:
        query: Query pencarian user (sudah di-preprocess, berupa string)
        bm25_index: Index BM25 yang sudah dibangun
        top_k: Jumlah dokumen teratas yang dikembalikan
    Returns:
        List of (doc_index, score) tuples, sorted descending by score
    """
    # Tokenisasi query (split ke list of kata)
    tokenized_query = query.split()

    # Dapatkan skor BM25 untuk semua dokumen
    scores = bm25_index.get_scores(tokenized_query)

    # Ambil top-k indeks berdasarkan skor tertinggi
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Return sebagai list of (doc_index, score)
    results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    return results


# --- Untuk testing mandiri ---
if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess_text

    # Load data yang sudah di-preprocessing
    print("Memuat dataset...")
    df = pd.read_csv('preprocessed_news_sample.csv')

    # Bangun corpus dari kolom 'processed_content' (sudah di-stem & stopword removal)
    print("Membangun BM25 index...")
    tokenized_corpus = [str(doc).split() for doc in df['processed_content']]
    bm25_index = build_bm25_index(tokenized_corpus)

    # Test pencarian
    query_raw = "kasus korupsi KPK"
    query_processed = preprocess_text(query_raw)
    print(f"\nQuery asli     : {query_raw}")
    print(f"Query processed: {query_processed}")

    results = search_bm25(query_processed, bm25_index, top_k=5)

    print(f"\n--- TOP 5 HASIL BM25 ---")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"\n[{rank}] Skor: {score:.4f}")
        print(f"    Judul: {df['title'].iloc[idx]}")
        print(f"    Tanggal: {df['date'].iloc[idx]}")
        print(f"    Snippet: {str(df['content'].iloc[idx])[:150]}...")
