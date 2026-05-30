import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_CACHE_FILE = "corpus_embeddings.npy"


def load_embedding_model():
    """Load pre-trained multilingual sentence transformer model."""
    print(f"Memuat model embedding '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model berhasil dimuat!")
    return model


def encode_corpus(model, corpus, cache_path=EMBEDDINGS_CACHE_FILE):
    """
    Encode seluruh corpus menjadi embedding vectors.
    Hasil di-cache ke file .npy agar tidak perlu encoding ulang.

    Args:
        model: SentenceTransformer model
        corpus: List of document content (GUNAKAN kolom 'content' ASLI,
                bukan 'processed_content', karena model multilingual
                sudah dilatih pada teks natural)
        cache_path: Path untuk menyimpan/memuat cache embeddings
    Returns:
        numpy array of shape (n_docs, embedding_dim)
    """
    # Cek apakah cache sudah ada
    if os.path.exists(cache_path):
        print(f"Memuat embeddings dari cache '{cache_path}'...")
        embeddings = np.load(cache_path)
        print(f"Embeddings dimuat dari cache! Shape: {embeddings.shape}")
        return embeddings

    # Jika belum ada cache, encode corpus
    print(f"Encoding {len(corpus)} dokumen (ini bisa memakan waktu 30-60 detik)...")
    
    # Pastikan semua elemen corpus adalah string
    corpus_clean = [str(text) for text in corpus]
    
    embeddings = model.encode(
        corpus_clean,
        show_progress_bar=True,
        batch_size=64
    )

    # Simpan ke cache
    np.save(cache_path, embeddings)
    print(f"Embeddings disimpan ke '{cache_path}'! Shape: {embeddings.shape}")

    return embeddings


def search_semantic(query, model, corpus_embeddings, top_k=10):
    """
    Melakukan semantic search menggunakan cosine similarity.

    Args:
        query: Query pencarian user (teks natural, TANPA preprocessing)
        model: SentenceTransformer model
        corpus_embeddings: Pre-computed corpus embeddings
        top_k: Jumlah dokumen teratas
    Returns:
        List of (doc_index, similarity_score) tuples, sorted descending
    """
    # Encode query
    query_embedding = model.encode([query])

    # Hitung cosine similarity antara query dan seluruh corpus
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # Ambil top-k indeks berdasarkan similarity tertinggi
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return sebagai list of (doc_index, score)
    results = [(int(idx), float(similarities[idx])) for idx in top_indices]

    return results


# --- Untuk testing mandiri ---
if __name__ == "__main__":
    import pandas as pd

    # Load data
    print("Memuat dataset...")
    df = pd.read_csv('preprocessed_news_sample.csv')

    # Load model & encode corpus
    model = load_embedding_model()
    corpus_embeddings = encode_corpus(model, df['content'].tolist())

    # Test pencarian (query TANPA preprocessing — teks natural)
    query = "dampak ekonomi dari kebangkrutan Silicon Valley Bank"
    print(f"\nQuery: {query}")

    results = search_semantic(query, model, corpus_embeddings, top_k=5)

    print(f"\n--- TOP 5 HASIL SEMANTIC SEARCH ---")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"\n[{rank}] Similarity: {score:.4f}")
        print(f"    Judul: {df['title'].iloc[idx]}")
        print(f"    Tanggal: {df['date'].iloc[idx]}")
        print(f"    Snippet: {str(df['content'].iloc[idx])[:150]}...")
