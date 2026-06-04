import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Sastrawi (Stemmer & Stopwords) — hanya untuk pipeline BM25
print("Menyiapkan Sastrawi (Ini butuh beberapa detik)...")
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()


# =============================================================================
# PIPELINE 1: Preprocessing untuk BM25 (Lexical Search)
# - Case folding, cleaning, stopword removal, stemming DILAKUKAN
# - Karena BM25 menggunakan pendekatan statistik (TF-IDF), 
#   preprocessing ini diperlukan untuk normalisasi term matching
# =============================================================================
def preprocess_for_bm25(text):
    """
    Preprocessing untuk pipeline BM25 (lexical search).
    
    Tahapan:
    - Case folding (huruf kecil)
    - Cleaning (hapus karakter selain huruf, angka, dan spasi)
    - Stopword removal (Sastrawi)
    - Stemming (Sastrawi)
    
    Note: Angka TETAP dipertahankan karena penting untuk konteks berita
    (tahun, jumlah korban, statistik, pasal hukum, dll.)
    """
    text = str(text).lower()                    # Case folding
    text = re.sub(r'[^a-z0-9\s]', '', text)     # Cleaning (pertahankan angka)
    text = stopword.remove(text)                 # Stopword removal
    text = stemmer.stem(text)                    # Stemming
    return text


# =============================================================================
# PIPELINE 2: Preprocessing untuk Semantic Search (Sentence Transformer)
# - Teks HARUS TETAP UTUH / ASLI
# - Case folding, stopword removal, stemming TIDAK DILAKUKAN
# - Alasan:
#   1. Sentence Transformer (BERT-based) sudah dilatih pada teks natural
#   2. Huruf besar/kecil penting untuk NER (nama orang, tempat, organisasi)
#   3. Stopword memberi konteks semantik ("makan nasi" != "nasi")
#   4. Stemming merusak tokenisasi subword BERT
#   5. Angka penting untuk informasi 5W1H (When, How)
# =============================================================================
def preprocess_for_semantic(text):
    """
    Preprocessing minimal untuk pipeline Semantic Search.
    
    Hanya melakukan:
    - Konversi ke string
    - Normalisasi whitespace (hapus spasi/newline berlebih)
    - Strip leading/trailing whitespace
    
    Teks asli TIDAK BOLEH diubah strukturnya karena Sentence Transformer
    memerlukan kalimat utuh untuk tokenisasi yang benar.
    """
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)             # Normalize whitespace saja
    return text


# --- BACKWARD COMPATIBILITY ---
# Alias untuk fungsi lama agar tidak break import yang sudah ada
preprocess_text = preprocess_for_bm25


# --- Script untuk re-generate preprocessed dataset ---
if __name__ == "__main__":
    # 1. Load sample data
    print("Memuat dataset sampel...")
    df = pd.read_csv('cleaned_news_sample.csv')

    # 2. Preprocessing untuk BM25 (kolom processed_content)
    print(f"Memulai proses Preprocessing BM25 pada {len(df)} data berita...")
    print("Silakan tunggu, jangan dimatikan. Sedang memproses...")
    df['processed_content'] = df['content'].apply(preprocess_for_bm25)

    # 3. Kolom 'content' ASLI tetap dipertahankan untuk Semantic Search
    # (Semantic search langsung menggunakan kolom 'content' tanpa modifikasi)

    # 4. Simpan hasil
    df.to_csv('preprocessed_news_sample.csv', index=False)
    print(f"\nSelesai! Data berhasil disimpan sebagai 'preprocessed_news_sample.csv'")

    # Tampilkan perbandingan
    print("\n--- PERBANDINGAN HASIL ---")
    print("ASLI (untuk Semantic Search):")
    print(df['content'].iloc[0][:150] + "...")
    print("\nPREPROCESSED BM25:")
    print(df['processed_content'].iloc[0][:150] + "...")