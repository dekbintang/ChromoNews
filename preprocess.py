import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. Load sample data
print("Memuat dataset sampel...")
df = pd.read_csv('cleaned_news_sample.csv')

# 2. Inisialisasi Sastrawi (Stemmer & Stopwords)
print("Menyiapkan Sastrawi (Ini butuh beberapa detik)...")
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

# 3. Fungsi Preprocessing Lengkap
def preprocess_text(text):
    # a. Case Folding (huruf kecil)
    text = str(text).lower()
    
    # b. Cleaning (Hanya ambil huruf a-z)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # c. Stopword Removal
    text = stopword.remove(text)
    
    # d. Stemming
    text = stemmer.stem(text)
    
    return text

# --- PERHATIAN ENGINEER ---
# Sastrawi melakukan pemrosesan kata per kata. Untuk 1000 baris berita,
# ini bisa memakan waktu 2 hingga 5 menit tergantung prosesor laptopmu.
print("Memulai proses Preprocessing pada 1000 data berita...")
print("Silakan tunggu, jangan dimatikan. Sedang memproses...")

# Kita aplikasikan fungsi preprocessing ke kolom 'content'
# Kita simpan di kolom baru 'processed_content' agar data asli tetap utuh untuk dievaluasi
df['processed_content'] = df['content'].apply(preprocess_text)

# Simpan hasil
df.to_csv('preprocessed_news_sample.csv', index=False)
print("\nSelesai! Data berhasil disimpan sebagai 'preprocessed_news_sample.csv'")

# Tampilkan perbandingan sebelum dan sesudah
print("\n--- PERBANDINGAN HASIL ---")
print("ASLI:")
print(df['content'].iloc[0][:150] + "...")
print("\nPREPROCESSED:")
print(df['processed_content'].iloc[0][:150] + "...")