import pandas as pd

# 1. Load data
df = pd.read_csv("data/news.csv")

# 2. Filtering kolom yang relevan saja untuk ChromoNews
selected_columns = ['id', 'date', 'title', 'content', 'summary']
df_clean = df[selected_columns].copy()

# 3. Handling Missing Values (Menghapus baris yang content atau date-nya kosong)
# Penjelasan logis: Kita tidak bisa mencari (retrieve) berita yang tidak ada isinya, 
# dan kita tidak bisa melakukan "Temporal Event Summarization" jika tanggalnya tidak ada.
df_clean = df_clean.dropna(subset=['content', 'date'])

# 4. Mereset index setelah penghapusan baris agar urutan tetap rapi
df_clean = df_clean.reset_index(drop=True)

print("--- INFO DATASET SETELAH DIBERSIHKAN ---")
print(df_clean.info())

# 5. Opsional untuk eksperimen awal: Sampling Data
# Menjalankan BM25 dan Semantic Search pada 32.000 data sekaligus di tahap uji coba 
# bisa memakan waktu berjam-jam untuk proses embedding. 
# Sebagai AI Engineer, best practice-nya adalah mengambil sampel dulu (misal 1000 data) 
# untuk memastikan pipeline berjalan lancar, baru nanti di-scale up ke seluruh data.

df_sample = df_clean.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"\nJumlah data yang akan dipakai untuk uji coba pipeline: {len(df_sample)} baris")

# Menyimpan data bersih agar tidak perlu cleaning berulang-ulang
df_sample.to_csv('cleaned_news_sample.csv', index=False)
print("File 'cleaned_news_sample.csv' berhasil disimpan!")