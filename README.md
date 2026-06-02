<div align="center">
  <img src="https://img.icons8.com/color/96/000000/dna-helix.png" alt="ChromoNews Logo"/>
  <h1>🧬 ChromoNews</h1>
  <p><b>Hybrid Search (BM25 + Semantic Search) & AI Temporal Summarization</b></p>

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
  [![Gemini API](https://img.shields.io/badge/Google%20Gemini-AI-orange.svg)](https://deepmind.google/technologies/gemini/)
</div>

<br/>

## 👥 Anggota Kelompok

Proyek ini dikembangkan oleh:

| NIM | Nama |
| :--- | :--- |
| **2405551034** | I Komang Cahya Kertha Yasa |
| **2405551049** | I Kadek Bintang Adi Bimantara |
| **2405551168** | Trio Suro Wibowo |
| **2405551019** | Richard Christian Mozart Diazoni |

---

## 🌟 Tentang Proyek

**ChromoNews** adalah sistem pencarian berita mutakhir yang dirancang untuk memberikan hasil pencarian yang sangat relevan dan mudah dipahami. Sistem ini menggabungkan kekuatan pencarian teks tradisional dan pemahaman makna konteks, yang kemudian disempurnakan dengan kecerdasan buatan (AI) untuk memberikan ringkasan peristiwa secara otomatis.

### 🚀 Fitur Utama

- 🔍 **Hybrid Search Engine**: Menggunakan *Reciprocal Rank Fusion (RRF)* untuk menggabungkan skor dari:
  - **BM25 (Lexical Search)**: Sangat akurat untuk pencocokan kata kunci eksak.
  - **Semantic Search**: Memahami konteks dan makna pencarian menggunakan model *embedding*.
- 🤖 **AI-Powered Summarization (Gemini)**: 
  - **Abstractive Summary**: Merangkum inti dari berbagai artikel berita terkait topik yang dicari menjadi ringkasan padat.
  - **Chronological Timeline**: Menyusun secara otomatis urutan peristiwa berdasarkan waktu terbit berita.
- 🎨 **Modern UI (Glassmorphism)**: Antarmuka yang elegan dan futuristik dibangun di atas platform Streamlit.

## 💻 Instalasi dan Penggunaan

### Prasyarat
Pastikan Anda telah menginstal **Python 3.9+**.

### Langkah-langkah
1. **Clone/Download repository ini**.
2. **Buat Virtual Environment** (Sangat disarankan):
   ```bash
   python -m venv env
   # Windows
   .\env\Scripts\activate
   # Mac/Linux
   source env/bin/activate
   ```
3. **Instal dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Siapkan Environment Variables**:
   - Buka file `.env` di direktori proyek (atau buat jika belum ada).
   - Tambahkan API Key Gemini Anda:
     ```env
     GEMINI_API_KEY=masukkan_api_key_anda_disini
     ```
5. **Jalankan Aplikasi**:
   ```bash
   streamlit run app.py
   ```
6. Buka browser dan akses alamat lokal yang diberikan (biasanya `http://localhost:8501`).