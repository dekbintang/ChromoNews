import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables dari .env file
load_dotenv()

def configure_gemini(api_key=None):
    """
    Configure Gemini API dengan API key.
    Jika api_key tidak diberikan, akan mencoba mengambil dari environment variable GEMINI_API_KEY.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Gemini API Key tidak ditemukan. Set environment variable GEMINI_API_KEY atau masukkan melalui UI.")
        
    genai.configure(api_key=api_key)

def summarize_articles(articles, query):
    """
    Meringkas kumpulan artikel berita menggunakan Gemini.
    
    Args:
        articles: List of dicts dengan keys: 'title', 'content', 'date'
        query: Query pencarian user (untuk konteks)
    Returns:
        {
            "abstract_summary": "...",      # Ringkasan abstraktif
            "chronological_summary": "..."  # Ringkasan kronologis
        }
    """
    if not articles:
        return {
            "abstract_summary": "Tidak ada artikel relevan untuk diringkas.",
            "chronological_summary": "Tidak ada peristiwa yang dapat disusun secara kronologis."
        }

    # Format artikel menjadi string untuk prompt
    formatted_articles = ""
    for i, article in enumerate(articles, 1):
        formatted_articles += f"\n--- ARTIKEL {i} ---\n"
        formatted_articles += f"Tanggal: {article['date']}\n"
        formatted_articles += f"Judul: {article['title']}\n"
        # Ambil max 1500 karakter per artikel agar tidak melebihi token limit
        content = str(article['content'])[:1500] 
        formatted_articles += f"Isi: {content}...\n"

    system_prompt = f"""Kamu adalah AI News Analyst ahli.
Diberikan kumpulan artikel berita yang terkait dengan topik pencarian pengguna: "{query}".

Tugasmu adalah menghasilkan output dalam format JSON yang valid, yang berisi dua kunci utama:
1. "abstract_summary": Rangkuman abstraktif inti dari seluruh artikel terkait topik "{query}" dalam 1-2 paragraf padat. Fokus pada fakta utama, tokoh penting, dan dampak.
2. "chronological_summary": Susunan perkembangan peristiwa berdasarkan urutan waktu (dari tanggal paling awal ke terbaru) yang ada di dalam artikel. Format nilainya adalah string yang berisi teks dengan bullet points atau list berurutan per tanggal.

Gunakan bahasa Indonesia yang baku dan mudah dipahami.
HANYA KEMBALIKAN JSON VALID TANPA MARKDOWN APAPUN (jangan pakai blok kode ```json).

Berikut adalah artikel-artikelnya:
{formatted_articles}
"""

    try:
        # Menggunakan gemini-flash-latest yang tersedia di API key saat ini
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(system_prompt)
        
        # Parse JSON dari respons
        response_text = response.text.strip()
        
        # Bersihkan dari blok markdown JSON jika Gemini masih ngeyel
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        result = json.loads(response_text)
        
        return {
            "abstract_summary": result.get("abstract_summary", "Gagal mengekstrak ringkasan abstraktif."),
            "chronological_summary": result.get("chronological_summary", "Gagal mengekstrak ringkasan kronologis.")
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Raw Response: {response.text}")
        return {
            "abstract_summary": "Terjadi kesalahan saat memformat respons dari AI (Bukan JSON yang valid).",
            "chronological_summary": response.text # Tampilkan raw response sebagai fallback
        }
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            "abstract_summary": f"Terjadi kesalahan saat memanggil API Gemini: {str(e)}",
            "chronological_summary": "Pastikan API Key valid dan kuota mencukupi."
        }

# --- Untuk testing mandiri ---
if __name__ == "__main__":
    import pandas as pd
    
    # Load env (pastikan ada .env dengan GEMINI_API_KEY)
    try:
        configure_gemini()
        print("Berhasil konfigurasi Gemini API.")
    except Exception as e:
        print(f"Peringatan: {e}")
        print("Memasukkan dummy response...")
        # Lanjut dengan simulasi tanpa exit untuk kemudahan testing jika belum ada key
        
    print("\nMemuat sampel data...")
    df = pd.read_csv('preprocessed_news_sample.csv')
    
    # Ambil 3 artikel teratas terkait "Rafael Alun" sebagai simulasi hasil Hybrid Search
    sample_docs = df[df['title'].str.contains('Rafael', case=False, na=False)].head(3)
    
    if len(sample_docs) > 0:
        articles_to_summarize = []
        for _, row in sample_docs.iterrows():
            articles_to_summarize.append({
                'title': row['title'],
                'date': row['date'],
                'content': row['content']
            })
            
        print(f"Mengirim {len(articles_to_summarize)} artikel ke Gemini...\n")
        
        try:
            summary = summarize_articles(articles_to_summarize, "Rafael Alun")
            print("=== RINGKASAN ABSTRAKTIF ===")
            print(summary["abstract_summary"])
            print("\n=== RINGKASAN KRONOLOGIS ===")
            print(summary["chronological_summary"])
        except Exception as e:
            print(f"Test gagal: {e}")
    else:
        print("Tidak ada data sampel untuk dites.")
