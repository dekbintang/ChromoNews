import streamlit as st
import pandas as pd
import time

from preprocess import preprocess_text
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion
from summarizer import configure_gemini, summarize_articles

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="ChromoNews | Hybrid Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM UNTUK DESAIN MODERN & GLASSMORPHISM ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;800&display=swap');

    /* Global Typography Override */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Gradient Line Accent for Main Header */
    .gradient-text {
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0, 210, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Article Card Specifics */
    .article-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
    }
    .article-date {
        color: #00d2ff;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
        display: inline-block;
        padding: 2px 8px;
        background: rgba(0, 210, 255, 0.1);
        border-radius: 12px;
    }
    .article-snippet {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .relevance-score {
        float: right;
        font-size: 0.8rem;
        color: #94a3b8;
        background: rgba(255, 255, 255, 0.05);
        padding: 3px 8px;
        border-radius: 8px;
    }

    /* Metric Cards in Dataset Tab */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-bottom: 3px solid #7b2ff7;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Summary Section Styling */
    .summary-box {
        background: linear-gradient(145deg, rgba(26,29,35,0.8) 0%, rgba(14,17,23,0.9) 100%);
        border-left: 4px solid #00d2ff;
        padding: 2rem;
        border-radius: 0 16px 16px 0;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Timeline Styling */
    .timeline {
        position: relative;
        padding-left: 30px;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .timeline::before {
        content: '';
        position: absolute;
        top: 0;
        bottom: 0;
        left: 11px;
        width: 2px;
        background: linear-gradient(to bottom, #00d2ff, #7b2ff7);
        border-radius: 2px;
    }
    .timeline-item {
        position: relative;
        margin-bottom: 1.5rem;
    }
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -30px;
        top: 6px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #0e1117;
        border: 2px solid #00d2ff;
        z-index: 1;
        box-shadow: 0 0 10px rgba(0,210,255,0.5);
    }

    /* Customizing Streamlit components */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(123, 47, 247, 0.4);
    }
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff;
        box-shadow: 0 0 0 1px #00d2ff;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI CACHING UNTUK LOAD DATA & MODEL (DIOPTIMASI) ---
@st.cache_resource(show_spinner="📄 Memuat Dataset...")
def load_data():
    try:
        df = pd.read_csv('preprocessed_news_sample.csv')
        return df
    except FileNotFoundError:
        st.error("File 'preprocessed_news_sample.csv' tidak ditemukan. Pastikan modul 1 & 2 sudah dijalankan.")
        return None

@st.cache_resource(show_spinner="🔍 Membangun BM25 Index...")
def load_bm25(_df):
    tokenized_corpus = [str(doc).split() for doc in _df['processed_content']]
    bm25_index = build_bm25_index(tokenized_corpus)
    return bm25_index

@st.cache_resource(show_spinner="🧠 Memuat Model Semantic Search...")
def load_semantic(_df):
    import gc
    model = load_embedding_model()
    corpus_embeddings = encode_corpus(model, _df['content'].tolist())
    gc.collect()  # Bersihkan memori yang tidak terpakai
    return model, corpus_embeddings

# --- INISIALISASI (BERTAHAP) ---
df = load_data()
if df is not None:
    bm25_index = load_bm25(df)
    semantic_model, corpus_embeddings = load_semantic(df)
else:
    bm25_index, semantic_model, corpus_embeddings = None, None, None

# --- SIDEBAR: KONFIGURASI & INFO ---
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    api_key_input = st.text_input("Gemini API Key", type="password", placeholder="Masukkan API Key Anda...", help="Diperlukan untuk fitur Summarization")
    
    st.markdown("---")
    st.markdown("### 📊 Parameter Search")
    top_k = st.slider("Jumlah Artikel (Top-K)", min_value=3, max_value=20, value=5, help="Jumlah artikel teratas yang akan di-retrieve dan dikirim ke AI Summarizer")
    rrf_k = st.number_input("RRF K Constant", min_value=1, max_value=100, value=60, help="Smoothing constant untuk Reciprocal Rank Fusion")
    
    st.markdown("---")
    st.markdown("### 💡 Contoh Query")
    st.markdown("""
    - `kasus korupsi KPK 2023`
    - `perkembangan kasus Rafael Alun pajak`
    - `dampak ekonomi Silicon Valley Bank`
    - `mudik lebaran 2023`
    """)

# --- MAIN APP HEADER ---
st.markdown('<h1 class="gradient-text">🧬 ChromoNews</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">News Retrieval using Hybrid BM25 & Semantic Search with AI Agent for Temporal Event Summarization</div>', unsafe_allow_html=True)

if df is None:
    st.stop() # Stop execution jika data gagal dimuat

# --- SEARCH BAR ---
search_col1, search_col2 = st.columns([5, 1])
with search_col1:
    query = st.text_input("🔍 Cari berita...", placeholder="Ketik topik berita, tokoh, atau peristiwa...", label_visibility="collapsed")
with search_col2:
    search_button = st.button("Search", use_container_width=True)

# --- TABS ---
tab_dataset, tab_ringkasan = st.tabs(["📂 Dataset", "📝 Hasil & Ringkasan"])

# --- TAB 1: DATASET ---
with tab_dataset:
    st.markdown("### Informasi Dataset")
    
    # Metric Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Berita</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">Bulan (Mar-Apr 2023)</div></div>', unsafe_allow_html=True)
    with m_col3:
        avg_len = int(df['content'].str.len().mean())
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_len}</div><div class="metric-label">Rata-rata Karakter</div></div>', unsafe_allow_html=True)
    
    # Dataframe Display
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Format kolom tanggal untuk tampilan
    display_df = df.copy()
    
    st.dataframe(
        display_df[['date', 'title', 'content']],
        use_container_width=True,
        hide_index=True,
        height=400
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: RINGKASAN & HASIL SEARCH ---
with tab_ringkasan:
    if search_button and query:
        
        # Validasi API Key jika ingin summarization
        if not api_key_input:
            st.warning("⚠️ Gemini API Key belum dimasukkan. Pencarian (Retrieval) akan tetap berjalan, tetapi fitur Summarization AI dimatikan. Masukkan API Key di sidebar untuk mengaktifkan AI Summarizer.")
        else:
            try:
                configure_gemini(api_key=api_key_input)
            except Exception as e:
                st.error(f"Gagal mengkonfigurasi Gemini API: {e}")
        
        # --- PROSES SEARCH (RETRIEVAL) ---
        with st.spinner("🔍 Mencari dokumen yang relevan (Hybrid Search)..."):
            start_time = time.time()
            
            # Preprocess query untuk BM25
            query_processed = preprocess_text(query)
            
            # 1. BM25 Search
            bm25_results = search_bm25(query_processed, bm25_index, top_k=20)
            
            # 2. Semantic Search
            semantic_results = search_semantic(query, semantic_model, corpus_embeddings, top_k=20)
            
            # 3. Hybrid Search (RRF)
            hybrid_results = reciprocal_rank_fusion(bm25_results, semantic_results, k=rrf_k, top_k=top_k)
            
            retrieval_time = time.time() - start_time
            
        st.success(f"Ditemukan {len(hybrid_results)} artikel yang relevan dalam {retrieval_time:.2f} detik.")
        
        # --- MENYIAPKAN DATA UNTUK SUMMARIZER & UI ---
        retrieved_articles = []
        for doc_idx, rrf_score in hybrid_results:
            row = df.iloc[doc_idx]
            retrieved_articles.append({
                'title': row['title'],
                'date': row['date'],
                'content': row['content'],
                'score': rrf_score
            })
            
        # --- PROSES SUMMARIZATION (AI AGENT) ---
        if api_key_input and len(retrieved_articles) > 0:
            with st.spinner("🤖 AI sedang membaca artikel dan menyusun ringkasan..."):
                start_ai_time = time.time()
                # Hanya kirim artikel ke fungsi summarizer (tanpa score)
                articles_for_ai = [{'title': a['title'], 'date': a['date'], 'content': a['content']} for a in retrieved_articles]
                
                summary = summarize_articles(articles_for_ai, query)
                ai_time = time.time() - start_ai_time
                
            # Menampilkan Ringkasan Abstraktif
            st.markdown("### 📋 Ringkasan Topik")
            st.markdown(f'<div class="summary-box">{summary.get("abstract_summary", "Ringkasan tidak tersedia.")}</div>', unsafe_allow_html=True)
            
            # Menampilkan Ringkasan Kronologis
            st.markdown("### ⏱️ Timeline Peristiwa")
            chrono_content = summary.get("chronological_summary", "")
            
            # Simple styling untuk kronologis agar masuk ke dalam glass card
            st.markdown(f"""
            <div class="glass-card">
                <div style="color: #e2e8f0; line-height: 1.6;">
                    {chrono_content}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"AI Generation Time: {ai_time:.2f}s")
            st.markdown("---")
            
        # --- MENAMPILKAN DAFTAR ARTIKEL HASIL RETRIEVAL ---
        st.markdown(f"### 📑 Top {len(retrieved_articles)} Artikel Relevan")
        
        for art in retrieved_articles:
            # Format snippet: ambil 200 karakter pertama
            snippet = str(art['content'])[:200].replace('\n', ' ') + "..."
            
            # HTML custom untuk card artikel
            card_html = f"""
            <div class="glass-card">
                <span class="relevance-score">RRF: {art['score']:.4f}</span>
                <div class="article-date">🕒 {art['date']}</div>
                <div class="article-title">{art['title']}</div>
                <div class="article-snippet">{snippet}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Opsi baca selengkapnya via expander Streamlit standar
            with st.expander("📖 Baca Artikel Penuh"):
                st.write(art['content'])

    elif search_button and not query:
        st.warning("Silakan masukkan kata kunci pencarian terlebih dahulu.")
    else:
        # Tampilan kosong saat belum search
        st.info("👈 Masukkan kata kunci pencarian dan klik 'Search' untuk melihat hasil.")
        
        # Placeholder Illustration
        st.markdown("""
        <div style="text-align: center; opacity: 0.5; padding: 4rem;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🔍</div>
            <h3 style="font-family: 'Outfit', sans-serif;">Menunggu Kueri Pencarian</h3>
            <p>Sistem siap mencari dan meringkas ratusan berita dalam hitungan detik.</p>
        </div>
        """, unsafe_allow_html=True)
