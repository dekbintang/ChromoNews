import streamlit as st
import pandas as pd
import time

from preprocess import preprocess_for_bm25
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion
from summarizer import configure_gemini, summarize_articles, extract_5w1h

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="ChromoNews | Hybrid Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FLAT MODERN (TANPA CARD / GLASSMORPHISM) ---
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

    /* --- Flat Article Section --- */
    .article-section {
        border-left: 3px solid #7b2ff7;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 0 8px 8px 0;
        transition: background 0.2s ease;
    }
    .article-section:hover {
        background: rgba(255, 255, 255, 0.04);
    }
    .article-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 0.4rem;
        font-family: 'Outfit', sans-serif;
    }
    .article-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }
    .article-date {
        color: #00d2ff;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .article-score {
        color: #94a3b8;
        font-size: 0.78rem;
    }
    .article-snippet {
        color: #cbd5e1;
        font-size: 0.93rem;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }

    /* --- Flat Metric Cards (Dataset Tab) --- */
    .metric-card {
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

    /* --- Flat Ringkasan Topik Section --- */
    .summary-section {
        border-left: 3px solid #00d2ff;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        color: #e2e8f0;
        line-height: 1.7;
        font-size: 0.95rem;
        background: rgba(0, 210, 255, 0.03);
        border-radius: 0 8px 8px 0;
    }

    /* --- 5W1H Inline Grid --- */
    .w5h1-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .w5h1-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.45rem 0.7rem;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.02);
    }
    .w5h1-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 40px;
        height: 22px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.6rem;
        font-family: 'Outfit', sans-serif;
        flex-shrink: 0;
        letter-spacing: 0.5px;
    }
    .badge-what  { background: rgba(0, 210, 255, 0.12); color: #00d2ff; }
    .badge-who   { background: rgba(123, 47, 247, 0.12); color: #a78bfa; }
    .badge-when  { background: rgba(16, 185, 129, 0.12); color: #10b981; }
    .badge-where { background: rgba(245, 158, 11, 0.12); color: #f59e0b; }
    .badge-why   { background: rgba(239, 68, 68, 0.12);  color: #ef4444; }
    .badge-how   { background: rgba(236, 72, 153, 0.12); color: #ec4899; }
    .w5h1-text {
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* --- Streamlit Component Overrides --- */
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
        st.markdown('<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">Bulan (Mar-Apr 2023)</div></div>', unsafe_allow_html=True)
    with m_col3:
        avg_len = int(df['content'].str.len().mean())
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_len}</div><div class="metric-label">Rata-rata Karakter</div></div>', unsafe_allow_html=True)
    
    # Dataframe Display
    display_df = df.copy()
    st.dataframe(
        display_df[['date', 'title', 'content']],
        use_container_width=True,
        hide_index=True,
        height=400
    )

# --- TAB 2: HASIL & RINGKASAN ---
with tab_ringkasan:
    if search_button and query:
        
        # Validasi API Key jika ingin summarization
        if not api_key_input:
            st.warning("⚠️ Gemini API Key belum dimasukkan. Pencarian (Retrieval) akan tetap berjalan, tetapi fitur Summarization AI dan 5W1H dimatikan. Masukkan API Key di sidebar untuk mengaktifkan fitur AI.")
        else:
            try:
                configure_gemini(api_key=api_key_input)
            except Exception as e:
                st.error(f"Gagal mengkonfigurasi Gemini API: {e}")
        
        # --- PROSES SEARCH (RETRIEVAL) ---
        with st.spinner("🔍 Mencari dokumen yang relevan (Hybrid Search)..."):
            start_time = time.time()
            
            # Preprocess query untuk BM25 (case folding + stopword + stemming)
            # Note: Query untuk Semantic Search menggunakan teks ASLI (tanpa preprocessing)
            query_processed = preprocess_for_bm25(query)
            
            # 1. BM25 Search (menggunakan query yang sudah di-preprocess)
            bm25_results = search_bm25(query_processed, bm25_index, top_k=20)
            
            # 2. Semantic Search (menggunakan query ASLI tanpa preprocessing)
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
        
        # --- PROSES AI: RINGKASAN TOPIK + EKSTRAKSI 5W1H ---
        all_5w1h = []
        if api_key_input and len(retrieved_articles) > 0:
            # Ringkasan Topik (Abstraktif)
            with st.spinner("🤖 AI sedang membaca artikel dan menyusun ringkasan..."):
                start_ai_time = time.time()
                articles_for_ai = [{'title': a['title'], 'date': a['date'], 'content': a['content']} for a in retrieved_articles]
                summary = summarize_articles(articles_for_ai, query)
                ai_time = time.time() - start_ai_time
            
            # Menampilkan Ringkasan Topik
            st.markdown("### 📋 Ringkasan Topik")
            st.markdown(f'<div class="summary-section">{summary.get("abstract_summary", "Ringkasan tidak tersedia.")}</div>', unsafe_allow_html=True)
            st.caption(f"AI Generation Time: {ai_time:.2f}s")
            st.markdown("---")
            
            # Ekstraksi 5W1H untuk setiap artikel
            with st.spinner("🔬 AI sedang menganalisis 5W1H dari setiap artikel..."):
                start_5w1h_time = time.time()
                for art in retrieved_articles:
                    article_data = {'title': art['title'], 'date': art['date'], 'content': art['content']}
                    result_5w1h = extract_5w1h(article_data, query)
                    all_5w1h.append(result_5w1h)
                time_5w1h = time.time() - start_5w1h_time
            st.caption(f"5W1H Extraction Time: {time_5w1h:.2f}s")
        
        # --- MENAMPILKAN ARTIKEL DENGAN 5W1H INLINE ---
        st.markdown(f"### 📑 Top {len(retrieved_articles)} Artikel Relevan")
        
        for i, art in enumerate(retrieved_articles):
            # Format snippet: ambil 200 karakter pertama
            snippet = str(art['content'])[:200].replace('\n', ' ') + "..."
            
            # Build 5W1H HTML jika tersedia
            w5h1_html = ""
            if i < len(all_5w1h):
                item = all_5w1h[i]
                w5h1_html = (
'<div class="w5h1-grid">'
'<div class="w5h1-item"><span class="w5h1-badge badge-what">WHAT</span>'
f'<span class="w5h1-text">{item.get("what", "Tidak terdeteksi")}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-who">WHO</span>'
f'<span class="w5h1-text">{item.get("who", "Tidak terdeteksi")}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-when">WHEN</span>'
f'<span class="w5h1-text">{item.get("when", "Tidak terdeteksi")}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-where">WHERE</span>'
f'<span class="w5h1-text">{item.get("where", "Tidak terdeteksi")}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-why">WHY</span>'
f'<span class="w5h1-text">{item.get("why", "Tidak terdeteksi")}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-how">HOW</span>'
f'<span class="w5h1-text">{item.get("how", "Tidak terdeteksi")}</span></div>'
'</div>'
                )
            
            # HTML untuk artikel (flat modern, tanpa card)
            article_html = (
f'<div class="article-section">'
f'<div class="article-title">{art["title"]}</div>'
f'<div class="article-meta">'
f'<span class="article-date">🕒 {art["date"]}</span>'
f'<span class="article-score">RRF: {art["score"]:.4f}</span>'
f'</div>'
f'<div class="article-snippet">{snippet}</div>'
f'{w5h1_html}'
f'</div>'
            )
            st.markdown(article_html, unsafe_allow_html=True)
            
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
