import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Cache resources to load only once
@st.cache_resource
def load_data():
    df = pd.read_csv("semantic_manga_dataset.csv")
    embeddings = np.load("manga_embeddings.npy")
    
    # Build FAISS index
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Load model
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return df, index, model

def recommend_faiss5(title, df, model, index, k=20, genre_filter=True, 
                    type_filter=True, allow_low_popularity=True, sort_by="rerank"):
    row = df[df['english_title'].str.lower() == title.lower()]
    if row.empty:
        return None, f"❌ Title '{title}' not found in dataset."

    row = row.iloc[0]
    semantic_text = row.get('semantic_text', '')
    original_title = row.get('english_title') or row.get('romaji_title') or "Unknown Title"
    original_type = row.get('format', 'MANGA')
    original_genres = set(str(row.get('genres', '')).split(','))

    # Embed query
    query = "Represent this sentence for retrieval: " + semantic_text
    query_vec = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vec, k + 50)

    candidates = []
    for i, dist in zip(indices[0], distances[0]):
        rec = df.iloc[i]
        rec_title = str(rec.get('english_title') or rec.get('romaji_title') or "Unknown Title")

        if not rec_title or rec_title.lower() == title.lower() or rec_title.lower() == 'nan':
            continue

        if type_filter and rec.get('format') != original_type:
            continue

        rec_genres = set(str(rec.get('genres', '')).split(','))
        if genre_filter and not original_genres & rec_genres:
            continue

        popularity = rec.get('popularity', 0)
        if not allow_low_popularity and popularity < 1000:
            continue

        genre_overlap = len(original_genres & rec_genres)
        title_overlap = len(set(original_title.lower().split()) & set(rec_title.lower().split()))
        rerank_score = dist + 0.02 * genre_overlap + 0.01 * title_overlap

        candidates.append({
            'title': rec_title,
            'similarity': dist,
            'rerank_score': rerank_score,
            'popularity': popularity,
            'description': rec.get('description', 'N/A'),
            'genres': rec.get('genres', 'N/A'),
            'format': rec.get('format', 'N/A'),
            'cover_url': rec.get('cover_url', '')
        })

    if sort_by == "similarity":
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
    elif sort_by == "popularity":
        candidates.sort(key=lambda x: x['popularity'], reverse=True)
    else:
        candidates.sort(key=lambda x: (x['rerank_score'], x['popularity']), reverse=True)

    return original_title, candidates[:k]

# Streamlit UI
def main():
    st.set_page_config(page_title="Manga Recommender", layout="wide")
    st.title("📚 Manga Recommendation Engine")
    
    # Load data
    df, index, model = load_data()
    
    with st.sidebar:
        st.header("Search Settings")
        title_query = st.selectbox(
            "Select a manga title",
            options=sorted(df['english_title'].dropna().unique()),
            index=0
        )
        k = st.slider("Number of recommendations", 5, 50, 20)
        genre_filter = st.checkbox("Filter by genre", value=True)
        type_filter = st.checkbox("Filter by type", value=True)
        allow_low_popularity = st.checkbox("Include less popular titles", value=True)
        sort_by = st.radio("Sort by", ["rerank", "similarity", "popularity"], index=0)

    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Finding recommendations..."):
            original_title, recommendations = recommend_faiss5(
                title=title_query,
                df=df,
                model=model,
                index=index,
                k=k,
                genre_filter=genre_filter,
                type_filter=type_filter,
                allow_low_popularity=allow_low_popularity,
                sort_by=sort_by
            )
            
            if recommendations is None:
                st.error(f"Title '{title_query}' not found.")
            else:
                st.success(f"Found {len(recommendations)} recommendations for: **{original_title}**")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['title']} (Score: {rec['rerank_score']:.3f})"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(rec['cover_url'] if rec['cover_url'] else "https://via.placeholder.com/150", width=200)
                        with col2:
                            st.markdown(f"**Genres:** {rec['genres']}  \n**Type:** {rec['format']}  \n**Popularity:** {rec['popularity']}")
                            st.caption(rec['description'][:500] + "...")

if __name__ == "__main__":
    main()
