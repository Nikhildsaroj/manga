import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import os

# AniList API GraphQL query
ANILIST_QUERY = """
query ($search: String) {
  Media(search: $search, type: MANGA) {
    id
    title {
      english
      romaji
    }
    coverImage {
      extraLarge
      large
      medium
      color
    }
    description(asHtml: false)
    genres
    format
    popularity
    meanScore
    siteUrl
    externalLinks {
      url
      site
    }
  }
}
"""

@st.cache_resource
def load_data():
    """Load dataset, embeddings, and build FAISS index"""
    df = pd.read_csv("semantic_manga_dataset.csv")
    embeddings = np.load("manga_embeddings.npy")
    
    # Build FAISS index
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    return df, index

@st.cache_resource
def load_model():
    """Load SentenceTransformer model with retry logic"""
    try:
        # Try to load from local cache first
        model = SentenceTransformer("bge-large-en-v1.5", local_files_only=True)
        return model
    except:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def download_model():
            return SentenceTransformer("BAAI/bge-large-en-v1.5")
        return download_model()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_anilist_data(title):
    """Fetch manga data from AniList API with retry logic"""
    variables = {'search': title}
    response = requests.post(
        'https://graphql.anilist.co',
        json={'query': ANILIST_QUERY, 'variables': variables}
    )
    if response.status_code == 200:
        return response.json()['data']['Media']
    return None

def recommend_faiss5(title, df, model, index, k=20, genre_filter=True, 
                    type_filter=True, allow_low_popularity=True, sort_by="rerank"):
    """Generate manga recommendations using FAISS"""
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

        # Skip invalid or duplicate titles
        if not rec_title or rec_title.lower() == title.lower() or rec_title.lower() == 'nan':
            continue

        # Apply filters
        if type_filter and rec.get('format') != original_type:
            continue

        rec_genres = set(str(rec.get('genres', '')).split(','))
        if genre_filter and not original_genres & rec_genres:
            continue

        popularity = rec.get('popularity', 0)
        if not allow_low_popularity and popularity < 1000:
            continue

        # Calculate rerank score
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

    # Sort results
    if sort_by == "similarity":
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
    elif sort_by == "popularity":
        candidates.sort(key=lambda x: x['popularity'], reverse=True)
    else:  # rerank
        candidates.sort(key=lambda x: (x['rerank_score'], x['popularity']), reverse=True)

    return original_title, candidates[:k]

def display_recommendation(rec, anilist_data=None):
    """Display a single recommendation card"""
    with st.expander(f"{rec['title']} (Score: {rec['rerank_score']:.3f})"):
        col1, col2 = st.columns([1, 3])
        with col1:
            if anilist_data and anilist_data.get('coverImage'):
                st.image(anilist_data['coverImage']['medium'], width=200)
            else:
                st.image(rec.get('cover_url', "https://via.placeholder.com/150"), width=200)
        
        with col2:
            st.markdown(f"""
            **Genres:** {rec['genres']}  
            **Type:** {rec['format']}  
            **Popularity:** {rec['popularity']}
            """)
            
            if anilist_data:
                st.markdown(f"**AniList Score:** {anilist_data.get('meanScore', 'N/A')}")
                if anilist_data.get('siteUrl'):
                    st.markdown(f"**[View on AniList]({anilist_data['siteUrl']})**")
            
            # Display description
            desc = anilist_data['description'] if anilist_data else rec['description']
            st.caption(desc[:500] + "..." if desc else "No description available")
            
            # Display external links
            if anilist_data and anilist_data.get('externalLinks'):
                st.markdown("**Official Links:**")
                for link in anilist_data['externalLinks']:
                    st.markdown(f"- [{link['site']}]({link['url']})")

def main():
    st.set_page_config(page_title="Manga Recommender", layout="wide")
    st.title("📚 Manga Recommendation Engine")
    
    # Load data and model
    df, index = load_data()
    model = load_model()
    
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
                
                # Get AniList data for main title
                main_data = get_anilist_data(original_title)
                if main_data:
                    st.image(main_data['coverImage']['large'], width=300)
                    st.markdown(f"**[View on AniList]({main_data['siteUrl']})**")
                
                st.divider()
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    anilist_data = get_anilist_data(rec['title'])
                    display_recommendation(rec, anilist_data)

if __name__ == "__main__":
    main()
