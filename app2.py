

import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from PIL import Image
import io

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
    df = pd.read_csv("semantic_manga_dataset.csv")
    embeddings = np.load("manga_embeddings.npy")
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return df, index, model

def get_anilist_data(title):
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
    # ... (keep your existing recommendation logic) ...
    return original_title, candidates[:k]

def main():
    st.set_page_config(page_title="Manga Recommender", layout="wide")
    st.title("📚 Manga Recommendation Engine")
    
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
                
                # Get AniList data for the main title
                main_data = get_anilist_data(original_title)
                if main_data:
                    st.image(main_data['coverImage']['large'], width=300)
                    st.markdown(f"**[View on AniList]({main_data['siteUrl']})**")
                
                st.divider()
                
                for i, rec in enumerate(recommendations, 1):
                    # Get AniList data for each recommendation
                    anilist_data = get_anilist_data(rec['title'])
                    
                    with st.expander(f"{i}. {rec['title']} (Score: {rec['rerank_score']:.3f})"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if anilist_data:
                                st.image(anilist_data['coverImage']['medium'], width=200)
                            else:
                                st.image("https://via.placeholder.com/150", width=200)
                        with col2:
                            st.markdown(f"""
                            **Genres:** {rec['genres']}  
                            **Type:** {rec['format']}  
                            **Popularity:** {rec['popularity']}
                            """)
                            
                            if anilist_data:
                                st.markdown(f"**AniList Score:** {anilist_data.get('meanScore', 'N/A')}")
                                st.markdown(f"**[View on AniList]({anilist_data['siteUrl']})**")
                            
                            # Display description
                            desc = anilist_data['description'] if anilist_data else rec['description']
                            st.caption(desc[:500] + "..." if desc else "No description available")
                            
                            # Display external links
                            if anilist_data and anilist_data.get('externalLinks'):
                                st.markdown("**Official Links:**")
                                for link in anilist_data['externalLinks']:
                                    st.markdown(f"- [{link['site']}]({link['url']})")

if __name__ == "__main__":
    main()
