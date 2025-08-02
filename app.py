import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="🎬 Anime & Movie Recommender", layout="wide")

# Load external CSS
with open("css/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------- LOAD MODELS & DATA -------------------
@st.cache_resource
def load_deep_model():
    model = load_model("model/hybrid_model.h5", compile=False)
    user_map = pd.read_csv("model/user_map.csv")
    item_map = pd.read_csv("model/item_map.csv")
    return model, user_map, item_map

@st.cache_data
def load_data():
    anime_df = pd.read_csv("data/anime.csv")
    movie_df = pd.read_csv("data/movies.csv")

    for df, is_anime in [(anime_df, True), (movie_df, False)]:
        if "description" not in df.columns:
            for col in ["synopsis", "overview", "desc"]:
                if col in df.columns:
                    df["description"] = df[col]
                    break
            else:
                df["description"] = ""
        if "title" not in df.columns:
            for col in ["name", "anime_name", "movie_title"]:
                if col in df.columns:
                    df["title"] = df[col]
                    break
            else:
                df["title"] = ""
        df["description"] = df["description"].fillna("")
        df["title"] = df["title"].fillna("")
        df["type"] = "Anime" if is_anime else "Movie"

    anime_df = anime_df[["title", "description", "type"]]
    movie_df = movie_df[["title", "description", "type"]]
    return pd.concat([anime_df, movie_df], ignore_index=True)

@st.cache_resource
def build_tfidf_model(descriptions):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(descriptions)
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(matrix)
    return nn, matrix

# -------------------- LOAD EVERYTHING --------------------
combined_df = load_data()
tfidf_model, tfidf_matrix = build_tfidf_model(combined_df["description"])
deep_model, user_map, item_map = load_deep_model()

user_names = {1: "Arjun", 2: "Priya", 3: "Raj", 4: "Kiran", 5: "Sneha"}
user_map["name"] = user_map["user_id"].map(user_names).fillna("User " + user_map["user_id"].astype(str))

# Sidebar Mode Selection
st.sidebar.markdown("<div class='sidebar-title'>Choose mode:</div>", unsafe_allow_html=True)
mode = st.sidebar.radio(
    label="",
    options=[
        "🔍 Search by Title",
        "👥 Recommended by Past User",
        "🎲 Random Suggestions",
        "💖 Get Picks Based on Your Favorite Genre"
    ],
    key="mode_selector"
)

st.title("🎥 Movie & Anime Recommender")

# -------------------- MODE 1: Search by Title --------------------
if mode == "🔍 Search by Title":
    st.subheader("🔍 Search-Based Recommendations")
    st.markdown("<div class='filter-label'>🎯 Filter Content Type:</div>", unsafe_allow_html=True)
    content_type = st.radio("", ("All", "Movie", "Anime"), horizontal=True, key="search_filter")

    filtered_df = combined_df if content_type == "All" else combined_df[combined_df["type"] == content_type]
    filtered_df["label"] = filtered_df["title"] + " (" + filtered_df["type"] + ")"
    selected_label = st.selectbox("Choose a title:", filtered_df["label"].unique())

    st.markdown("<div class='filter-label'>📂 Recommend Similar:</div>", unsafe_allow_html=True)
    rec_filter = st.radio("", ("All", "Movie", "Anime"), horizontal=True, key="rec_filter")

    if st.button("🔍 Recommend Similar"):
        try:
            title, selected_type = selected_label.rsplit(" (", 1)
            selected_type = selected_type.strip(")")
            selected_row = combined_df[(combined_df["title"] == title) & (combined_df["type"] == selected_type)]

            if selected_row.empty:
                st.warning("❌ Could not find the selected title.")
            else:
                selected_index = selected_row.index[0]
                distances, indices = tfidf_model.kneighbors(tfidf_matrix[selected_index], n_neighbors=50)

                recommendations = []
                for i in indices[0]:
                    if i == selected_index:
                        continue
                    row = combined_df.iloc[i]
                    if row["title"] == title:
                        continue
                    if rec_filter == "All" or row["type"] == rec_filter:
                        recommendations.append(row)
                    if len(recommendations) == 5:
                        break

                st.subheader("✨ Top 5 Recommendations:")
                cols = st.columns(5)
                for idx, rec in enumerate(recommendations):
                    with cols[idx]:
                        st.markdown(f"<div class='card search-results'><strong>{rec['title']}</strong><br/>({rec['type']})</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# -------------------- MODE 2: Recommended by Past User --------------------
elif mode == "👥 Recommended by Past User":
    st.subheader("👥 Recommendations Based on Past Users (Hybrid Model)")

    user_dropdown = st.selectbox(
        "Select a past user:",
        user_map.apply(lambda row: f"{row['name']} (User {row['user_id']})", axis=1)
    )

    selected_user_id = int(user_dropdown.split("User ")[-1].replace(")", ""))
    selected_user_name = user_map[user_map["user_id"] == selected_user_id]["name"].values[0]

    if st.button("🎯 Show Recommendations"):
        user_idx = user_map[user_map["user_id"] == selected_user_id]["user_idx"].values[0]
        item_ids = item_map["item_idx"].values
        user_ids = np.full(len(item_ids), user_idx)

        preds = deep_model.predict([user_ids, item_ids], verbose=0).flatten()
        top_indices = preds.argsort()[::-1][:10]
        top_items = item_map.iloc[top_indices]

        st.markdown(f"### 👥 Top 10 Picks Based on **{selected_user_name}**'s Preferences")
        for _, row in top_items.iterrows():
            st.markdown(f"<div class='card'><strong>🎬 {row['title']}</strong><br/>📂 Type: {row['type']}<br/>🎭 Genre: {row.get('genre', 'Unknown')}<br/>📚 Source: {row.get('source', 'N/A')}</div>", unsafe_allow_html=True)

# -------------------- MODE 3: Random Suggestions --------------------
elif mode == "🎲 Random Suggestions":
    st.subheader("🎲 Random Suggestions")
    st.markdown("<div class='filter-label'>🎯 Random from:</div>", unsafe_allow_html=True)
    random_type = st.radio("", ("All", "Movie", "Anime"), horizontal=True)

    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = time.time()

    elapsed = time.time() - st.session_state.last_refresh_time
    remaining = max(0, 5 - int(elapsed))

    if remaining > 0:
        st.info(f"⏳ Please wait {remaining} seconds...")
        time.sleep(1)
        st.rerun()

    if st.button("🔄 Refresh Suggestions"):
        st.session_state.last_refresh_time = time.time()
        st.rerun()

    rand_df = combined_df if random_type == "All" else combined_df[combined_df["type"] == random_type]
    rand_samples = rand_df.sample(n=5, random_state=random.randint(1, 10000))
    cols = st.columns(5)
    for idx, row in enumerate(rand_samples.itertuples()):
        with cols[idx]:
            st.markdown(f"<div class='card'><strong>{row.title}</strong><br/>({row.type})</div>", unsafe_allow_html=True)

# -------------------- MODE 4: Picks Based on Genre --------------------
elif mode == "💖 Get Picks Based on Your Favorite Genre":
    st.subheader("💖 Get Picks Based on Your Favorite Genre")

    def get_genre_list(df):
        genres = df["genre"].dropna().str.lower().str.split(",").explode().str.strip().unique()
        return sorted([g.title() for g in genres if g])

    if "genre" in item_map.columns and "type" in item_map.columns:
        genre_source_df = item_map.copy()
    else:
        genre_source_df = combined_df.copy()
        genre_source_df["genre"] = genre_source_df["description"].str.extract(r"Genre: ([\w\s,]+)", expand=False).fillna("Unknown")
        genre_source_df["type"] = genre_source_df.get("type", "Unknown")

    genre_list = get_genre_list(genre_source_df)
    genre_selected = st.selectbox("🎭 Choose a Genre:", genre_list)
    genre_filter_type = st.radio("📂 Filter by:", ("All", "Movie", "Anime/OVA"), horizontal=True)

    if st.button("🎁 Show Genre Picks"):
        df = genre_source_df.dropna(subset=["genre"]).copy()
        df["genre_list"] = df["genre"].str.lower().str.split(",").apply(lambda x: [g.strip() for g in x])
        matched = df[df["genre_list"].apply(lambda g_list: genre_selected.lower() in g_list)]

        if genre_filter_type == "Movie":
            matched = matched[matched["type"].str.lower() == "movie"]
        elif genre_filter_type == "Anime/OVA":
            matched = matched[matched["type"].str.lower().isin(["anime", "ova"])]

        if matched.empty:
            st.warning("😔 No items found for this genre and filter.")
        else:
            st.success(f"🎉 Showing recommendations for *{genre_selected}* ({genre_filter_type})")
            picks = matched.sample(n=min(5, len(matched)), random_state=random.randint(1, 9999))
            cols = st.columns(len(picks))
            for idx, (_, row) in enumerate(picks.iterrows()):
                with cols[idx]:
                    st.markdown(f"<div class='card genre-card'><strong>🎬 {row['title']}</strong><br/>📂 Genre: {row.get('genre', 'Unknown')}<br/>📚 Source: {row.get('source', 'N/A')}</div>", unsafe_allow_html=True)
