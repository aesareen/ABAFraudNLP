# streamlit_app.py

import ast
import re
from collections import Counter
from urllib.parse import urlparse

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.decomposition import PCA


# ---------------------------------------------------
# SUPABASE CONFIG – YOUR URL + ANON KEY
# ---------------------------------------------------
SUPABASE_URL = "https://ppkqpyqsxqstzkgaxqta.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBwa3FweXFzeHFzdHprZ2F4cXRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI5NjU0MTYsImV4cCI6MjA3ODU0MTQxNn0.hI-L408giaWDcW-d-ntosvMVXtZjfL9QBQtQ8ZDrefk"
)

# Base REST URL
REST_BASE = f"{SUPABASE_URL}/rest/v1"

# Common headers for all REST calls
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Table names (NO 'public.' in front)
ARTICLES_TABLE = "articles"
EXTRACT_TABLE = "article_extract"
EMBEDDINGS_TABLE = "article_embeddings"


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fraud News Explorer",
    layout="wide",
)

st.title("Fraud News Explorer")
st.write(
    "Explore ABA fraud-related articles, see where key words appear, "
    "get simple metrics, and view a few quick visualizations."
)


# -------------------------------------------------
# HELPER: call Supabase REST Data API
# -------------------------------------------------
def fetch_table(table_name: str, limit: int = 20) -> pd.DataFrame:
    """
    Fetch up to `limit` rows from a Supabase table using the REST Data API.
    Returns a pandas DataFrame.
    """
    url = f"{REST_BASE}/{table_name}"
    params = {"select": "*", "limit": limit}

    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data)


def compute_embedding_coords(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take article_embeddings dataframe and add 2D PCA coordinates (x, y).
    Handles both list and string representations of vectors.
    """
    if emb_df.empty or "embedding" not in emb_df.columns:
        emb_df["x"] = np.nan
        emb_df["y"] = np.nan
        return emb_df

    emb_df = emb_df.copy()
    emb_df["x"] = np.nan
    emb_df["y"] = np.nan

    vectors = []
    valid_idx = []
    expected_dim = None

    for i, val in enumerate(emb_df["embedding"]):
        try:
            if isinstance(val, list):
                vec = np.array(val, dtype="float32")
            elif isinstance(val, str):
                vec = np.array(ast.literal_eval(val), dtype="float32")
            else:
                continue

            vec = vec.flatten()
            if expected_dim is None:
                expected_dim = vec.shape[0]
            if vec.shape[0] != expected_dim:
                continue

            vectors.append(vec)
            valid_idx.append(i)
        except Exception:
            continue

    if len(vectors) < 2:
        return emb_df

    X = np.vstack(vectors)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    emb_df.loc[emb_df.index[valid_idx], "x"] = coords[:, 0]
    emb_df.loc[emb_df.index[valid_idx], "y"] = coords[:, 1]

    return emb_df


# -------------------------------------------------
# LOAD & JOIN DATA FROM SUPABASE
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(max_rows: int = 20):
    """
    Load data from the three Supabase tables and join them into a single DF.
    """
    try:
        articles_df = fetch_table(ARTICLES_TABLE, limit=max_rows)
        extract_df = fetch_table(EXTRACT_TABLE, limit=max_rows * 10)
        emb_df = fetch_table(EMBEDDINGS_TABLE, limit=max_rows)
    except Exception as e:
        st.error(
            "I couldn't connect to the database. "
            "Please check your internet connection and access keys.\n\n"
            f"Technical error: {e}"
        )
        return None

    if articles_df.empty and extract_df.empty and emb_df.empty:
        st.error(
            "I couldn't load any rows from the database. "
            "Make sure your tables have data and that your RLS policies "
            "allow reads for this key."
        )
        return None

    # Rename article columns to nicer names
    if "article_name" in articles_df.columns:
        articles_df = articles_df.rename(columns={"article_name": "title"})
    if "raw_content" in articles_df.columns:
        articles_df = articles_df.rename(columns={"raw_content": "content"})
    if "date_published" in articles_df.columns:
        articles_df["published_date"] = pd.to_datetime(
            articles_df["date_published"], errors="coerce"
        )

    # Treat article_extract.text as a fraud topic / tag
    if "text" in extract_df.columns:
        extract_df = extract_df.rename(columns={"text": "fraud_topic"})

    # Compute 2D coords from embeddings and keep only needed columns
    emb_df = compute_embedding_coords(emb_df)
    if "article_id" in emb_df.columns:
        emb_df_small = emb_df[["article_id", "x", "y"]]
    else:
        emb_df_small = pd.DataFrame()

    # Join: articles ← extract (tags) ← embeddings (coords)
    df = articles_df
    if not extract_df.empty and "article_id" in extract_df.columns:
        df = df.merge(
            extract_df[["article_id", "fraud_topic"]],
            on="article_id",
            how="left",
        )
    if not emb_df_small.empty:
        df = df.merge(emb_df_small, on="article_id", how="left")

    return df


# -------------------------------------------------
# TEXT / METRICS HELPERS
# -------------------------------------------------
def filter_by_keyword(df: pd.DataFrame, keyword: str, case_sensitive: bool) -> pd.DataFrame:
    """Filter rows where keyword appears in title, content, or fraud_topic."""
    if not keyword:
        return df.iloc[0:0]

    text_cols = [c for c in ["title", "content", "fraud_topic"] if c in df.columns]
    if not text_cols:
        return df.iloc[0:0]

    mask = False
    for col in text_cols:
        series = df[col].fillna("")
        col_mask = series.str.contains(
            keyword,
            case=case_sensitive,
            regex=False,
        )
        mask = mask | col_mask

    return df[mask].copy()


def get_top_terms(text_series: pd.Series, n: int = 15) -> pd.DataFrame:
    """Simple word-frequency extractor from a Series of text."""
    stopwords = {
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "a",
        "is",
        "for",
        "on",
        "that",
        "with",
        "as",
        "by",
        "an",
        "this",
        "are",
        "be",
        "from",
        "at",
        "it",
        "was",
        "has",
        "have",
        "will",
        "can",
        "may",
        "not",
        "its",
    }
    tokens = []
    for text in text_series.dropna():
        words = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
        tokens.extend(w for w in words if w not in stopwords)

    counts = Counter(tokens).most_common(n)
    if not counts:
        return pd.DataFrame(columns=["term", "count"])
    return pd.DataFrame(counts, columns=["term", "count"])


def make_timeline_chart(df: pd.DataFrame):
    if "published_date" not in df.columns:
        return None
    tmp = df.dropna(subset=["published_date"]).copy()
    if tmp.empty:
        return None

    tmp["month"] = tmp["published_date"].dt.to_period("M").astype(str)

    agg = (
        tmp.groupby("month")["article_id"]
        .count()
        .reset_index()
        .rename(columns={"article_id": "count"})
        .sort_values("month")
    )

    chart = (
        alt.Chart(agg)
        .mark_line(point=True, color="#4F46E5")
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("count:Q", title="Articles mentioning the word"),
            tooltip=["month", "count"],
        )
    )
    return chart


def make_fraud_topic_chart(df: pd.DataFrame):
    if "fraud_topic" not in df.columns:
        return None

    agg = (
        df.groupby("fraud_topic")["article_id"]
        .count()
        .reset_index()
        .rename(columns={"article_id": "count"})
        .sort_values("count", ascending=False)
    )

    chart = (
        alt.Chart(agg)
        .mark_bar(color="#10B981")  # green
        .encode(
            x=alt.X("count:Q", title="Number of articles"),
            y=alt.Y("fraud_topic:N", sort="-x", title="Fraud topic (article_extract.text)"),
            tooltip=["fraud_topic", "count"],
        )
    )
    return chart


def make_top_terms_chart(df: pd.DataFrame):
    if "content" not in df.columns:
        return None
    combined = df["content"].fillna("")
    terms_df = get_top_terms(combined)
    if terms_df.empty:
        return None

    chart = (
        alt.Chart(terms_df)
        .mark_bar(color="#0EA5E9")  # blue
        .encode(
            x=alt.X("count:Q", title="Frequency"),
            y=alt.Y("term:N", sort="-x", title="Top terms in article content"),
            tooltip=["term", "count"],
        )
    )
    return chart


def make_embedding_chart(df: pd.DataFrame):
    if "x" not in df.columns or "y" not in df.columns:
        return None

    scatter_data = df.dropna(subset=["x", "y"]).copy()
    if scatter_data.empty:
        return None

    color_encoding = (
        alt.Color("fraud_topic:N", title="Fraud topic")
        if "fraud_topic" in scatter_data.columns
        else alt.value("#6366F1")
    )

    title_col = "title" if "title" in scatter_data.columns else "article_id"

    chart = (
        alt.Chart(scatter_data)
        .mark_circle(size=70, opacity=0.85)
        .encode(
            x=alt.X("x:Q", title="Embedding dimension 1"),
            y=alt.Y("y:Q", title="Embedding dimension 2"),
            color=color_encoding,
            tooltip=[
                title_col,
                "published_date",
                "fraud_topic" if "fraud_topic" in scatter_data.columns else alt.value(""),
            ],
        )
        .interactive()
    )
    return chart


def make_source_chart(df: pd.DataFrame):
    """Bar chart of top article sources (domains) – currently unused."""
    if "source_url" not in df.columns:
        return None

    tmp = df.copy()
    tmp["source_domain"] = (
        tmp["source_url"]
        .fillna("")
        .apply(lambda x: urlparse(x).netloc.replace("www.", ""))
    )

    agg = (
        tmp.groupby("source_domain")["article_id"]
        .nunique()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )

    if agg.empty:
        return None

    chart = (
        alt.Chart(agg)
        .mark_bar(color="#F97316")  # orange
        .encode(
            x=alt.X("count:Q", title="Number of articles"),
            y=alt.Y("source_domain:N", sort="-x", title="Source"),
            tooltip=["source_domain", "count"],
        )
    )
    return chart


def make_keyword_share_chart(df_all: pd.DataFrame, filtered: pd.DataFrame):
    """
    Show how prominent this keyword is over time:
    share of all fraud articles that mention the word (by month).
    """
    if "published_date" not in df_all.columns or "published_date" not in filtered.columns:
        return None

    # All articles with a date
    all_tmp = df_all.dropna(subset=["published_date"]).copy()
    if all_tmp.empty:
        return None
    all_tmp["month"] = all_tmp["published_date"].dt.to_period("M").astype(str)

    # Filtered articles with a date
    filt_tmp = filtered.dropna(subset=["published_date"]).copy()
    if filt_tmp.empty:
        return None
    filt_tmp["month"] = filt_tmp["published_date"].dt.to_period("M").astype(str)

    # Count total articles per month
    if "article_id" in all_tmp.columns:
        total = (
            all_tmp.groupby("month")["article_id"]
            .nunique()
            .reset_index(name="total_articles")
        )
    else:
        total = (
            all_tmp.groupby("month")
            .size()
            .reset_index(name="total_articles")
        )

    # Count keyword articles per month
    if "article_id" in filt_tmp.columns:
        matched = (
            filt_tmp.groupby("month")["article_id"]
            .nunique()
            .reset_index(name="matched_articles")
        )
    else:
        matched = (
            filt_tmp.groupby("month")
            .size()
            .reset_index(name="matched_articles")
        )

    # Merge + compute share
    merged = total.merge(matched, on="month", how="left").fillna(0)
    merged["share"] = merged["matched_articles"] / merged["total_articles"]

    if merged.empty:
        return None

    chart = (
        alt.Chart(merged)
        .mark_line(point=True, color="#EC4899")  # pink
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y(
                "share:Q",
                title="Share of articles mentioning the word",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                "month",
                alt.Tooltip("matched_articles:Q", title="Articles with keyword"),
                alt.Tooltip("total_articles:Q", title="All articles"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
    )
    return chart


# -------------------------------------------------
# SIDEBAR: SETTINGS
# -------------------------------------------------
st.sidebar.header("Settings")

max_rows = st.sidebar.number_input(
    "Max rows to load from each table (use 20 for assignment):",
    min_value=5,
    max_value=100,
    step=5,
    value=20,
)

case_sensitive = st.sidebar.checkbox("Case-sensitive search?", value=False)

st.sidebar.markdown(
    "**Tables used:**  \n"
    f"- `{ARTICLES_TABLE}`  \n"
    f"- `{EXTRACT_TABLE}`  \n"
    f"- `{EMBEDDINGS_TABLE}`"
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df_all = load_data(max_rows=max_rows)
if df_all is None or df_all.empty:
    st.stop()

# Optional short note instead of showing the raw URLs
st.caption("Data source: ABA fraud news articles stored in a database.")
st.markdown("---")


# -------------------------------------------------
# SEARCH + ANALYSIS
# -------------------------------------------------
st.subheader("Keyword search across fraud articles")

# Preset keywords dropdown
preset_options = [
    "Type a custom word…",
    "fraud",
    "scam",
    "scams",
    "phishing",
    "cybersecurity",
    "regulation",
    "financial services",
]

preset = st.selectbox("Quick-pick a common keyword", preset_options, index=1)

default_examples = "Examples: fraud, scams, regulation, financial services"
custom_keyword = st.text_input(
    "Or type your own word",
    value="" if preset != "Type a custom word…" else "fraud",
    help=default_examples,
)

# Final keyword: if user typed something, use that; otherwise use preset (unless preset = first option)
if custom_keyword.strip():
    keyword = custom_keyword.strip()
elif preset != "Type a custom word…":
    keyword = preset
else:
    keyword = ""

# Threshold just for a warning (charts are always shown if there are results)
min_articles = st.slider(
    "Show a warning if fewer than this many articles mention the word",
    min_value=1,
    max_value=50,
    value=5,
)

if keyword:
    filtered = filter_by_keyword(df_all, keyword, case_sensitive=case_sensitive)

    if filtered.empty:
        st.warning("No articles found containing that word in title, content, or topic.")
    else:
        # Distinct article count (falls back to row-count if article_id missing)
        if "article_id" in filtered.columns:
            n_articles = len(filtered["article_id"].unique())
        else:
            n_articles = len(filtered)

        st.markdown(
            f"Showing results for **`{keyword}`** in **{len(filtered)}** article rows "
            f"from **{n_articles}** distinct articles."
        )

        # --- EXTRA METRICS ---
        if "source_url" in filtered.columns:
            unique_sources = filtered["source_url"].dropna().nunique()
        else:
            unique_sources = 0

        if "content" in filtered.columns:
            avg_words = (
                filtered["content"]
                .fillna("")
                .str.split()
                .str.len()
                .mean()
            )
            avg_words_display = f"{avg_words:.0f}"
        else:
            avg_words_display = "N/A"

        # --- KPIs ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Articles mentioning the word", n_articles)

        with col2:
            if "fraud_topic" in filtered.columns:
                st.metric(
                    "Distinct fraud topics",
                    filtered["fraud_topic"].nunique(),
                )
            else:
                st.metric("Distinct fraud topics", "N/A")

        with col3:
            st.metric("Unique sources", unique_sources)

        with col4:
            if "published_date" in filtered.columns:
                dates = filtered["published_date"].dropna()
                if not dates.empty:
                    date_range = f"{dates.min().date()} → {dates.max().date()}"
                else:
                    date_range = "N/A"
            else:
                date_range = "N/A"
            st.metric("Avg. article length (words)", avg_words_display)

        # Optional warning when the keyword is rare
        if n_articles < min_articles:
            st.warning(
                f"Only {n_articles} articles mention this word — that's below your "
                f"warning threshold of {min_articles}. Charts are still shown, "
                "but interpret trends cautiously."
            )

        st.markdown("---")

        # --- Charts row 1: timeline + fraud topics ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Articles over time")
            timeline_chart = make_timeline_chart(filtered)
            if timeline_chart is not None:
                st.altair_chart(timeline_chart, use_container_width=True)
            else:
                st.caption("No date information available.")

        with c2:
            st.markdown("#### Fraud topics (from `article_extract.text`)")
            topic_chart = make_fraud_topic_chart(filtered)
            if topic_chart is not None:
                st.altair_chart(topic_chart, use_container_width=True)
            else:
                st.caption("No `fraud_topic` data available.")

        # --- Top terms ---
        st.markdown("#### Top terms in article content")
        terms_chart = make_top_terms_chart(filtered)
        if terms_chart is not None:
            st.altair_chart(terms_chart, use_container_width=True)
        else:
            st.caption("Not enough text to compute term frequencies.")

        # --- Keyword share over time ---
        st.markdown("#### Share of fraud articles that mention this keyword")
        share_chart = make_keyword_share_chart(df_all, filtered)
        if share_chart is not None:
            st.altair_chart(share_chart, use_container_width=True)
        else:
            st.caption("Not enough date information to compute keyword share.")

        # --- Embedding map ---
        st.markdown("#### Embedding map of similar articles")
        emb_chart = make_embedding_chart(filtered)
        if emb_chart is not None:
            st.altair_chart(emb_chart, use_container_width=True)
        else:
            st.caption(
                "No embedding coordinates available (check `article_embeddings.embedding`)."
            )

        # --- Articles table (always shown when we have results) ---
        st.markdown("#### Articles table")

        table_df = filtered.copy()
        if "article_id" in table_df.columns:
            table_df = table_df.drop_duplicates(subset="article_id")

        display_cols = [
            c
            for c in [
                "published_date",
                "title",
                "fraud_topic",
                "source_url",
            ]
            if c in table_df.columns
        ]

        st.dataframe(
            table_df[display_cols],
            use_container_width=True,
        )

        # CSV download for filtered results
        csv_data = table_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered articles as CSV",
            data=csv_data,
            file_name=f"articles_{keyword}.csv",
            mime="text/csv",
        )
else:
    st.info("Choose a preset or type a word above to start the analysis.")

# ------------------------------------------------- END OF FILE
