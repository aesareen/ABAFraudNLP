# streamlit_app.py

import os
import sys

# Add project root to path to allow importing from 'agents'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import ast
import re
import asyncio
from collections import Counter
from urllib.parse import urlparse
from nltk.corpus import stopwords
from agents.summarization_agent import initialize_summarization_agent
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from supabase import create_client, Client

# ============================================================
# Environment / Supabase config
# ============================================================

load_dotenv('config/.env', override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL or SUPABASE_KEY is missing. "
        "Add them to your .env file and restart the app."
    )

# Create Supabase client
CLIENT: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Table names – you can override via .env if needed
ARTICLES_TABLE = os.getenv("ARTICLES_TABLE", "articles")
EXTRACT_TABLE = os.getenv("EXTRACT_TABLE", "article_extract")
EMBEDDINGS_TABLE = os.getenv("EMBEDDINGS_TABLE", "article_embeddings")

# Streamlit session state
if "max_rows" not in st.session_state:
    st.session_state.max_rows = 20
if "case_sensitive" not in st.session_state:
    st.session_state.case_sensitive = True
if "keyword" not in st.session_state:
    st.session_state.keyword = ""
if "enable_llm_analysis" not in st.session_state:
    st.session_state.enable_llm_analysis = None
if "llm_summary" not in st.session_state:
    st.session_state.llm_summary = None


# ============================================================
# Small helpers
# ============================================================

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


# ------------------------------------------------------------
# Supabase helpers
# ------------------------------------------------------------
def fetch_table(
    table_name: str,
    limit: int | None = None,
    select: str = "*",
) -> pd.DataFrame:
    """
    Fetch rows from a Supabase table using the Supabase client and return a DataFrame.
    """
    query = CLIENT.table(table_name).select(select)
    
    if limit is not None:
        query = query.limit(limit)
    
    response = query.execute()
    return pd.DataFrame(response.data)

@st.cache_data
def fetch_number_of_rows(table_name: str) -> int:
    table = fetch_table(table_name)
    if table.empty:
        return 0
    return len(table)


def compute_embedding_coords(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'embedding' column and compute 2D PCA coordinates (x, y).
    Handles both JSON arrays and stringified Python lists.
    """
    if emb_df.empty or "embedding" not in emb_df.columns:
        emb_df["x"] = np.nan
        emb_df["y"] = np.nan
        return emb_df

    emb_df = emb_df.copy()
    emb_df["x"] = np.nan
    emb_df["y"] = np.nan

    vectors: list[np.ndarray] = []
    valid_idx: list[int] = []
    expected_dim: int | None = None

    for i, val in enumerate(emb_df["embedding"]):
        try:
            if isinstance(val, list):
                vec = np.array(val, dtype="float32")
            elif isinstance(val, str):
                # Supabase often stores JSON arrays as strings
                vec = np.array(ast.literal_eval(val), dtype="float32")
            else:
                continue

            vec = vec.flatten()
            if expected_dim is None:
                expected_dim = vec.shape[0]
            elif vec.shape[0] != expected_dim:
                # Skip weird / inconsistent dimensions
                continue

            vectors.append(vec)
            valid_idx.append(i)
        except Exception:
            continue

    if len(vectors) < 2:
        # Not enough data for PCA
        return emb_df

    X = np.vstack(vectors)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    emb_df.loc[emb_df.index[valid_idx], "x"] = coords[:, 0]
    emb_df.loc[emb_df.index[valid_idx], "y"] = coords[:, 1]
    return emb_df


# ------------------------------------------------------------
# Load & join data
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(max_rows: int = 20) -> pd.DataFrame | None:
    """
    Load data from the three Supabase tables and return a unified DataFrame.
    """
    try:
        articles_df = fetch_table(ARTICLES_TABLE, limit=max_rows)
        extract_df = fetch_table(EXTRACT_TABLE, limit=max_rows * 10)
        emb_df = fetch_table(EMBEDDINGS_TABLE, limit=max_rows)
    except Exception as e:
        st.error(
            "I could not connect to Supabase. "
            "Please check SUPABASE_URL / SUPABASE_KEY and table names.\n\n"
            f"Technical error: {e}"
        )
        return None

    if articles_df.empty and extract_df.empty and emb_df.empty:
        st.error(
            "No data was loaded from Supabase. "
            "Check that the tables have rows and RLS allows read access."
        )
        return None

    # --- Articles: rename / clean columns ---
    if "article_name" in articles_df.columns:
        articles_df = articles_df.rename(columns={"article_name": "title"})
    if "raw_content" in articles_df.columns:
        articles_df = articles_df.rename(columns={"raw_content": "content"})
    if "date_published" in articles_df.columns:
        articles_df["published_date"] = pd.to_datetime(
            articles_df["date_published"], errors="coerce"
        )

    # Ensure article_id exists
    if "article_id" not in articles_df.columns and "id" in articles_df.columns:
        articles_df = articles_df.rename(columns={"id": "article_id"})

    # Add a simple "source" column from source_url if not present
    if "source" not in articles_df.columns and "source_url" in articles_df.columns:
        articles_df["source"] = articles_df["source_url"].apply(extract_domain)

    # --- Extract table: topics ---
    if not extract_df.empty:
        if "text" in extract_df.columns:
            extract_df = extract_df.rename(columns={"text": "fraud_topic"})

    # --- Embeddings & PCA ---
    emb_df = compute_embedding_coords(emb_df)
    if not emb_df.empty:
        emb_small = emb_df[["article_id", "x", "y"]] if "article_id" in emb_df.columns else pd.DataFrame()
    else:
        emb_small = pd.DataFrame()

    # --- Join all tables on article_id ---
    df = articles_df

    if not extract_df.empty and "article_id" in extract_df.columns:
        df = df.merge(
            extract_df[["article_id", "fraud_topic"]],
            on="article_id",
            how="left",
        )

    if not emb_small.empty and "article_id" in emb_small.columns:
        df = df.merge(emb_small, on="article_id", how="left")

    return df


# ============================================================
# Text / NLP helpers
# ============================================================

STOPWORDS = stopwords.words("english")


def filter_by_keyword(df: pd.DataFrame, keyword: str, case_sensitive: bool) -> pd.DataFrame:
    """
    Filter rows where keyword appears in title, content, or fraud_topic.
    """
    if df.empty or not keyword:
        return df

    cols = [c for c in ["title", "content", "fraud_topic"] if c in df.columns]
    if not cols:
        return df

    flags = False if case_sensitive else True  # used for .str.contains case param

    mask = False
    for col in cols:
        series = df[col].astype("string")
        mask = mask | series.str.contains(keyword, case=case_sensitive, regex=False, na=False)

    return df[mask]


def get_top_terms(text_series: pd.Series, n: int = 15) -> pd.DataFrame:
    """
    Simple word-frequency helper for English-like text.
    """
    tokens: list[str] = []
    for text in text_series.dropna():
        words = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
        tokens.extend(w for w in words if w not in STOPWORDS)

    counts = Counter(tokens).most_common(n)
    if not counts:
        return pd.DataFrame(columns=["term", "count"])
    return pd.DataFrame(counts, columns=["term", "count"])

@st.cache_resource
async def generate_llm_summary(keyword: str) -> str:
    agent = initialize_summarization_agent()
    summary = await agent.generate_article_keyword_summary(keyword)
    return summary


# ============================================================
# Chart helpers (Altair)
# ============================================================

def make_timeline_chart(df: pd.DataFrame):
    if "published_date" not in df.columns or "article_id" not in df.columns:
        return None

    tmp = df.dropna(subset=["published_date"]).copy()
    if tmp.empty:
        return None

    tmp["month"] = tmp["published_date"].dt.to_period("M").dt.to_timestamp()
    group = (
        tmp.groupby("month")["article_id"]
        .nunique()
        .reset_index(name="article_count")
    )

    if group.empty:
        return None

    chart = (
        alt.Chart(group)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %y")),
            y=alt.Y("article_count:Q", title="Number of matching articles"),
            tooltip=["month:T", "article_count:Q"],
        )
        .properties(height=260)
    )
    return chart


def make_fraud_topic_chart(df: pd.DataFrame):
    if "fraud_topic" not in df.columns or "article_id" not in df.columns:
        return None

    tmp = df.dropna(subset=["fraud_topic"]).copy()
    if tmp.empty:
        return None

    group = (
        tmp.groupby("fraud_topic")["article_id"]
        .nunique()
        .reset_index(name="article_count")
        .sort_values("article_count", ascending=False)
        .head(15)
    )

    total = group["article_count"].sum()
    group["share"] = group["article_count"] / total

    chart = (
        alt.Chart(group)
        .mark_bar()
        .encode(
            x=alt.X("article_count:Q", title="Number of matching articles"),
            y=alt.Y("fraud_topic:N", sort="-x", title="Fraud topic"),
            color=alt.Color("article_count:Q", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("fraud_topic:N", title="Fraud topic"),
                alt.Tooltip("article_count:Q", title="Articles"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=260)
    )
    return chart


def make_topics_heatmap(df: pd.DataFrame):
    """
    Heatmap: fraud_topic vs month.
    """
    if "fraud_topic" not in df.columns or "published_date" not in df.columns or "article_id" not in df.columns:
        return None

    tmp = df.dropna(subset=["fraud_topic", "published_date"]).copy()
    if tmp.empty:
        return None

    tmp["month"] = tmp["published_date"].dt.to_period("M").dt.to_timestamp()

    group = (
        tmp.groupby(["fraud_topic", "month"])["article_id"]
        .nunique()
        .reset_index(name="article_count")
    )

    if group.empty:
        return None

    topic_order = (
        group.groupby("fraud_topic")["article_count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    chart = (
        alt.Chart(group)
        .mark_rect()
        .encode(
            x=alt.X(
                "month:T",
                title="Month",
                axis=alt.Axis(format="%b %y", labelAngle=-40),
            ),
            y=alt.Y(
                "fraud_topic:N",
                title="Fraud topic",
                sort=topic_order,
            ),
            color=alt.Color(
                "article_count:Q",
                title="Articles",
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[
                alt.Tooltip("fraud_topic:N", title="Fraud topic"),
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("article_count:Q", title="Matching articles"),
            ],
        )
        .properties(height=260)
    )
    return chart


def make_top_terms_chart(df_terms: pd.DataFrame):
    if df_terms.empty:
        return None

    chart = (
        alt.Chart(df_terms)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Mentions"),
            y=alt.Y("term:N", sort="-x", title="Keyword"),
            color=alt.Color("count:Q", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=["term:N", "count:Q"],
        )
        .properties(height=260)
    )
    return chart


def prepare_wordcloud_data(df_terms: pd.DataFrame, n_cols: int = 6) -> pd.DataFrame:
    if df_terms.empty:
        return df_terms

    df = df_terms.copy().reset_index(drop=True)
    df["rank"] = np.arange(len(df))
    df["col"] = df["rank"] % n_cols
    df["row"] = df["rank"] // n_cols
    return df


def make_wordcloud_chart(df_terms: pd.DataFrame):
    if df_terms.empty:
        return None

    wc_df = prepare_wordcloud_data(df_terms, n_cols=6)

    chart = (
        alt.Chart(wc_df)
        .mark_text()
        .encode(
            x=alt.X("col:O", axis=None),
            y=alt.Y("row:O", axis=None, sort="descending"),
            text="term:N",
            size=alt.Size(
                "count:Q",
                title="Frequency",
                scale=alt.Scale(range=[18, 70]),
            ),
            color=alt.Color(
                "count:Q",
                scale=alt.Scale(scheme="blues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("term:N", title="Term"),
                alt.Tooltip("count:Q", title="Mentions"),
            ],
        )
        .properties(height=260)
    )
    return chart


def make_embedding_chart(df: pd.DataFrame):
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return None

    tmp = df.dropna(subset=["x", "y"]).copy()
    if tmp.empty:
        return None

    if "fraud_topic" not in tmp.columns:
        tmp["fraud_topic"] = "Unknown"

    tooltip_fields = []
    if "title" in tmp.columns:
        tooltip_fields.append(alt.Tooltip("title:N", title="Title"))
    if "published_date" in tmp.columns:
        tooltip_fields.append(alt.Tooltip("published_date:T", title="Date", format="%Y-%m-%d"))
    if "fraud_topic" in tmp.columns:
        tooltip_fields.append(alt.Tooltip("fraud_topic:N", title="Fraud topic"))
    if "source" in tmp.columns:
        tooltip_fields.append(alt.Tooltip("source:N", title="Source"))

    chart = (
        alt.Chart(tmp)
        .mark_circle(size=70, opacity=0.85)
        .encode(
            x=alt.X("x:Q", title="Embedding X"),
            y=alt.Y("y:Q", title="Embedding Y"),
            color=alt.Color("fraud_topic:N", title="Fraud topic"),
            tooltip=tooltip_fields,
        )
        .properties(height=320)
        .interactive()
    )
    return chart


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Fraud News Explorer", layout="wide")

st.title("Fraud News Explorer")
st.write(
    "This app is part of a class project. "
    "It looks at ABA fraud-related news articles and shows how key words, topics, "
    "and sources show up over time."
)
st.caption("Data source: ABA fraud news articles stored in a Supabase Postgres database.")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    
    st.session_state.all_articles = st.checkbox("Load all articles", value=False)
    total_rows = fetch_number_of_rows(ARTICLES_TABLE)
    # If "Load all articles" is checked, update max_rows to total count
    if st.session_state.all_articles:
        st.session_state.max_rows = total_rows
        st.number_input(
            "Maximum number of articles to load",
            min_value=5,
            max_value=total_rows,
            step=5,
            value=total_rows,
            disabled=True,
            help="All articles are being loaded"
        )
    else:
        st.session_state.max_rows = st.number_input(
            "Maximum number of articles to load",
            min_value=5,
            max_value=total_rows,
            step=5,
            value=st.session_state.max_rows if st.session_state.max_rows <= 150 else 20,
        )
    st.session_state.case_sensitive = st.checkbox("Case-sensitive search", value=False)

    st.markdown("**Tables used:**")
    st.markdown(f"- `{ARTICLES_TABLE}`")
    st.markdown(f"- `{EXTRACT_TABLE}`")
    st.markdown(f"- `{EMBEDDINGS_TABLE}`")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df_all = load_data(max_rows=st.session_state.max_rows)
if df_all is None or df_all.empty:
    st.stop()

st.markdown("---")

# ------------------------------------------------------------
# Keyword search section
# ------------------------------------------------------------
st.subheader("Keyword search across fraud articles")

preset_options = [
    "Type a custom word…",
    "fraud",
    "scam",
    "consumer protection",
    "scams",
    "check fraud",
    "phishing",
    "cybersecurity",
    "risk management",
    "regulation",
    "financial services",
]

cols_kw = st.columns([2, 3])
with cols_kw[0]:
    preset = st.selectbox("Preset keywords", preset_options, index=1)
with cols_kw[1]:
    custom = st.text_input(
        "Or type your own word",
        value="",
        help="Examples: fraud, scams, regulation, financial services",
    )

min_articles = st.slider(
    "Warning threshold for number of matching articles",
    min_value=1,
    max_value=50,
    value=5,
)

# Decide keyword
st.session_state.keyword = ""
if custom.strip():
    st.session_state.keyword = custom.strip()
elif preset != preset_options[0]:
    st.session_state.keyword = preset

if not st.session_state.keyword:
    st.info("Choose a preset keyword or type your own word to explore the articles.")
    st.stop()

# Filter
filtered_df = filter_by_keyword(df_all, st.session_state.keyword, case_sensitive=st.session_state.case_sensitive)

if filtered_df.empty:
    st.warning(f"No articles found that mention '{st.session_state.keyword}'. Try a different word.")
    st.stop()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
if "article_id" in filtered_df.columns:
    n_articles = filtered_df["article_id"].nunique()
else:
    n_articles = len(filtered_df)

unique_sources = filtered_df["source"].nunique() if "source" in filtered_df.columns else 0

if "content" in filtered_df.columns:
    word_counts = filtered_df["content"].fillna("").apply(
        lambda x: len(str(x).split())
    )
    avg_words = int(word_counts.mean()) if not word_counts.empty else 0
else:
    avg_words = 0

if "published_date" in filtered_df.columns:
    dates = filtered_df["published_date"].dropna()
    if not dates.empty:
        date_min = dates.min().strftime("%Y-%m-%d")
        date_max = dates.max().strftime("%Y-%m-%d")
        date_range_text = f"{date_min} → {date_max}"
    else:
        date_range_text = "N/A"
else:
    date_range_text = "N/A"

if "fraud_topic" in filtered_df.columns:
    n_topics = filtered_df["fraud_topic"].dropna().nunique()
else:
    n_topics = 0

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Articles mentioning '{st.session_state.keyword}'", n_articles)
k2.metric("Distinct fraud topics", n_topics if n_topics else "N/A")
k3.metric("Unique sources", unique_sources)
k4.metric("Avg. article length (words)", avg_words if avg_words else "N/A")

st.caption(f"Date range of matching articles: {date_range_text}")

if n_articles < min_articles:
    st.warning(
        f"Only {n_articles} articles mention this word — below your warning threshold of {min_articles}."
    )
if st.session_state.keyword:
    st.session_state.enable_llm_analysis = st.checkbox("Enable LLM analysis", value=False)

st.markdown("---")

# ------------------------------------------------------------
# LLM analysis
# ------------------------------------------------------------
if st.session_state.enable_llm_analysis:
    st.subheader("LLM analysis")
    st.write("LLM analysis is enabled")
    with st.spinner("Generating summary..."):
        summary = asyncio.run(generate_llm_summary(st.session_state.keyword))
    st.session_state.llm_summary = summary
    st.markdown(st.session_state.llm_summary)

    st.markdown('---')

# ------------------------------------------------------------
# Tabs for charts and table
# ------------------------------------------------------------
tab_overview, tab_text, tab_embed, tab_table = st.tabs(
    ["Overview charts", "Text insights", "Embedding map", "Articles table"]
)

# ---------- Overview tab ----------
with tab_overview:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Timeline of matching articles")
        chart_timeline = make_timeline_chart(filtered_df)
        if chart_timeline is not None:
            st.altair_chart(chart_timeline, use_container_width=True)
        else:
            st.info("Not enough date information to build a timeline.")

    with col_right:
        st.subheader("Top fraud topics in matching articles")
        chart_topics = make_fraud_topic_chart(filtered_df)
        if chart_topics is not None:
            st.altair_chart(chart_topics, use_container_width=True)
        else:
            st.info("No fraud topic information available.")

    st.subheader("Fraud topics by month (heatmap)")
    heatmap = make_topics_heatmap(filtered_df)
    if heatmap is not None:
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Not enough topic + date information to build a heatmap.")

# ---------- Text insights tab ----------
with tab_text:
    st.subheader("Top terms in article content")
    if "content" in filtered_df.columns:
        terms_df = get_top_terms(filtered_df["content"], n=25)
        terms_chart = make_top_terms_chart(terms_df)
        if terms_chart is not None:
            st.altair_chart(terms_chart, use_container_width=True)
        else:
            st.info("Not enough text to compute term frequencies.")
    else:
        st.info("No article content available.")

    st.subheader("Word cloud of top terms")
    if "content" in filtered_df.columns:
        wc_terms_df = get_top_terms(filtered_df["content"], n=40)
        wc_chart = make_wordcloud_chart(wc_terms_df)
        if wc_chart is not None:
            st.altair_chart(wc_chart, use_container_width=True)
        else:
            st.info("Not enough text to build a word cloud.")
    else:
        st.info("No article content available to build a word cloud.")

# ---------- Embedding map tab ----------
with tab_embed:
    st.subheader("Embedding map of similar articles")
    emb_chart = make_embedding_chart(filtered_df)
    if emb_chart is not None:
        st.altair_chart(emb_chart, use_container_width=True)
    else:
        st.info(
            "No valid embeddings were found. "
            "Check that the embeddings table has an 'embedding' column."
        )

# ---------- Articles table tab ----------
with tab_table:
    st.subheader("Articles table")

    table_cols = [
        c
        for c in ["published_date", "title", "fraud_topic", "source_url", "source"]
        if c in filtered_df.columns
    ]
    if table_cols:
        table_df = (
            filtered_df[table_cols]
            .sort_values(
                by="published_date" if "published_date" in table_cols else table_cols[0],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # Download button
        csv = table_df.to_csv(index=False).encode("utf-8")
        file_keyword = re.sub(r"\W+", "_", st.session_state.keyword.lower())
        st.download_button(
            label="Download matching articles as CSV",
            data=csv,
            file_name=f"articles_{file_keyword}.csv",
            mime="text/csv",
        )
    else:
        st.info("No columns available to display an articles table.")

