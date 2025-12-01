
# ABAFraudNLP

AI-powered NLP system for scraping, enriching, and analyzing fraud-related articles from the American Bankkers Association (ABA).

**Team:** Maksim Dimitrijević • Arnav Sareen • Ana Abreu • Nabeel Balighuddin  
UNC Charlotte • DTSC 3602 – USAA Fraud Analytics Project  

---

## 1. Clear First Impression

**One-sentence summary:**  
ABAFraudNLP automatically scrapes ABA fraud articles, cleans and processes them, generates LLM-powered summaries and keywords, embeds them using OpenAI vectors, and visualizes fraud trends in a searchable Streamlit dashboard backed by Supabase.

---

## 2. Quick Start

### Install with `uv`
```bash
uv sync
````

### Environment Variables

The app expects a `.env` file inside the `config/` directory.

```bash
cp config/.env.example config/.env
```

Then edit `config/.env`:

```env
OPENAI_API_KEY=your_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Run the full pipeline

```bash
# 1. Scrape ABA articles → data/scraped_markdown_results/
uv run python scripts/scraper.py

# 2. Generate keywords using LLM agent
uv run python agents/summarization_agent.py

# 3. Upload articles + keywords + embeddings to Supabase
uv run python scripts/upload_to_supabase.py

# 4. (Optional) BM25 keyword search service
uv run python scripts/bm25.py
```

### Launch the Streamlit dashboard

```bash
uv run streamlit run streamlit/streamlit_app.py
```

---

## 3. Visuals / Application Design

### 3.1 Architecture Diagram

```mermaid
flowchart TD
    A["ABA Fraud Articles"] --> B["Scraper (Crawl4AI)"]
    B --> C["Raw Markdown & JSON"]
    C --> D["Cleaning + Standardization"]
    D --> E["LLM Summaries + Keywords"]
    D --> F["OpenAI Embeddings"]
    E --> G["Supabase: article_extract"]
    D --> H["Supabase: articles"]
    F --> I["Supabase: article_embeddings"]
    H --> J["BM25 Index (bm25.py)"]
    G --> K["Streamlit Dashboard"]
    I --> K
    H --> K
```

---

### 3.2 Real Streamlit Dashboard Visuals

![Dashboard](dash.png)
![Embedding Map](embed.png)
![Category Frequency](rename.png)
![Heatmap](images/heat.png)

---

### 3.3 Supabase Database Schema

![Supabase Schema](supa.png)

---

### 3.4 Folder Structure (Real)

```
ABAFraudNLP/
├── agents/
│   ├── prompts.py
│   └── summarization_agent.py
├── config/
│   └── .env.example
├── data/
│   ├── scraped_markdown_results/
│   ├── scraped_json_results/
│   └── manual_search.txt
├── images/
├── scripts/
│   ├── scraper.py
│   ├── upload_to_supabase.py
│   └── bm25.py
├── streamlit/
│   └── streamlit_app.py
├── visualizations/
│   └── visualization.ipynb
├── pyproject.toml
└── README.md
```

---

## 3a. What We Did (with real examples)

### 3a.1 Real Sample Article (Short Preview)

**Source:**
`data/scraped_markdown_results/aba-consumer-group-urge-action-by-voice-service-providers-to-combat-fraud.md`

**Preview:**

> “A coalition of consumer groups is urging voice service providers to take stronger action to combat fraudulent robocalls and phone-based scams. The groups emphasized the growing risks posed by spoofed caller IDs and called for more aggressive enforcement…”

---

### 3a.2 Minimal Example of LLM Keyword Extraction

```python
agent = initialize_summarization_agent()

response = await agent.generate_article_keywords(
    article_text,
    []
)

print(response)
```

---

### 3a.3 Minimal Example of Embedding Generation

```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=clean_text
).data[0].embedding
```

---

### 3a.4 Minimal Example of BM25 Search

```python
bm25, tokenized = create_bm25_index(all_articles)
results = query_bm25_index("cyber fraud attack")
```

---

## 4. Clear Findings

### What This Project Reveals

* Banking Security and Cyber Fraud dominate ABA reporting
* Customer Impact appears frequently
* Elder Fraud and Training/Awareness are underrepresented
* Embeddings cluster articles into meaningful groups
* Co-occurrence patterns show strong links (e.g., Cyber Fraud ↔ Banking Security)

---

## Key Visual Findings (Real)

![Category Frequency](images/frequency_chart.png)
![Heatmap](images/heatmap.png)
![Embedding Map](images/embedding_map.png)

---

# Additional Links

* Streamlit: [https://streamlit.io](https://streamlit.io)
* Supabase: [https://supabase.com](https://supabase.com)
* crawl4ai: [https://github.com/unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)

```
```






