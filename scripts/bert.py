from summarizer import Summarizer
from keybert import KeyBERT, KeyLLM
import nltk
import openai
import glob
import json
from keybert.llm import LiteLLM
from dotenv import load_dotenv
from upload_to_supabase import upload_keywords_to_supabase, upload_summary_to_supabase
import os
from rich import print
from rich.progress import Progress

load_dotenv(dotenv_path="config/.env", override=True)

# Stop words are words that are commonly used in the English language that are not useful for keyword extraction
# So, we can use the built-in NLTK stopwords to remove them from our articles
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords


SUMMARIZER = Summarizer()
CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM = LiteLLM("gpt-5-mini")
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def summarize_article(article: str) -> str:
    """Summarize an article using BERT

    Args:
        article (str): The article to summarize

    Returns:
        str: The summarized article
    """
    # Remove stopwords from the article
    article = ' '.join([word for word in article.split() if word not in stopwords.words('english')])
    summary = SUMMARIZER(article, min_length=60, max_length=200)
    return summary

def generate_article_summaries(article_contents: list[str]) -> list[str]:
    """Generate summaries for a list of articles
    
    Args:
        article_contents (list[str]): The contents of the articles
    
    Returns:
        list[str]: The summaries for the articles
    """
    summaries = [summarize_article(article) for article in article_contents]
    return summaries

# KeyBert has this really amazing capability where we can create keywords in a first pass with BERT
# Then use an LLM to refine, touch them up, and expand them—which directly aligns with our goals of using both traditional and emerging NLP technologies!
def generate_keywords(article_contents: list[str]) -> list[str]:
    """Generate keywords for a list of articles
    
    Args:
        article_contents (list[str]): The contents of the articles
    
    Returns:
        list[str]: The keywords for the articles
    """
    kw_model = KeyBERT(llm=LLM)
    keywords = kw_model.extract_keywords(article_contents, keyphrase_ngram_range=(1, 2), stop_words='english')
    return keywords

def touch_up_keywords(keywords: list[str]) -> list[str]:
    """Touch up on the generated keywords by lowercasing them and removing extraneous punctuation
    
    Args:
        keywords (list[str]): The keywords to touch up
    
    Returns:
        list[str]: The touched up keywords
    """
    keywords = [keyword.lower().strip().replace("<", "").replace(">","").replace("#","") for keyword in keywords]
    return keywords

def generate_and_touch_up_keywords(article_contents: list[str]) -> list[str]:
    """Generate and touch up keywords for a list of articles
    
    Args:
        article_contents (list[str]): The contents of the articles
    
    Returns:
        list[str]: The generated and touched up keywords
    """
    keywords = generate_keywords(article_contents)
    keywords = [touch_up_keywords(keyword_list) for keyword_list in keywords]
    return keywords

if __name__ == "__main__":
    article_files = glob.glob(os.path.join(project_root, "data/scraped_json_results/*.json"))
    article_names = []
    article_contents = []
    for article_file in article_files:
        with open(article_file, "r") as f:
            json_data = json.load(f)
            article_name = json_data[0]["article_name"]
            article_names.append(article_name)  
            article_content = json_data[0]["raw_content"]
            article_contents.append(article_content)
    
    print(f"Generating keywords for {len(article_names)} articles")
    keywords = generate_and_touch_up_keywords(article_contents)
    print(f"Generating summaries for {len(article_names)} articles")
    summaries = generate_article_summaries(article_contents)
    with Progress() as progress:
        task = progress.add_task("Processing articles...", total=len(article_names))
        for article_name, keywords_each, summary in zip(article_names, keywords, summaries):
            upload_keywords_to_supabase(article_name, keywords_each)
            upload_summary_to_supabase(article_name, summary)
            print(f"[green]Successfully uploaded keywords and summaries for {article_name}[/green]")
            progress.advance(task)
