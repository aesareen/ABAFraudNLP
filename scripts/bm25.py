from rank_bm25 import BM25Okapi
import pickle
import os
import string
import glob
from rich.progress import Progress
from dotenv import load_dotenv
from rich import print

load_dotenv(dotenv_path="config/.env", override=True)

BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH")


def create_bm25_index(documents: list[str], save: bool = False) -> BM25Okapi:
    """Given a list of documents, generate a BM25 Index

    Args:
        documents (list[str]): a list of raw text documents to index
        save (bool, optional): whether to save the index to a pickled file. Defaults to False.

    Returns:
        BM25Okapi: a BM25 Index object
    """
    # remove punctuation from documents and then split each word into tokens based on whitespace
    tokenized_corpus = [
        doc.translate(str.maketrans("", "", string.punctuation)).split(" ")
        for doc in documents
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    if save:
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25, f)
    return bm25


def get_or_create_bm25_index(documents: list[str], save: bool = False):
    """Given a list of documents, get or create a BM25 Index

    Args:
        documents (list[str]): a list of raw text documents to index
        save (bool, optional): whether to save the index to a pickled file. Defaults to False.

    Returns:
        BM25Okapi: a BM25 Index object
    """
    if os.path.exists(BM25_INDEX_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)
    else:
        bm25 = create_bm25_index(documents, save)
    return bm25


def query_bm25_index(query: str, bm25: BM25Okapi, corpus: list[str], n: int = 3):
    """Query a BM25 Index

    Args:
        query (str): query string
        bm25 (BM25Okapi): BM25 Index Object
        corpus (list[str]): a list of raw text documents to index
        n (int, optional): the number of documents to return. Defaults to 3.

    Returns:
        list[float]: a list of scores for each document in the corpus
    """
    # In order to get good results, we need to follow the same pre-processing steps that we did when we created our index
    tokenized_query = query.translate(str.maketrans("", "", string.punctuation)).split(
        " "
    )
    scores = bm25.get_top_n(tokenized_query, corpus, n=n)
    return scores

if __name__ == "__main__":
    # We can load in the markdown files we got from Crawl4AI
    query = "ATM fraud"
    markdown_files = glob.glob("data/scraped_markdown_results/*.md")
    article_contents = []

    print("[bold green]Loading articles...[/bold green]")

    with Progress() as progress:
        for file in progress.track(markdown_files, description="Loading articles"):
            with open(file, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            article_contents.append(markdown_content)
    
    bm25 = get_or_create_bm25_index(article_contents)
    top_documents = query_bm25_index(query, bm25, article_contents)
    print(top_documents)