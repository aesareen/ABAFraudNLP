from rank_bm25 import BM25Okapi
import pickle
import os
from fastmcp import FastMCP
from pydantic import Field
import logging
import string
import glob
from rich.progress import Progress
from dotenv import load_dotenv

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(dotenv_path=os.path.join(project_root, "config/.env"), override=True)

BM25_INDEX_PATH: str = os.path.join(project_root, os.getenv("BM25_INDEX_PATH"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


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
    return bm25, tokenized_corpus


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
        bm25, tokenized_corpus = create_bm25_index(documents, save)
    return bm25, tokenized_corpus



# This is needed when we set up this as a MCP Server as we need to know where the project root is to access the data folder
markdown_files = glob.glob(os.path.join(project_root, "data/scraped_markdown_results/*.md"))
article_contents = []

LOGGER.info("Loading articles...")
for file in markdown_files:
    with open(file, "r", encoding="utf-8") as f:
        markdown_content = f.read()
    article_contents.append(markdown_content)
LOGGER.info("Articles loaded successfully")

LOGGER.info("Creating BM25 Index...")

BM25, tokenized_corpus = get_or_create_bm25_index(article_contents)

mcp_server = FastMCP(
    name="BM25_Index",
    instructions="This server provides the ability to query a BM25 Index. This should be used for all keyword-based searches, while semantic searches should be done with the PGVectors Server.",
)


@mcp_server.tool(
    description="Using a BM25 Keyword Based Search, return relevant articles from the database of articles. Submit a query and get the top 3 articles back"
)
def query_bm25_index(
    query: str = Field(..., description="the query string"),
):
    """Query a BM25 Index

    Args:
        query (str): query string
        n (int, optional): the number of documents to return. Defaults to 3.

    Returns:
        list[str]: a list of the top N article contents (original markdown)
    """
    # In order to get good results, we need to follow the same pre-processing steps that we did when we created our index
    tokenized_query = query.translate(str.maketrans("", "", string.punctuation)).split(
        " "
    )
    # Get BM25 scores for all documents
    scores = BM25.get_scores(tokenized_query)

    # Get indices of top N documents sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    # Return the original article contents for the top N documents
    top_documents = [article_contents[i] for i in top_indices]

    return top_documents


if __name__ == "__main__":
    # We can load in the markdown files we got from Crawl4AI
    mcp_server.run(show_banner=False)
