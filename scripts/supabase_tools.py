import supabase
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
import logging
from openai import OpenAI

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(dotenv_path=os.path.join(project_root, "config/.env"), override=True)

SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

CLIENT: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
OPENAI_CLIENT: OpenAI = OpenAI(api_key=OPENAI_API_KEY)

LOGGER = logging.getLogger(__name__)


def get_embeddings(text: str):
    """Generate embeddings using OpenAI's embedding model."""
    response = OPENAI_CLIENT.embeddings.create(
        input=text.lower().strip(),
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

mcp_server = FastMCP(
    name="supabase_tools",
    instructions="This server provides the ability to query the supabase tables in a variety of ways",
)

@mcp_server.tool(
    description="Query the supabase table for the most relevant articles and summaries based on a query",
)
def query_embeddings_from_supabase(query: str = Field(..., description="The query to search for")) -> str:
    embedded_query = get_embeddings(query)
    response = CLIENT.rpc("match_documents", {
        "query_embedding": embedded_query,
        "match_threshold": 0.4,
        "match_count": 4
    }).execute()
    LOGGER.debug(f"Response: {response.data}")
    # Ideally we would return the article names and contents, so we can do a join on the article_id and get the article names and contents
    article_ids = [item['article_id'] for item in response.data]
    articles = CLIENT.table("articles").select("article_name, raw_content").in_("article_id", article_ids).execute()
    summaries = CLIENT.table("article_extract").select("text").eq("type", "summary").in_("article_id", article_ids).execute()
    article_names = [article['article_name'] for article in articles.data]
    article_contents = [article['raw_content'] for article in articles.data]
    summaries = [summary['text'] for summary in summaries.data]
    
    output = ""
    for i, (article_name, summary) in enumerate(zip(article_names, summaries)):
        output += f"Article {i+1}: {article_name}\nArticle Summary: {summary}\n\n"
    
    return output

if __name__ == "__main__":
    # print(query_embeddings_from_supabase("cybersecurity"))
    mcp_server.run(show_banner=False)