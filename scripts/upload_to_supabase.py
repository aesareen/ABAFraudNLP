from supabase import create_client, Client
import os
from dotenv import load_dotenv
import glob
import json
from datetime import datetime, timezone
from rich import print
from postgrest.exceptions import APIError
from rich.progress import Progress
from openai import OpenAI

load_dotenv(dotenv_path="config/.env", override=True)

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
openai_api_key: str = os.getenv("OPENAI_API_KEY")

CLIENT: Client = create_client(url, key)
OPENAI_CLIENT: OpenAI = OpenAI(api_key=openai_api_key)

def get_embeddings(text: str):
    response = OPENAI_CLIENT.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def upload_json_to_supabase(json_content: dict):
    json_dict = json_content[0]
    article_name = json_dict.get('name')

    if not article_name:
        raise ValueError("Article name is required")

    # Check if article with this name already exists
    existing = CLIENT.table("articles").select("*").eq("name", article_name).execute()

    # If article exists, update it
    if existing.data:
        response = CLIENT.table("articles").update(json_dict).eq("name", article_name).execute()
        return response
    else: # otherwise, insert it
        json_dict['created_at'] = datetime.now(timezone.utc).isoformat()
        response = CLIENT.table("articles").insert(json_dict).execute()
        return response

def upload_embeddings_to_supabase(article_name: str, raw_content: str):
    if not article_name:
        raise ValueError("Article name is required")
    if not raw_content:
        raise ValueError("You must pass in some raw text content from an ABA article to get embeddings")
    
    # Check if article with this name already exists
    existing_article = CLIENT.table("articles").select("*").eq("article_name", article_name).single().execute().data.get('article_id')
    if existing_article:
        response = CLIENT.table("article_embeddings").insert({
            "article_id": existing_article,
            "embedding": get_embeddings(raw_content)
        }).execute()
        return response.data, "inserted"
    else:
        raise ValueError("Article not found")
        
    return response

def upload_keywords_to_supabase(article_name: str, keywords: list[str]) -> bool:
    if not article_name:
        raise ValueError("Article name is required")
    if not keywords:
        raise ValueError("You must pass in some keywords to upload")
    
    # Check if article with this name already exists
    existing_article = CLIENT.table("articles").select("*").eq("article_name", article_name).single().execute().data.get('article_id')
    if existing_article:
        for keyword in keywords:
            CLIENT.table("article_extract").insert({
                "article_id": existing_article,
                "type": "keyword",
                "text": keyword
            }).execute()
    else:
        raise ValueError("Article not found")
    
    return True

def upload_summary_to_supabase(article_name: str, summary: str):
    if not article_name:
        raise ValueError("Article name is required")
    if not summary:
        raise ValueError("You must pass in a summary to upload")
    
    # Check if article with this name already exists
    existing_article = CLIENT.table("articles").select("*").eq("article_name", article_name).single().execute().data.get('article_id')
    if existing_article:
        CLIENT.table("article_extract").insert({
            "article_id": existing_article,
            "type": "summary",
            "text": summary
        }).execute()
    else:
        raise ValueError("Article not found")
    
    return True

if __name__ == "__main__":
    # Load in all the JSON files in our data folder
    json_files = glob.glob("data/scraped_json_results/*.json")
    with Progress() as progress:
        for file in progress.track(json_files, description="Uploading articles"):
            with open(file, "r", encoding="utf-8") as f:
                json_content = json.load(f)
            try:
                # upload_response = upload_json_to_supabase(json_content)
                embedding_response = upload_embeddings_to_supabase(json_content[0]['article_name'], json_content[0]['raw_content'])
                print(f"[green]Successfully uploaded {file} to Supabase[/green]")
            except APIError as e:
                print(f"[red]Failed to upload {file} to Supabase[/red]")
                print(f"Error: {e}")
                continue
            except ValueError as e:
                print(f"[red]Validation error for {file}: {e}[/red]")
                continue