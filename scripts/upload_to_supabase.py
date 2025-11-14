from supabase import create_client, Client
import os
from dotenv import load_dotenv
import glob
import json
from datetime import datetime, timezone
from rich import print
from postgrest.exceptions import APIError
from rich.progress import Progress

load_dotenv(dotenv_path="config/.env")

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

CLIENT: Client = create_client(url, key)

def upload_to_supabase(json_content: dict):
    json_dict = json_content[0]
    article_name = json_dict.get('name')

    if not article_name:
        raise ValueError("Article name is required")

    # Check if article with this name already exists
    existing = CLIENT.table("articles").select("*").eq("name", article_name).execute()

    # If article exists, update it
    if existing.data:
        response = CLIENT.table("articles").update(json_dict).eq("name", article_name).execute()
        return response.data, "updated"
    else: # otherwise, insert it
        json_dict['created_at'] = datetime.now(timezone.utc).isoformat()
        response = CLIENT.table("articles").insert(json_dict).execute()
        return response.data, "inserted"

if __name__ == "__main__":
    # Load in all the JSON files in our data folder
    json_files = glob.glob("data/scraped_json_results/*.json")
    with Progress() as progress:
        for file in progress.track(json_files, description="Uploading articles"):
            with open(file, "r", encoding="utf-8") as f:
                json_content = json.load(f)
            try:
                response, operation = upload_to_supabase(json_content)
                print(f"[green]Successfully {operation} {file} to Supabase[/green]")
            except APIError as e:
                print(f"[red]Failed to upload {file} to Supabase[/red]")
                print(f"Error: {e}")
                continue
            except ValueError as e:
                print(f"[red]Validation error for {file}: {e}[/red]")
                continue