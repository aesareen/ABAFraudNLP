import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from rich import print
from urllib.parse import urlparse
import re

def load_in_manual_sources(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file.readlines()]

def extract_filename_from_url(url):
    """Extract a human-readable filename from a URL."""
    # Parse the URL to get the path
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')  # Remove trailing slashes
    
    # Get the last segment of the path
    segments = [s for s in path.split('/') if s]  # Filter out empty segments
    
    if segments:
        # Use the last segment as filename
        filename = segments[-1]
        
        # If it looks like a file with extension, remove it
        if '.' in filename and not filename.startswith('.'):
            filename = filename.rsplit('.', 1)[0]
        
        # Clean the filename - remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        return filename if filename else 'index'
    
    # Fallback: use domain name
    return parsed.netloc.replace('.', '_')

async def scrape_website(urls):
    config = CrawlerRunConfig(
        verbose=True,
        stream=True,
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(),
        page_timeout=60000,  # 60 seconds timeout
        delay_before_return_html=2.0,  # Wait 2 seconds before extracting
    )
    
    successful = 0
    failed = 0
    seen_filenames = {}  # Track filenames to handle duplicates
    
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun_many(
            urls=urls,
            config=config
        ):
            if result.success:
                print(f"[green]Successfully crawled: {result.url}[/green]")
                
                # Extract clean, human-readable filename
                base_filename = extract_filename_from_url(result.url)
                
                # Handle duplicate filenames by appending a counter
                filename = base_filename
                if filename in seen_filenames:
                    seen_filenames[filename] += 1
                    filename = f"{base_filename}_{seen_filenames[filename]}"
                else:
                    seen_filenames[filename] = 0
                
                # Save with UTF-8 encoding
                try:
                    filepath = f"data/scraped_results/{filename}.md"
                    with open(filepath, "w", encoding='utf-8') as file:
                        file.write(result.markdown)
                    print(f"  → Saved as: [cyan]{filename}.md[/cyan]")
                    successful += 1
                except Exception as e:
                    print(f"[red]Error saving {result.url}: {e}[/red]")
                    failed += 1
            else:
                print(f"[red]Failed to crawl: {result.url}[/red]")
                print(f"Error: {result.error_message}")
                print("---")
                failed += 1
                # Don't return False - continue processing other URLs
    
    print(f"\n[blue]Summary: {successful} successful, {failed} failed[/blue]")
    return True  # Return success as long as we processed something

async def main():
    urls = load_in_manual_sources('data/manual_search.txt')
    print(f"[blue]Scraping {len(urls)} URLs[/blue]")
    success = await scrape_website(urls)
    if success:
        print("[green]Scraping completed successfully[/green]")
    else:
        print("[red]Scraping failed[/red]")    

if __name__ == "__main__":
    asyncio.run(main())