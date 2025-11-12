import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from rich import print
from urllib.parse import urlparse
import re

def load_in_manual_sources(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file.readlines()]

def extract_filename_from_url(url):
    """Extract a human-readable filename from a URL."""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    
    segments = [s for s in path.split('/') if s]
    
    if segments:
        filename = segments[-1]
        
        if '.' in filename and not filename.startswith('.'):
            filename = filename.rsplit('.', 1)[0]
        
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        return filename if filename else 'index'
    
    return parsed.netloc.replace('.', '_')


def parse_result(markdown_content: str):
    lines = markdown_content.split('\n')
    date_lines = []
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    result = lines
    for i, line in enumerate(lines):
        if any(month in line.strip().lower() for month in months):
            date_lines.append(line)
            continue
        if line.strip().startswith('# '):
            result = lines[i:]
        # The minute we see this copyright sign, everything after this is absolutely useless, so we can stop
        if '©' in line.strip():
            result = date_lines + result
            return '\n'.join(result)
    
    return '\n'.join(result)


async def scrape_website(urls):
    config = CrawlerRunConfig(
        verbose=True,
        stream=True,
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(
            options = {
                "ignore_links": True,
                "ignore_images": True,
                "skip_internal_links": True,
            },
            content_filter = PruningContentFilter(
                threshold=0.80,
                threshold_type="dynamic",
                min_word_threshold=0
        ), 
        ),
        excluded_tags = [],
        exclude_social_media_links = True,
        exclude_external_links = True,
        page_timeout=60000, 
        delay_before_return_html=2.0, 
    )
    
    successful = 0
    failed = 0
    seen_filenames = {}
    
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun_many(
            urls=urls,
            config=config
        ):
            if result.success:
                print(f"[green]Successfully crawled: {result.url}[/green]")
                
                base_filename = extract_filename_from_url(result.url)
                
                filename = base_filename
                if filename in seen_filenames:
                    seen_filenames[filename] += 1
                    filename = f"{base_filename}_{seen_filenames[filename]}"
                else:
                    seen_filenames[filename] = 0

                result = parse_result(result.markdown)
                
                try:
                    filepath = f"data/scraped_results/{filename}.md"
                    with open(filepath, "w", encoding='utf-8') as file:
                        file.write(result)
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
                
    
    print(f"\n[blue]Summary: {successful} successful, {failed} failed[/blue]")
    return True  

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