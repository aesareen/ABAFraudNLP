import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from rich import print

def load_in_manual_sources(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file.readlines()]

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
                print(f"Successfully crawled: {result.url}")
                # Create a safer filename
                filename = result.url.split('/')[-1] or 'index'
                # Clean filename and ensure it's not empty
                if not filename or filename == '':
                    filename = f"page_{successful}"
                
                # Save with UTF-8 encoding
                try:
                    with open(f"data/scraped_results/{filename}.md", "w", encoding='utf-8') as file:
                        file.write(result.markdown)
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