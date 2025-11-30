import asyncio
import json
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
    JsonCssExtractionStrategy,
    LLMConfig,
    AdaptiveConfig,
    AdaptiveCrawler
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from rich import print
from urllib.parse import urlparse
import re
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="config/.env", override=True)

json_schemas = {
    "website": {
        "name": "Article",
        "baseSelector": "body",
        "fields": [
            {
                "name": "article_name",
                "selector": "h1",
                "type": "text",
            },
            {
                "name": "date_published",
                "selector": "ul.unlisted li.byline__inline",
                "type": "list",
                "fields": [{"name": "date", "type": "text"}],
            },
            {
                "name": "raw_content",
                "selector": "span.rich-text",
                "type": "text",
            },
        ],
    },
    "journal": {
        "name": "Journal",
        "baseSelector": "div.post-wrapper",
        "fields": [
            {
                "name": "article_name",
                "selector": "h1.jeg_post_title",
                "type": "text",
            },
            {
                "name": "date_published",
                "selector": "div.jeg_meta_date a:nth-of-type(1)",
                "type": "text",
            },
            {
                "name": "raw_content",
                "selector": "div.entry-content",
                "type": "text",
            },
        ],
    },
}


def load_in_manual_sources(filepath):
    with open(filepath, "r") as file:
        return [line.strip() for line in file.readlines()]


def extract_filename_from_url(url):
    """Extract a human-readable filename from a URL."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    segments = [s for s in path.split("/") if s]

    if segments:
        filename = segments[-1]

        if "." in filename and not filename.startswith("."):
            filename = filename.rsplit(".", 1)[0]

        filename = re.sub(r'[<>:"/\\|?*]', "", filename)

        return filename if filename else "index"

    return parsed.netloc.replace(".", "_")

async def scrape_website(urls, config):
    seen_filenames = {}
    successful = 0
    failed = 0
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun_many(urls=urls, config=config):
            if result.success:
                print(f"[green]Successfully crawled: {result.url}[/green]")

                base_filename = extract_filename_from_url(result.url)

                filename = base_filename
                if filename in seen_filenames:
                    seen_filenames[filename] += 1
                    filename = f"{base_filename}_{seen_filenames[filename]}"
                else:
                    seen_filenames[filename] = 0

                json_result = json.loads(result.extracted_content)

                # Sometimes articles say "For Immediate Release", and we don't really care for that, so we can overwrite the field with the actual date
                if (
                    isinstance(json_result[0]["date_published"], list)
                    and len(json_result[0]["date_published"]) > 1
                ):
                    json_result[0]["date_published"] = json_result[0]["date_published"][
                        1
                    ]["date"].split("Published ")[-1].strip()
                elif (
                    isinstance(json_result[0]["date_published"], list)
                    and len(json_result[0]["date_published"]) == 1
                ):
                    json_result[0]["date_published"] = json_result[0]["date_published"][
                        0
                    ]["date"].split("Published ")[-1].strip()
                elif isinstance(
                    json_result[0]["date_published"], str
                ):  # This is for journal ABA
                    json_result[0]["date_published"] = json_result[0]["date_published"]
                else:
                    json_result[0]["date_published"] = None

                # Sometimes there is weird unicode characters in our raw content, so we can convert to ASCII and then convert back to get rid of them
                json_result[0]["raw_content"] = (
                    json_result[0]["raw_content"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )

                json_result[0]["source_url"] = result.url

                markdown_result = result.markdown.fit_markdown # get the markdown only for the target elements

                try:
                    filepath = f"data/scraped_markdown_results/{filename}.md"
                    # with open(filepath, "w", encoding="utf-8") as file:
                    #     file.write(markdown_result)
                    print(f"  → Saved as: [cyan]{filename}.md[/cyan]")
                    # with open(
                    #     f"data/scraped_json_results/{filename}.json",
                    #     "w",
                    #     encoding="utf-8",
                    # ) as file:
                    #     json.dump(json_result, file, indent=4)
                    print(f"  → Saved as: [cyan]{filename}.json[/cyan]")
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


async def scrape_aba_website(urls):
    config = CrawlerRunConfig(
        verbose=True,
        stream=True,
        cache_mode=CacheMode.BYPASS,
        target_elements = [field['selector'] for field in json_schemas["website"]["fields"]],
        extraction_strategy=JsonCssExtractionStrategy(
            schema=json_schemas["website"], verbose=True
        ),
        markdown_generator=DefaultMarkdownGenerator(
            options={
                "ignore_links": True,
                "ignore_images": True,
                "skip_internal_links": True,
            },
            content_filter=PruningContentFilter(
                threshold=0.80, threshold_type="dynamic", min_word_threshold=0
            ),
        ),
        excluded_tags=[],
        exclude_social_media_links=True,
        exclude_external_links=True,
        page_timeout=60000,
        delay_before_return_html=2.0,
    )

    return await scrape_website(urls, config)


async def scrape_aba_journal(urls):
    config = CrawlerRunConfig(
        verbose=True,
        stream=True,
        cache_mode=CacheMode.BYPASS,
        target_elements = [field['selector'] for field in json_schemas["journal"]["fields"]],
        extraction_strategy=JsonCssExtractionStrategy(schema=json_schemas["journal"]),
        markdown_generator=DefaultMarkdownGenerator(
            options={
                "ignore_links": True,
                "ignore_images": True,
                "skip_internal_links": True,
            },
            content_filter=PruningContentFilter(
                threshold=0.80, threshold_type="dynamic", min_word_threshold=0
            ),
        ),
        excluded_tags=[],
        exclude_social_media_links=True,
        exclude_external_links=True,
        page_timeout=60000,
        delay_before_return_html=2.0,
    )

    return await scrape_website(urls, config)


# Adaptive Deep Crawling, but doesn't really work, might be helpful at some point
# async def adaptive_crawling_aba_journal(query: str):
#     search_page_async_crawler_config = CrawlerRunConfig(
#         css_selector = '.list-complex',
#         user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#         cache_mode=CacheMode.BYPASS,
#         magic=True,
#         exclude_external_links=False,
#         excluded_tags=['nav', 'header', 'footer', 'script', 'style'],
#     )
#     article_async_crawler_config = CrawlerRunConfig(
#         verbose=True,
#         stream=True,
#         cache_mode=CacheMode.BYPASS,
#         target_elements = [field['selector'] for field in json_schemas["website"]["fields"]],
#         extraction_strategy=JsonCssExtractionStrategy(
#             schema=json_schemas["website"], verbose=True
#         ),
#         markdown_generator=DefaultMarkdownGenerator(
#             options={
#                 "ignore_links": True,
#                 "ignore_images": True,
#                 "skip_internal_links": True,
#             },
#             content_filter=PruningContentFilter(
#                 threshold=0.80, threshold_type="dynamic", min_word_threshold=0
#             ),
#         ),
#         excluded_tags=[],
#         exclude_social_media_links=True,
#         exclude_external_links=True,
#         page_timeout=60000,
#         delay_before_return_html=2.0,
#     )

#     adaptive_config = AdaptiveConfig(
#         strategy="embedding",
#         embedding_model='openai/text-embedding-3-small',
#         embedding_llm_config = {
#             "provider": 'openai/text-embedding-3-small',
#             "api_token": os.getenv("OPENAI_API_KEY"),
#             "temperature": 0.5,
#         },
#         confidence_threshold = 0.6,
#         # n_query_variations = 12,
#         min_gain_threshold = .001,
#         # embedding_min_confidence_threshold=.4,
#         max_pages = 35,
#         top_k_links = 5,
#         # embedding_min_relative_improvement = 0.2,
        
#     )

#     async with AsyncWebCrawler(config=search_page_async_crawler_config) as crawler:
#         # original_arun = crawler.arun

#         # async def constrained_arun(url: str, **kwargs):
#         #     kwargs['config'] = search_page_async_crawler_config
#         #     return await original_arun(url, **kwargs)
        
#         # crawler.arun = constrained_arun

#         adaptive = AdaptiveCrawler(crawler = crawler, config=adaptive_config)
#         result = await adaptive.digest(
#             start_url = "https://bankingjournal.aba.com/category/newsbytes/",
#             query = query,
#             # config = async_crawler_config,
#         )
#         adaptive.print_stats(detailed=True)

#     print('here!')
#     # print(result)
#     # if result.metrics.get("is_irrelevant", False):
#     #     print("Query is unrelated to content!")
#     #     return "Query is unrelated to content!"
#     # else:
#     print(result.crawled_urls)
#     print(f'Pending URLS: {result.pending_links[:4]}')
#     for page in adaptive.get_relevant_content(top_k=3):
#         print(page['url'])




async def main():
    urls = load_in_manual_sources("data/manual_search.txt")
    website_urls = []
    journal_urls = []
    for url in urls:
        if url.startswith("https://bankingjournal.aba.com"):
            journal_urls.append(url)
        else:
            website_urls.append(url)
    print(f"[blue]Scraping {len(website_urls)} website URLs[/blue]")
    success = await scrape_aba_website(website_urls)
    if success:
        print("[green]Website scraping completed successfully[/green]")
    else:
        print("[red]Website scraping failed[/red]")
    print(f"[blue]Scraping {len(journal_urls)} journal URLs[/blue]")
    success = await scrape_aba_journal(journal_urls)
    if success:
        print("[green]Journal scraping completed successfully[/green]")
    else:
        print("[red]Journal scraping failed[/red]")


if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(adaptive_crawling_aba_journal("latest news on check fraud"))
