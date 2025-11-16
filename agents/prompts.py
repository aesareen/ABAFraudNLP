from pydantic import BaseModel, Field

class ListOfKeywords(BaseModel):
    keywords: list[str] = Field(description="A list of keywords extracted from the article")

class KeywordList(BaseModel):
    article_name: str = Field(description="The name of the article")
    keywords: ListOfKeywords


tools: str = """
    - BM25: A tool that allows you to query a BM25 Index. This should be used for all keyword-based searches, while semantic searches should be done with the PGVectors Server.
"""

prompt_templates: dict[str, dict[str, str]] = {
    "summarization_agent": {
        "system_prompt": {
            "prompt": f"You are a skilled assistant that summarizes articles from the American Bar Association's website and is able to extract prominent keywords from these articles, generate comprehensive, detailed, yet succinct summaries of their information, and reference additional sources as needed. Use have access to the following tools to accomplish this: {tools}",
            "response_format": None
        },
        "extract_keywords_from_query": {
            "prompt": "Find the relevant articles based upon the following query and then extract the prominent keywords from each article. Ensure your keywords are just the most high-level, overall keywords that would likely apply to multiple articles. Do not include specific keywords that may be only true for a single article or a small subset of articles.: {query}",
            "response_format": KeywordList
        },
        "extract_keywords_from_article": {
            "prompt": "Extract the prominent keywords from the following article. Ensure your keywords are just the most high-level, overall keywords that would likely apply to multiple articles. Do not include specific keywords that may be only true for a single article or a small subset of articles. Limit your keywords to 3-5 primary keywords and prioritize previous keywords if they are applicable to the article. Do not use any tools for this. \n Article: {article}",
            "response_format": KeywordList
        }
    }
}


