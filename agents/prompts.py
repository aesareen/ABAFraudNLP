tools: str = """
    - BM25: A tool that allows you to query a BM25 Index. This should be used for all keyword-based searches, while semantic searches should be done with the PGVectors Server.
"""

prompt_templates: dict[str, dict[str, str]] = {
    "summarization_agent": {
        "system_prompt": (f"You are a skilled assistant that summarizes articles from the American Bar Association's website and is able to extract prominent keywords from these articles, generate comprehensive, detailed, yet succinct summaries of their information, and reference additional sources as needed. Use have access to the following tools to accomplish this: {tools}"),
        "user_prompt": "Summarize the following article: {article}",
    }
}


