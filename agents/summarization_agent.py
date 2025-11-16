from litellm import Router
from pocket_agent import PocketAgent, AgentConfig
import logging
import os
import asyncio
from logging import getLogger, basicConfig, INFO
from prompts import prompt_templates
from typing import Any
from dotenv import load_dotenv

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

logging.basicConfig(
    filename=os.path.join(project_root, "logs/summarization_agent.log"),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = getLogger(__name__)

load_dotenv(dotenv_path=os.path.join(project_root, "config/.env"), override=True)

router_config: dict[str, list[dict[str, Any]]] = {
    "models": [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "tpm": 500000,  # tokens per minute (I am not on a very high tier)
                "rpm": 500,  # requsts per minute
            },
        }
    ]
}

ROUTER: Router = Router(model_list=router_config["models"])


def initialize_summarization_agent():
    """Initialize the summarization agent"""
    summarization_agent_mcp_config: dict[str, dict[str, str]] = {
        "mcpServers": {
            "BM25_Index": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "bm25.py"],
                # Theoretically, our files could be anywhere, so this is a consistent way to ensure that we are accessing the BM25 Script
                "cwd": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "scripts"
                ),
            }
        }
    }

    config: AgentConfig = AgentConfig(
        llm_model="gpt-5-mini",
        system_prompt=prompt_templates["summarization_agent"]["system_prompt"],
        agent_id="summarization_agent",
        allow_images=False,
        messages=[],
    )

    summarization_agent: PocketAgent = PocketAgent(
        agent_config=config,
        mcp_config=summarization_agent_mcp_config,
        router=ROUTER,
        logger=LOGGER,
    )

    LOGGER.info("Summarization agent initialized successfully")
    return summarization_agent


async def main():
    summarization_agent = initialize_summarization_agent()

    response = await summarization_agent.run("Find me the top articles on check fraud and give me just the top keywords from each")

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
