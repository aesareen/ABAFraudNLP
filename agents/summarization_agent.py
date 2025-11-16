from litellm import Router
from pocket_agent import PocketAgent, AgentConfig
import logging
import os
import asyncio
from logging import getLogger, basicConfig, INFO
from prompts import prompt_templates, KeywordList, ListOfKeywords
from scripts.upload_to_supabase import upload_keywords_to_supabase
from typing import Any
from dotenv import load_dotenv
from rich import print
import copy
import glob
import json

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

logging.basicConfig(
    filename=os.path.join(project_root, "logs/summarization_agent.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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


class SummarizationAgent(PocketAgent):
    def _prepare_schema_for_openai(self, schema_dict: dict) -> dict:
        """
        Recursively add 'additionalProperties': false to all object types in the schema.
        This is required for OpenAI's strict mode structured outputs.

        Args:
            schema_dict: The JSON schema dictionary

        Returns:
            Modified schema with additionalProperties set to false
        """

        schema = copy.deepcopy(schema_dict)

        def add_additional_properties(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object":
                    obj["additionalProperties"] = False

                # Recursively process all nested dictionaries
                for key, value in obj.items():
                    if isinstance(value, dict):
                        add_additional_properties(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                add_additional_properties(item)
            return obj

        return add_additional_properties(schema)

    async def run(self, user_input: str, schema=None, schema_name="", **kwargs):
        """
        Given a specified schema, run the agent, run all the tools, and enforce the schema on the final response.

        Args:
            user_input: The user query
            schema: Pydantic model class to enforce on final response
            schema_name: Name for the schema (used in response_format)
        """
        if schema is None:
            raise ValueError("Schema is required")
        if schema_name is None or schema_name == "":
            raise ValueError("Schema name is required")

        # Store the schema class for validation later
        schema_class = schema

        # Generate JSON schema and prepare it for OpenAI strict mode
        schema_dict = schema_class.model_json_schema()
        openai_compatible_schema = self._prepare_schema_for_openai(schema_dict)

        await self.add_user_message(user_input)

        step_result = await self.step()

        # Continue until no more tool calls
        while step_result.llm_message.tool_calls is not None:
            step_result = await self.step()

        # Phase 2: If the last response doesn't match our schema, request a final formatted response
        # Try to validate the current response
        try:
            response_content = step_result.llm_message.content
            validated_response = schema_class.model_validate_json(response_content)
            return validated_response
        except Exception as e:
            # Response doesn't match schema, request formatted version
            self.logger.debug(
                f"Response validation failed: {e}. Requesting formatted response."
            )

            await self.add_user_message(
                f"Please format your findings as a JSON object matching the {schema_name} schema."
            )

            # Final step with schema enforcement
            final_step = await self.step(
                tool_choice="none",  # Disable tools for final response
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": openai_compatible_schema,
                        "strict": True,
                    },
                },
            )

            response_content = final_step.llm_message.content
            validated_response = schema_class.model_validate_json(response_content)
            return validated_response

    async def execute_user_input_loop(self, schema=None, schema_name=None):
        """
        Interactive loop for keyword extraction.

        Args:
            schema: JSON schema dict to enforce on responses (defaults to KeywordList)
            schema_name: Name for the schema
        """
        while True:
            user_input = input(
                "Enter a query to extract keywords from ('quit' to stop): "
            )
            if user_input.lower() == "quit":
                break

            formatted_prompt = prompt_templates["summarization_agent"][
                "extract_keywords_from_query"
            ]["prompt"].format(query=user_input)

            response_json = await self.run(
                formatted_prompt, schema=schema, schema_name=schema_name
            )

            print(f"\nExtracted Keywords:\n{response_json.model_dump_json(indent=2)}\n")

    async def generate_article_keywords(self, article: str, previous_extracted_keywords: list[str] = []):
        if len(previous_extracted_keywords):
            await self.add_user_message(f"Context: Previously extracted keywords: {', '.join(previous_extracted_keywords)}")
        
        formatted_prompt = prompt_templates["summarization_agent"]["extract_keywords_from_article"]["prompt"].format(article=article)
        
        response_json = await self.run(formatted_prompt, schema=ListOfKeywords, schema_name="ListOfKeywords")
        return response_json.keywords


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
        system_prompt=prompt_templates["summarization_agent"]["system_prompt"][
            "prompt"
        ],
        agent_id="summarization_agent",
        allow_images=False,
        messages=[],
    )

    summarization_agent: SummarizationAgent = SummarizationAgent(
        agent_config=config,
        mcp_config=summarization_agent_mcp_config,
        router=ROUTER,
        logger=LOGGER,
    )

    LOGGER.debug("Summarization agent initialized successfully")
    return summarization_agent


async def main():
    summarization_agent = initialize_summarization_agent()

    article_files = glob.glob(os.path.join(project_root, "data/scraped_json_results/*.json"))
    keywords_dict = {}
    keywords_list = []
    for article_file in article_files:
        with open(article_file, "r") as f:
            article_data = json.load(f)
            article_name = article_data[0]["article_name"]
            article_content = article_data[0]["raw_content"]
            response = await summarization_agent.generate_article_keywords(article_content, list(set(keywords_list)))
            keywords_dict[article_name] = response 
            keywords_list.extend(response)

    for article_name, keywords in keywords_dict.items():
        upload_keywords_to_supabase(article_name, keywords)
        print(f"[green]Successfully uploaded keywords for {article_name}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
