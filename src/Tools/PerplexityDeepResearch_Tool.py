import os
import asyncio
from agents import Agent, FileSearchTool, Runner, WebSearchTool, trace, set_trace_processors, OpenAIChatCompletionsModel
from openai import AsyncOpenAI


INSTRUCTIONS = (
    "You are an an expert business analyst providing detailed research on the provided topic."
)
MODEL_NAME = "sonar-deep-research"

client = AsyncOpenAI(api_key=os.getenv("PERPLEXITY_KEY"), base_url=os.getenv("PERPLEXITY_BASE_URL"))

PerplexityDeepResearch_Tool = Agent(
        name="PerplexityDeepResearch_Tool",
        instructions=INSTRUCTIONS,
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    )

