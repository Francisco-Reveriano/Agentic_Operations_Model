import asyncio
import os
from agents import Agent, FileSearchTool, Runner, trace

Knowledge_Base_Search_Tool = Agent(
        name="Knowledge_Base_Search_Tool",
        instructions="You are a helpful knowledge base search agent. Answer the user's question based on the provided knowledge base.",
        tools=[
            FileSearchTool(
                max_num_results=10,
                vector_store_ids=[os.getenv("ASSISTANT_VECTOR_KEY")],
                include_search_results=True,
            )
        ],
    )