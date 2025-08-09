import os
import asyncio
from agents import Agent, Runner, WebSearchTool, RunConfig, set_default_openai_client, HostedMCPTool
from agents.model_settings import ModelSettings
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from typing import List, Optional, Union
from agents.tool import function_tool
import requests

@function_tool
def query_perplexity_sec_sonar_pro_tool(user_input: str, search_after_date_filter:str="1/1/2023") -> str:
    """
    Queries the Perplexity SEC Sonar-Pro API tool to retrieve data based on the
    provided user input and optional date filter. This function performs a POST
    request to the Perplexity API endpoint, sending necessary headers and payload
    that include user input, search mode, and filtering parameters.

    :param user_input: A string containing the message or query from the user.
    :param search_after_date_filter: A string specifying the date filter in the format
        "MM/DD/YYYY". Defaults to "1/1/2023".
    :return: The response returned as a JSON object from the Perplexity SEC Sonar-Pro API.
    """
    url = os.getenv("PERPLEXITY_BASE_URL") + "/chat/completions"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {os.getenv('PERPLEXITY_KEY')}",
        "content-type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": user_input}],
        "stream": False,
        "search_mode": "sec", # Search Specifically the SEC
        "search_after_date_filter": search_after_date_filter,
    }

    response = requests.post(url, headers=headers, json=payload)

    return response.json()

# 3. Create Instruction Prompt
PERPLEXITY_SEC_SONAR_PRO = "You are an expert financial analyst performing deep empirical research on Security Exchange Commission (SEC) fillings."

# 4. Create your Agent with the wrapped tool
PerplexitySECSonarPro_Tool = Agent(
    name="PerplexitySECSonarPro_Tool",
    instructions=PERPLEXITY_SEC_SONAR_PRO,
    tools=[query_perplexity_sec_sonar_pro_tool],
)