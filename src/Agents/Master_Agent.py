import os
from agents import Runner, agent, trace, set_trace_processors, Agent, ModelSettings
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor
import asyncio
from typing import Literal, List, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field, conlist
from IPython.display import display, Markdown, HTML, Image
load_dotenv()
# Import Necessary Libraries
from src.Tools.Agentic_Calculator_Tool import Agentic_Calculator_Tool
from src.Tools.PerplexitySECSonarPro_Tool import PerplexitySECSonarPro_Tool
from src.Tools.Search_Tool import Search_Tool
from src.Tools.OpenAIDeepResearch_Tool import OpenAIDeepResearch_Tool
from src.Tools.Critic_Tool import Critic_Tool

ANALYSIS_QA_PROMPT = '''
System = You are a Generative-AI analyst who quantifies how GenAI REDUCES labor demand in banking processes.

# Scenarios
1. High Scenario assumes that Generative AI will severely impact the process.
2. Medium Scenario assumes that Generative AI will have moderate process on the scenarios.
3. Low Scenario assumes that Generative AI will have minimal impact on the scenarios.

# Output Method
1. For each scenario, return a **vector of five NEGATIVE year-by-year % changes** (Y1–Y5) *and* the **5-year cumulative % change** (also NEGATIVE).  
   • *Rule: every element must be ≤ 0. Positive numbers are disallowed.*  
   • Use “−0 %” only if the model can justify true stasis.

# Instruction
1. Parse the user’s baseline labor metrics (FTEs, $ cost, etc.).
2. Use <PerplexitySECSonarPro_Tool> to map the banking process and extract GenAI disclosures/benchmarks from SEC filings.
3. Use <Search_Tool> for supplementary evidence (industry adoption rates, productivity studies).
4. Use <Agentic_Calculator_Tool> to estimate realistic automation rates and resulting labor-savings trajectory.
5. Compute year-by-year and cumulative labor change vectors **ensuring all % values are negative or zero**.
6. Run the draft answer through <Critic_Tool>; **# Critic integration: explicitly incorporate every piece of feedback returned by <Critic_Tool> before finalizing**. Iterate until hallucination risk = “Low”.

# Tool Description
1. <PerplexitySECSonarPro_Tool>: Provides research capabilities for Security Exchange Commission (SEC) fillings.
2. <Agentic_Calculator_Tool>: Provides a tool for evaluating the appropriateness of Generative AI for a specific business activity.
3. <Search_Tool>: Provides a tool for searching for information on a given topic.
4. <Critic_Tool>: Provides a tool for assessing the quality of the final answer.
'''

class Agentic_Calculator_Tool_Output(BaseModel):
    high_scenario: conlist(float, min_length=5, max_length=5)
    "High-scenario cumulative vector percentage change"
    medium_scenario: conlist(float, min_length=5, max_length=5)
    "Medium-scenario cumulative vector percentage change"
    low_scenario: conlist(float, min_length=5, max_length=5)
    "Low-scenario cumulative vector percentage change"
    high_scenario_reasoning: str
    "Provide a detailed reasoning chain of thought for the high-scenario cumulative vector percentage change"
    medium_scenario_reasoning: str
    "Provide a detailed reasoning chain of thought for the medium-scenario cumulative vector percentage change"
    low_scenario_reasoning: str
    "Provide a detailed reasoning chain of thought for the low-scenario cumulative vector percentage change"
    hallucination_score: Literal["Low", "Medium", "High"]
    "Provide a score of the response quality. Low = Low hallucination risk, Medium = Medium hallucination risk, High = High hallucination risk"

q_a_agent = Agent(
    name="q_a_agent",
    instructions=ANALYSIS_QA_PROMPT,
    output_type=Agentic_Calculator_Tool_Output,
    model=os.getenv("LLM_MODEL"),
    model_settings=ModelSettings(reasoning={"effort": "high"}),
    tools=[
            PerplexitySECSonarPro_Tool.as_tool(
                tool_name="PerplexitySECSonarPro_Tool",
                tool_description="PerplexitySECSonarPro_Tool provides research capabilities for Security Exchange Commission (SEC) fillings."
    ),
           Agentic_Calculator_Tool.as_tool(
               tool_name="Agentic_Calculator_Tool",
               tool_description="Agentic_Calculator_Tool provides a tool for evaluating the appropriateness of Generative AI for a specific business activity."
           ),
           Search_Tool.as_tool(
               tool_name="Search_Tool",
               tool_description="Search_Tool provides a tool for searching for information on a given topic."
           ),
            Critic_Tool.as_tool(
                tool_name="Critic_Tool",
                tool_description="Critic_Tool provides a tool for assessing the quality of the final answer."
            )
    ],
)