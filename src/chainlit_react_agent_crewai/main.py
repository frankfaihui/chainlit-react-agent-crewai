import chainlit as cl

from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState

# Pydantic model for company information
class CompanyInfo(BaseModel):
    name: str = Field(..., description="The name of the company")
    industry: str = Field(..., description="The industry the company operates in")
    target_audience: Optional[str] = Field(None, description="The target audience or customer segment")
    competitors: Optional[List[str]] = Field(None, description="List of main competitors")
    unique_selling_points: Optional[List[str]] = Field(None, description="Key unique selling points or value propositions")
    years_in_business: Optional[int] = Field(None, description="Number of years the company has been in business")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "TechAI",
                "industry": "AI Software",
                "target_audience": "Small to medium businesses",
                "competitors": ["AICompany", "SmartTools Inc."],
                "unique_selling_points": ["User-friendly interface", "Affordable pricing"],
                "years_in_business": 5
            }
        }

model = ChatOpenAI(model_name="gpt-4o-mini")

# Agent functions that will be called as tools
def branding_research_agent(state: Annotated[dict, InjectedState], company: CompanyInfo) -> str:
    """Do a branding research for a given company with detailed information."""
    # You can use state to access the full agent state if needed
    
    # Now we have access to detailed structured information about the company
    research_results = f"Research Results for {company.name}:\n"
    research_results += f"- Industry: {company.industry}\n"
    
    if company.target_audience:
        research_results += f"- Target Audience: {company.target_audience}\n"
    
    if company.competitors:
        research_results += f"- Main Competitors: {', '.join(company.competitors)}\n"
    
    if company.unique_selling_points:
        research_results += f"- Unique Selling Points: {', '.join(company.unique_selling_points)}\n"
    
    if company.years_in_business:
        research_results += f"- Company Maturity: {company.years_in_business} years in business\n"
    
    research_results += "\nRecommended Branding Approach: Based on the information provided, focus on highlighting the company's unique selling points in marketing materials and differentiate from competitors through targeted messaging for the specific audience."
    
    return research_results

def develop_strategy_agent(state: Annotated[dict, InjectedState], target_audience: str) -> str:
    """Develop a marketing strategy based on research findings and campaign objectives."""
    return f"Marketing Strategy for {target_audience}: Focus on digital channels with personalized messaging highlighting AI-powered solutions. Recommend a 3-month campaign with weekly social media posts and monthly webinars."

def generate_campaign_brief_agent(state: Annotated[dict, InjectedState], strategy: str) -> str:
    """Generate a detailed campaign brief aligning with the strategy and objectives."""
    return f"Campaign Brief based on '{strategy}': 12-week campaign timeline with creative assets for LinkedIn, Twitter, and email sequences. Weekly content calendar with key messaging points and call-to-action recommendations."

def validate_campaign_agent(state: Annotated[dict, InjectedState], brief: str) -> str:
    """Facilitate human review and approval of the campaign brief."""
    return f"Validation Report for '{brief}': Campaign brief approved with minor adjustments to tone. Ensure all creative assets maintain consistent branding. Ready for execution upon final stakeholder sign-off."

def execute_campaign_agent(state: Annotated[dict, InjectedState], channels: str) -> str:
    """Implement the approved campaign across selected channels."""
    return f"Campaign Execution Report for channels '{channels}': Campaign launched successfully across specified channels. Initial metrics show 15% engagement rate. Weekly performance reports scheduled for stakeholder review."

# https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor-tool-calling
tools = [branding_research_agent, develop_strategy_agent, generate_campaign_brief_agent, 
         validate_campaign_agent, execute_campaign_agent]

def prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    # user_name = config["configurable"].get("user_name")
    system_msg = f"You are a knowledgeable and friendly assistant specialized in marketing. Your job is to help users with questions strictly related to marketing. If users ask about anything else, politely steer the conversation back to marketing topics."
    return [{"role": "system", "content": system_msg}] + state["messages"]

memory = MemorySaver()
agent_executor = create_react_agent(model, tools, prompt=prompt, checkpointer=memory)

@cl.on_message
async def on_message(message: cl.Message):
    final_answer = cl.Message(content="")

    async for step, metadata in agent_executor.astream(
        {"messages": [HumanMessage(content=message.content)]},
        config={"configurable": {"thread_id": cl.context.session.id}},
        stream_mode="messages"
    ):
        await final_answer.stream_token(step.content)
    
    await final_answer.send()

