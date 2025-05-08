import chainlit as cl

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState

@tool
def branding_research(company: str) -> str:
    """Do a branding research for a given company."""
    return f"Research Results for {company}: This is a company that sells AI-powered marketing tools."

@tool
def develop_strategy(target_audience: str) -> str:
    """Develop a marketing strategy based on research findings and campaign objectives."""
    return f"Marketing Strategy for {target_audience}: Focus on digital channels with personalized messaging highlighting AI-powered solutions. Recommend a 3-month campaign with weekly social media posts and monthly webinars."

@tool
def generate_campaign_brief(strategy: str) -> str:
    """Generate a detailed campaign brief aligning with the strategy and objectives."""
    return f"Campaign Brief based on '{strategy}': 12-week campaign timeline with creative assets for LinkedIn, Twitter, and email sequences. Weekly content calendar with key messaging points and call-to-action recommendations."

@tool
def validate_campaign(brief: str) -> str:
    """Facilitate human review and approval of the campaign brief."""
    return f"Validation Report for '{brief}': Campaign brief approved with minor adjustments to tone. Ensure all creative assets maintain consistent branding. Ready for execution upon final stakeholder sign-off."

@tool
def execute_campaign(channels: str) -> str:
    """Implement the approved campaign across selected channels."""
    return f"Campaign Execution Report for channels '{channels}': Campaign launched successfully across specified channels. Initial metrics show 15% engagement rate. Weekly performance reports scheduled for stakeholder review."

model = ChatOpenAI(model_name="gpt-4o-mini")
tools = [branding_research, develop_strategy, generate_campaign_brief, validate_campaign, execute_campaign]

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
    response = await agent_executor.ainvoke(
        {"messages": [HumanMessage(content=message.content)]},
        config={"configurable": {"thread_id": cl.user_session.get("session_id")}}
    )
    
    result = response["messages"][-1].content
    
    await cl.Message(content=result).send()
