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

from crews.brand_research_crew.crew import BrandResearchCrew
from crewai.task import TaskOutput

class BrandResearchInfo(BaseModel):
    company: str = Field(..., description="The company name")
    topic: str = Field(..., description="The topic of the research")

# Agent functions that will be called as tools
def brand_research_agent(state: Annotated[dict, InjectedState], info: BrandResearchInfo) -> str:
    """Do a branding research for a given company with detailed information."""

    # Create a message to track progress
    progress_message = cl.run_sync(
        cl.Message(content="Starting brand research...").send()
    )

    try:
        # Create a callback function to update progress
        def on_task_callback(task_output: TaskOutput):
            step_agent = task_output.agent if task_output.agent else "Agent"
            step_task = task_output.name if task_output.name else "task"
            
            cl.run_sync(
                cl.Message(content=f"Agent: {step_agent} is working on task: {step_task}...").send()
            )

        # Create the crew with callback
        crew = BrandResearchCrew().crew()
        crew.task_callback = on_task_callback

        # Run the crew
        inputs = info.model_dump()
        result = crew.kickoff(inputs)
        

        cl.run_sync(
            cl.Message(content=f"Research completed!").send()
        )

        return result.raw
    
    except Exception as e:

        cl.run_sync(
            cl.Message(content=f"Error: {e}").send()
        )

        return f"Error: {e}"

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
tools = [brand_research_agent, develop_strategy_agent, generate_campaign_brief_agent, 
         validate_campaign_agent, execute_campaign_agent]

def prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    # user_name = config["configurable"].get("user_name")
    system_msg = f"You are a knowledgeable and friendly assistant specialized in marketing. Your job is to help users with questions strictly related to marketing. If users ask about anything else, politely steer the conversation back to marketing topics."
    return [{"role": "system", "content": system_msg}] + state["messages"]

model = ChatOpenAI(model_name="gpt-4o-mini")
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

