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

model = ChatOpenAI(model_name="gpt-4o-mini")
tools = [branding_research]

def prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    # user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant who can answer marketing questions ONLY. Always direct the user to ask questions about the marketing plan."
    return [{"role": "system", "content": system_msg}] + state["messages"]

memory = MemorySaver()
agent_executor = create_react_agent(model, tools, prompt=prompt, checkpointer=memory)

@cl.on_message
async def on_message(message: cl.Message):
    response = agent_executor.invoke(
        {"messages": [HumanMessage(content=message.content)]},
        config={"configurable": {"thread_id": cl.user_session.get("session_id")}}
    )
    
    result = response["messages"][-1].content
    
    await cl.Message(content=result).send()
