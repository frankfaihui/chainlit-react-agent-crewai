import chainlit as cl
import requests
import os
from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model


from crews.brand_research_crew.crew import BrandResearchCrew
from crewai.task import TaskOutput
from dotenv import load_dotenv
load_dotenv()

class BrandResearchInfo(BaseModel):
    company: str = Field(..., description="The company name")
    topic: Optional[str] = Field(None, description="The topic of the research")

# You can provide static information to the graph at runtime, like a user_id or API credentials
# https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#access-config

# Agent functions that will be called as tools
def brand_research_agent(state: Annotated[dict, InjectedState], config: RunnableConfig, info: BrandResearchInfo) -> str:
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

        return f"""Successfully completed the research. Research result: {result.raw}"""
    
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

class GoogleAdsCampaignInfo(BaseModel):
    login_customer_id: Optional[str] = Field(os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"), description="The Google Ads Manager Account ID")

def google_ads_campaign_agent(state: Annotated[dict, InjectedState], config: RunnableConfig) -> str:
    """Manage google ads campaigns, get customers"""
    import requests
    
    login_customer_id = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    
    # Get the token from config
    token = config["configurable"].get("token")
    if not token:
        return "Error: Authorization token not found in config"
    
    # Get request to fetch accessible customers
    url = f"https://my-google-ads-flask-29233987206.us-west1.run.app/google-ads/customers"
    params = {"login_customer_id": login_customer_id}
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise exception for non-2xx responses
        
        # Parse the JSON response
        customer_data = response.json()
        
        # # Format the output
        # result = f"Successfully retrieved accessible customers for manager account ID: {login_customer_id}.\n\nAccessible Customers:\n"
        
        # # Add each customer to the result
        # for customer in customer_data.get("customers", []):
        #     result += f"- Customer ID: {customer.get('id')}, Name: {customer.get('name')}\n"
        
        return customer_data
    
    except requests.exceptions.RequestException as e:
        return f"Error accessing Google Ads API: {str(e)}"
    except ValueError as e:
        return f"Error parsing response from Google Ads API: {str(e)}"


def get_campaigns(state: Annotated[dict, InjectedState], config: RunnableConfig, customer_id: str) -> str:
    """Get campaigns for a specific customer account"""
    import requests
    
    login_customer_id = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
   
    # Get the token from config
    token = config["configurable"].get("token")
    if not token:
        return "Error: Authorization token not found in config"
    
    # Get request to fetch campaigns
    url = f"https://my-google-ads-flask-29233987206.us-west1.run.app/google-ads/customers/{customer_id}/campaigns"
    params = {"login_customer_id": login_customer_id}
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise exception for non-2xx responses
        
        # Parse the JSON response
        campaign_data = response.json()
        
        return campaign_data
    
    except requests.exceptions.RequestException as e:
        return f"Error accessing Google Ads API: {str(e)}"
    except ValueError as e:
        return f"Error parsing response from Google Ads API: {str(e)}"

# https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor-tool-calling
tools = [brand_research_agent, develop_strategy_agent, generate_campaign_brief_agent, 
         validate_campaign_agent, google_ads_campaign_agent, get_campaigns]

def prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    # user_name = config["configurable"].get("user_name")
    system_msg = f"You are a knowledgeable and friendly assistant specialized in marketing. Your job is to help users with questions strictly related to marketing. If users ask about anything else, politely steer the conversation back to marketing topics."
    return [{"role": "system", "content": system_msg}] + state["messages"]

# model = ChatOpenAI(model_name="gpt-4o-mini")

model = init_chat_model(
    "openai:gpt-4o-mini",
    # temperature=0
)
# memory = MemorySaver()
agent_executor = create_react_agent(model, tools, prompt=prompt)

# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     # Fetch the user matching username from your database
#     # and compare the hashed password with the value stored in the database
#     if (username, password) == ("admin", "admin"):
#         return cl.User(
#             identifier="admin", metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None

# just for testing should store in db or redis
tokens = {}

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  # Store user identifier as key with token as value
  tokens[default_user.identifier] = token
  
  return default_user

@cl.on_chat_start
async def on_chat_start():

    app_user = cl.user_session.get("user")
    print(app_user)
    # await cl.Message(f"Hello {app_user.identifier}").send()

@cl.on_message
async def on_message(message: cl.Message):
    final_answer = cl.Message(content="")

    messages = cl.chat_context.to_openai()

    async for step, metadata in agent_executor.astream(
        {"messages": [
            *messages,
        ]},
        config={
            "configurable": {
                "thread_id": cl.context.session.id,
                "token": os.getenv("GOOGLE_ADS_REFRESH_TOKEN")
            }
        },
        stream_mode="messages"
    ):
        await final_answer.stream_token(step.content)
    
    await final_answer.send()

@cl.on_chat_resume
async def on_chat_resume(thread):
    pass