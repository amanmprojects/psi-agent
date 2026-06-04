import os

import dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ReasoningContentBlock
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from tools.bash import bash

dotenv.load_dotenv()


def get_api_key() -> str:
    return os.environ.get("BEDROCK_API_KEY", "")


checkpointer = InMemorySaver()

agent = create_agent(
    model=ChatOpenAI(
        api_key=get_api_key,
        base_url="https://bedrock-mantle.ap-south-1.api.aws/v1",
        model="moonshotai.kimi-k2.5",
    ),
    tools=[bash],
    system_prompt="You are a helpful assistant. You can use the bash tool to run bash commands.",
    checkpointer=checkpointer,
)
