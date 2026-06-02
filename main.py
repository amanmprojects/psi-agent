import os

import dotenv
from langchain.agents import create_agent
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
        model="zai.glm-5",
    ),
    tools=[bash],
    system_prompt="You are a helpful assistant. You can use the bash tool to run bash commands.",
    checkpointer=checkpointer,
)


def main():
    print("Welcome to Psi Agent")
    while True:
        user_input = input("> ")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "abcd"}},
        )
        print(result["messages"][-1].content_blocks)


if __name__ == "__main__":
    main()
