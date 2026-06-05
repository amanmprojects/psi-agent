import operator
import os
from typing import Literal

from langchain.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from tools import tools, tools_by_name


def get_api_key() -> str:
    """Get the API key from the environment variable BEDROCK_API_KEY."""
    return os.environ.get("BEDROCK_API_KEY", "")


model = ChatOpenAI(
    api_key=get_api_key,
    base_url="https://bedrock-mantle.ap-south-1.api.aws/v1",
    model="zai.glm-4.7" or "moonshotai.kimi-k2.5",
).bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    chat_id: int | str


def set_chat_id(state: AgentState):
    pass


# Default node that runs on every telegram message
def llm_call(state: AgentState):
    """LLM handles the message from the user"""
    return {
        "messages": [
            model.invoke(
                [SystemMessage(content="You are a helpful assistant.")]
                + state["messages"]
            )
        ]
    }


# Tool node
from langchain.messages import ToolMessage


def tool_node(state: AgentState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build agent

# Build workflow
agent_builder = StateGraph(AgentState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile(checkpointer=InMemorySaver())
print("Compiled Agent")
