from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.runnables.config import RunnableConfig


def add(a: int, b: int, config: RunnableConfig) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """

    return a + b

def multiply(a: int, b: int, config: RunnableConfig) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """

    return a * b

def divide(a: int, b: int, config: RunnableConfig) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    
    return a / b


tools = [add, multiply, divide]

# Define LLM with bound tools
llm = ChatOllama(model="llama3.2:3b-instruct-fp16")
llm_with_tools = llm.bind_tools(tools)

# Node
def assistant(state: MessagesState, config: RunnableConfig):
   model_name = config["configurable"].get("model", "llama3.1")
   llm = ChatOllama(model=model_name)
   llm_with_tools = llm.bind_tools(tools)
   return {"messages": [llm_with_tools.invoke(state["messages"],config=config)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()

class MathAgent:
    @staticmethod
    def getGraph() -> StateGraph:
        return graph
    



