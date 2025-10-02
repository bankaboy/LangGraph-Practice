# get the necessary imports
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# load the keys/ids deom from the env file
load_dotenv()

# initiate the llm
llm = init_chat_model(
    "anthropic:claude-3-5-haiku-20241022"
)

# structured output parser
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field (
        ...,
        description = "Classify if the message requires an emotional or logical response"
    )


# setup the state of the graph, define what state agents will have access to
class State(TypedDict):
    messages: Annotated[list, add_messages] # messages will be of type list and can be modified with add_messages
    message_type: str | None # decides the kind of request it is to route it to the correct agent


# define a graph builder that will work using this state
graph_builder = StateGraph(State)

# build the nodes that are in the route

# takes a state and returns a modified state or next state in the graph
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# register the node in the graph
graph_builder.add_node("chatbot", chatbot)

# always need start and end node
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# run the graph
graph = graph_builder.compile()

