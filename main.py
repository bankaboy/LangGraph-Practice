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



# build the nodes that are in the route

# takes a state and decides if the request requires a logical or emotional response
def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier) # will only output that follows the pydantic model

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions"""
        },
        { "role": "user", "content": last_message.content}
    ])

    return {"message_type": result.message_type}


# define the logical agent, gives a logical response
def logical_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role" : "system",
            "content" : """You are a purely logical assistant. Focus only on facts and information.
                        Provide clear, concise answers based on logic and evidence.
                        Do not address emotions or provide emotional support.
                        Be direct and straightforward in your responses."""
        }, 
        {
            "role" : "user",
            "content" : last_message.content
        }
    ]

    reply = llm.invoke(messages)
    return {"messages" : [{"role": "assistant", "content" : reply.content}]}


# define the emotional agent, gives an emotional response
def emotional_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role" : "system",
            "content" : """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
        }, 
        {
            "role" : "user",
            "content" : last_message.content
        }
    ]

    reply = llm.invoke(messages)
    return {"messages" : [{"role": "assistant", "content" : reply.content}]}


# select which agent to be routed to
def router(state: State):
    message_type = state.get("message_type", "logical")

    if message_type == "emotional":
        return {"next": "emotional"}
    
    return {"next": "logical"}




# define a graph builder that will work using this state
graph_builder = StateGraph(State)

# always need start and end node
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("emotional", emotional_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

# add the conditional edges
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "emotional" : "emotional",
        "logical" : "logical" 
    }
)

graph_builder.add_edge("emotional", END)
graph_builder.add_edge("logical", END)

# run the graph
graph = graph_builder.compile()


# run the graph
def run_chatbot():

    # initial state
    state = {
        "messages" : [],
        "message_type" : None
    }

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        # take the user message and add it to the current state
        state["messages"] = state.get("messages" , []) + [
            {"role": "user", "content": user_input}
        ]

        # invoke the graph using the current state
        state = graph.invoke(state)

        # if there were any messages print them out
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


# start the conversation
if __name__ == "__main__":
    run_chatbot()