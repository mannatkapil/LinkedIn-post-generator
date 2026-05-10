import asyncio
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import settings

llm = init_chat_model(
    model="openai/gpt-oss-120b", model_provider="groq", api_key=settings.GROQ_API_KEY
)


class LinkdinState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def content_stratergy_node(state: LinkdinState) -> LinkdinState:
    user_message = state["messages"]

    prompt = "you are an agent where user provide you a topic and you need to stratergise content for linkidin post"
    system_message = SystemMessage(prompt)

    response = await llm.ainvoke([system_message] + user_message)
    return {"messages": [response]}


async def post_generator_node(state: LinkdinState) -> LinkdinState:
    user_message = state["messages"]

    prompt = "you are a linkdin post generator where you need to create a linkdin post"
    system_message = SystemMessage(prompt)

    response = await llm.ainvoke([system_message] + user_message)
    return {"messages": [response]}


async def tone_improver_node(state: LinkdinState) -> LinkdinState:
    user_message = state["messages"]

    prompt = "you are an agent who will imporve the tone of the post for linkdin"
    system_message = SystemMessage(prompt)

    response = await llm.ainvoke([system_message] + user_message)
    return {"messages": response}


async def hashtag_generator_node(state: LinkdinState) -> LinkdinState:
    user_message = state["messages"]

    prompt = (
        "you are an agent who add hastan to the post of linkdin according to the topic"
    )
    system_message = SystemMessage(prompt)

    response = await llm.ainvoke([system_message] + user_message)
    return {"messages": response}


memory = InMemorySaver()
graph = StateGraph(state_schema=LinkdinState)

graph.add_node("content stratergy", content_stratergy_node)
graph.add_node("post generator", post_generator_node)
graph.add_node("tone improver", tone_improver_node)
graph.add_node("hashtag generator", hashtag_generator_node)


graph.add_edge(START, "content stratergy")
graph.add_edge("content stratergy", "post generator")
graph.add_edge("post generator", "tone improver")
graph.add_edge("tone improver", "hashtag generator")
graph.add_edge("hashtag generator", END)

workflow = graph.compile(checkpointer=memory)


async def run_workflow():
    config = {"configurable": {"thread_id": "abc_123"}}
    while True:
        user_input = input("Enter your message:")

        if user_input.lower() == "exit":
            break

        initial_state: LinkdinState = {"messages": [HumanMessage(content=user_input)]}

        final_state: LinkdinState = await workflow.ainvoke(initial_state, config=config)

        print(final_state["messages"][-1].content)


asyncio.run(run_workflow())  