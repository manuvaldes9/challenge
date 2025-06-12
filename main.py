import os
import openai
from langchain_core.runnables import RunnableLambda

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("The variable OPENAI_API_KEY is not defined")

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from rag_chain import create_rag_chain, build_retriever


class ChatInput(BaseModel):
    question: str = Field(
        "",
        description="Ask about Promtior"
    )


app = FastAPI()

retriever = build_retriever()
rag_chain = create_rag_chain(retriever)

answer_runnable = RunnableLambda(
    lambda inp: rag_chain.invoke(inp)["result"]
).with_types(input_type=ChatInput)

add_routes(
    app,
    answer_runnable,
    path="/chat",
    playground_type="default",
    include_callback_events=False,
    enable_feedback_endpoint=False,
    enable_public_trace_link_endpoint=False,
)
