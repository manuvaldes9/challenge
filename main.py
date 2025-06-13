import os
import openai
from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from app.rag_chain import create_rag_chain

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("The variable OPENAI_API_KEY is not defined")

class ChatInput(BaseModel):
    question: str = Field(
        "",
        description="Ask about Promtior"
    )

app = FastAPI(
    title="RAG chain app for Promtior",
    version="1.0",
    description="A simple RAG chain app",
)

rag_chain = create_rag_chain()

answer_runnable = RunnableLambda(
    lambda inp: rag_chain.invoke(inp)["result"]
).with_types(input_type=ChatInput)

add_routes(
    app,
    answer_runnable,
    path="/chat",
    playground_type="default",
)
