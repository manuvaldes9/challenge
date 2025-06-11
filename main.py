import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("The variable OPENAI_API_KEY is not defined")
    
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from rag_chain import create_rag_chain, build_retriever, State

app = FastAPI()

retriever = build_retriever()
rag_chain = create_rag_chain(retriever)

answer_runnable = RunnableLambda(
    lambda state: rag_chain.invoke(state)["answer"]
).with_types(input_type=State)

add_routes(
    app,
    answer_runnable,
    path="/chat",
    playground_type="default",
)