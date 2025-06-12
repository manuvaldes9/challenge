import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("The variable OPENAI_API_KEY is not defined")
    
from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from rag_chain import create_rag_chain, build_retriever
from typing import Any

class ChatInput(BaseModel):
    question: str

app = FastAPI()

retriever = build_retriever()
rag_chain = create_rag_chain(retriever)


# Función que procesa el chat manteniendo la estructura de Pydantic
def chat_wrapper(inputs: Any) -> str:
    # Extraer la pregunta del input
    question = inputs.get("question") if isinstance(inputs, dict) else str(inputs)

    # Crear state y ejecutar RAG
    state = {"question": question}
    result = rag_chain.invoke(state)
    return result["answer"]


answer_runnable = RunnableLambda(
    chat_wrapper,
    name="rag_chatbot"
)

add_routes(
    app,
    answer_runnable,
    path="/chat",
    input_type=ChatInput,
    output_type=str,
    playground_type="default",
)