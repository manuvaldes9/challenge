from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.ingestion_processing  import build_retriever


def prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a useful assistant that answers questions about Promptior"),
        ("system", "You ONLY can answer questions with the context you have"),
        ("system", "Use the context to respond the question of the user in a clear and concise way"),
        ("system", "If you don't have the answer in your context answer 'I don't have that information'"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])


# Construct RAG chain
def create_rag_chain(retriever=build_retriever()):
    try:
        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt()},
            return_source_documents=False,
            input_key="question"
        )
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        raise e
