from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict, List
from langgraph.graph import END, StateGraph

# Estado tipado para LangChain
class State(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: str

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("Use the following context to respond the question of the user in a clear concise way"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


def build_retriever():
    # Carga de documentos y preparacion
    loader = TextLoader("data/promtior_content.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    
    # Embeddings y vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

def create_rag_chain(retriever):
    # Construir cadena RAG    
    def retrieve(state: State):
        retrieved_docs = retriever.similarity_search(state["question"])
        return {"question": state["question"], "context": retrieved_docs}
    
    
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.set_finish_point("generate")
    graph = builder.compile()

    return graph