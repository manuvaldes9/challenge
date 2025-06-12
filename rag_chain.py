from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

def _prompt():
    return ChatPromptTemplate.from_messages([
        ("You are a useful Promtior assistant"),
        ("You only can answer questions with the context you have"),
        ("Use the context to respond the question of the user in a clear and concise way"),
        ("If you don't have the answer in your context answer 'I don't have that information'"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

def build_retriever():
    # Carga de documentos y preparacion
    loader = TextLoader("data/promtior_content.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    # Embeddings y vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

def create_rag_chain(retriever):
    # Construir cadena RAG
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": _prompt()},
        return_source_documents=False,
        input_key="question"
    )
