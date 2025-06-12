from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate


def _prompt():
    return ChatPromptTemplate.from_messages([
        ("system","You are a useful Promtior assistant"),
        ("system","You only can answer questions with the context you have"),
        ("system","Use the context to respond the question of the user in a clear and concise way"),
        ("system","If you don't have the answer in your context answer 'I don't have that information'"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    # Document uploads, embeddings and vector store
def build_retriever():
    try:
        loader = TextLoader("data/promtior_content.txt", encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        docs = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store.as_retriever()
    except Exception as e:
        print(f"Error building retriever: {e}")
        raise e

    # Construct RAG chain
def create_rag_chain(retriever):
    try:
        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": _prompt()},
            return_source_documents=False,
            input_key="question"
        )
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        raise e
