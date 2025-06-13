import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Information ingestion
def doc_creation ():
    all_docs = []
    scrape_path = "https://www.promtior.ai/service" #CREAR VARIABLES GLOBALES
    pdf_path = "data/AI_Engineer.pdf"

    print(f"📄 Loading scrape: {scrape_path}")
    loader = WebBaseLoader(scrape_path)
    web_docs = loader.load()
    all_docs.extend(web_docs)

    if os.path.exists(pdf_path):
        try:
            print(f"📄 Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pdf_file = loader.load()
            all_docs.extend(pdf_file)
        except Exception as e:
            print(f"❌ Error loading PDF {pdf_path}: {e}")
    else:
        print(f"⚠️ PDF not found: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n=== ", "\n## ", "\n• ", ". ", "\n", " "]
    )
    docs = splitter.split_documents(all_docs)

    return docs

# Embeddings and vector store
def build_retriever(docs = doc_creation()):
    if docs is None:
        docs = doc_creation()
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(docs, embeddings)

        print("✅ Retriever built successfully")

        return vector_store.as_retriever()

    except Exception as e:
        print(f"Error building retriever: {e}")
        raise e