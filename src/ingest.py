from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "/app/chroma_db"

def ingest_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(
        model="mistral",
        base_url="http://ollama:11434"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIR
    )

    vectordb.persist()