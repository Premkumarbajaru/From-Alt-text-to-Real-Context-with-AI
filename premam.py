# Install all dependencies in one cell
# Import all modules at the top
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from google.colab import drive
import os

# Security improvement: Use environment variables (create .env file for local dev)
os.environ["PINECONE_API_KEY"] = "pcsk_5QUEsp_SJmeeW8C5Nvbx9jbpLp6bCGZrq76uMRiUhcir9vZ4p4x4rFGEhUyzde8NkpFt6F"
os.environ["INDEX_NAME"] = "premam"

if __name__ == "__main__":
    # Document loading
    print("Loading Documents...")
    loader = TextLoader("./information.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Text splitting
    print("Splitting Documents...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_documents)} chunks")

    # Initialize embeddings
    print("Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Pinecone vector store
    print("Uploading to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=split_documents,
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"],
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    print(f"Successfully inserted {len(split_documents)} documents")