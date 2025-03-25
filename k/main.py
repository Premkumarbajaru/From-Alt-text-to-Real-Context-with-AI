from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Pinecone as PineconeLangChain  # Renamed
from pinecone import Pinecone
import os

if __name__ == "__main__":
    # 1) Initialize Pinecone client
    pinecone_api_key = "pcsk_2CVdBa_TyiKFcvREUS2NUG4P7sn2mnM8ijhphQfGdqGDTRZh1XV1nzTJXvX5rZ6igRytsX"
    index_name = "premam"
    
    # Create Pinecone client instance
    pc = Pinecone(api_key=pinecone_api_key)

    # 2) Loading documents
    print("Loading Documents...")
    loader = TextLoader("./information.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # 3) Splitting documents
    print("Splitting Documents...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = splitter.split_documents(documents)
    print(f"Split into {len(split_documents)} chunks")

    # 4) Initialize embeddings
    print("Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5) Create index if needed
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,  # Must match all-MiniLM-L6-v2 embedding size
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-west-2"
                }
            }
        )
        print(f"Index '{index_name}' created successfully.")

    # 6) Insert into Pinecone using LangChain integration
    print("Inserting Documents into VectorDB...")
    PineconeLangChain.from_documents(
        documents=split_documents,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key
    )

    print(f"Successfully inserted {len(split_documents)} documents")