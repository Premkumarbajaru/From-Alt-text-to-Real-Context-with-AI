from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Google Generative AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-Flash")

# Define the query
query = "Where was the Taj Mahal constructed?"

# Create a prompt template
prompt_template = PromptTemplate(input_variables=["query"], template="{query}")

# Initialize Pinecone Vector Store
vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

# Pull the prompt from the hub
prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Create a combined documents chain
combined_docs_chain = create_stuff_documents_chain(llm, prompt)

# Create a retriever chain
retriever_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combined_docs_chain=combined_docs_chain)

# Invoke the retriever chain with the query
result = retriever_chain.invoke({"input": query})

# Print the result
print(result)