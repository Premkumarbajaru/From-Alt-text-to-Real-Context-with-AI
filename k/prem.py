pip install langchain_google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
if __name__ == '__main__':
 llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key='AIzaSyAc1Lv9Q2OVQx3KdAuw6oHhoKSxhm6Rbzs');
 output = llm.invoke("What is the capital of India ?")
 print(output)
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
pip install langchain-google-genai
if __name__ == '__main__':
 summary_prompt  = """
 You are a translater where you will be given input text
 {input}
 you need to translate that input text into desired language i.e {convertLanguage}
 """
 prompt_tempalate = PromptTemplate(input_variables=["input", "convertlanguage"], template=summary_prompt)
 llm = ChatGoogleGenerativeAI(
 model="gemini-1.5-flash",
 api_key='AIzaSyAc1Lv9Q2OVQx3KdAuw6oHhoKSxhm6Rbzs'
 )
 chain = prompt_tempalate | llm
 res = chain.invoke({"input": "My name is Aayush and I live in Delhi.","convertLanguage":"hindi"});
 print(res)
pip install dotenv
 from dotenv import load_dotenv
 import os
 load_dotenv
  pip install langchain_community
  pip install langchain_pinecone
   pip install langchain_huggingface
    from langchain_community.document_loaders import TextLoader
 from langchain_text_splitters import CharacterTextSplitter
 from langchain_pinecone import PineconeVectorStore
 from langchain_huggingface import HuggingFaceEmbeddings
  import os
 from dotenv import load_dotenv, find_dotenv
 pinecone_api_key = os.getenv("pcsk_3pwcW3_DiP14zkL5bxxWukaYiPaW49QHLUy1EiXWQR8LvtNNVB9zKSrtmruTtnhc1MtAEn")
 PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
 PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
 INDEX_NAME = os.environ.get("INDEX_NAME")
 import os
 os.environ["PINECONE_API_KEY"] = "pcsk_5QUEsp_SJmeeW8C5Nvbx9jbpLp6bCGZrq76uMRiUhcir9vZ4p4x4rFGEhUyzde8NkpFt6F"
 os.environ["PINECONE_ENVIRONMENT"] = "PINECONE_ENVIRONMENT"
 os.environ["INDEX_NAME"] = "indexname"
 from google.colab import drive
 drive.mount('/content/drive')  # Mount your Google Drive
 if __name__ == "__main__":
 print("Loading Documents...")
 loader = TextLoader=("information.txt")
 document = loader. load()
 print(f"Loaded (len(document)> documents")
  print("Splitting Documents...")
 splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
 split_documents = splitter.split_documents(document)
 print(f"Split ({len(document)}) documents into ({len(split_documents)}) chunks")

  print("Started Embedding Docuents...")
 embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  from langchain_pinecone import PineconeVectorStore
 pcsk_5QUEsp_SJmeeW8C5Nvbx9jbpLp6bCGZrq76uMRiUhcir9vZ4p4x4rFGEhUyzde8NkpFt6F
 vector_db = PineconeVectorStore.from_documents(split_documents, embeddings, index_name="indexname", pinecone_api_key="pcsk_5QUEsp_SJmeeW8C5N"
 print(f"Inserted {len(split_documents)} documents into VectorDB")