from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

summary_prompt = """
You are a translator where you will be given input text:
{input}
You need to translate that input text into the desired language i.e. {convertLanguage}
"""

prompt_template = PromptTemplate(input_variables=["input", "convertLanguage"], template=summary_prompt)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

chain = prompt_template | llm

res = chain.invoke({ "input": "My name is Aayush and I live in Delhi.", "convertLanguage": "Spanish" })

print(res)
