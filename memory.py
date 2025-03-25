import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# Set up the API key (ensure you replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzNlNvf7Y_M78xgNMAA2OvXmn4OkCvU4g"

# Initialize memory
memory = ConversationBufferMemory()

# Initialize the chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create a conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

while True:
    # Get user input
    user_input = input("\nYou: ").strip()

    # Check for exit command
    if user_input.lower() in ['bye', 'exit']:
        print("\nGoodbye!")
        print("\nConversation History:\n", conversation.memory.buffer)
        break

    if not user_input:
        print("\nAI: Please enter a message.")
        continue

    # Get AI response
    response = conversation.run(user_input)

    print("\nAI:", response)
