# LangChain Tutorial with Google Gemini
# Install required packages:
# pip install langchain-google-genai python-dotenv

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# 1. Environment Setup
# -------------------
# Load API key from .env file
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# 2. Model Initialization
# ----------------------

# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory

# Initialize the chat model
model = ChatGroq(model="llama3-8b-8192", api_key=api_key)


# 3. Basic Usage Example
# ---------------------
def basic_example():
    print("\n=== Basic Model Usage ===")
    response = model.invoke("What is LangChain?")
    print(f"Response: {response.content}\n")


# 4. Using Prompt Templates
# ------------------------
def prompt_template_example():
    print("\n=== Using Prompt Templates ===")
    prompt = ChatPromptTemplate.from_template("Explain {concept} in simple terms")
    chain = prompt | model
    response = chain.invoke({"concept": "LangChain"})
    print(f"Response: {response.content}\n")


# 5. Building a Chain
# ------------------
def chain_example():
    print("\n=== Using LLMChain ===")
    prompt = ChatPromptTemplate.from_template("Give me 3 {adjective} facts about {topic}")
    chain = prompt | model
    response = chain.invoke({"adjective": "interesting", "topic": "artificial intelligence"})
    print(f"Response: {response['text']}\n")


# 6. Using Memory
# --------------
def memory_example():
    print("\n=== Using Conversation Memory ===")
    memory = ConversationBufferMemory()
    memory.save_context({"input": "My name is User"}, {"output": "Nice to meet you, User!"})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )

    chain = LLMChain(llm=model, prompt=prompt, memory=memory)

    print("First response:")
    response = chain.invoke({"input": "What's my name?"})
    print(f"Response: {response['text']}\n")

    print("Second response:")
    response = chain.invoke({"input": "Tell me something about AI"})
    print(f"Response: {response['text']}\n")


# 7. Interactive Chat
# ------------------
def interactive_chat():
    print("\n=== Interactive Chat ===")
    print("Simple Chatbot (type 'exit' to quit)")
    print("---------------------------------")

    messages = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        messages.append(HumanMessage(content=user_input))
        response = model.invoke(messages)
        messages.append(response)

        print(f"Bot: {response.content}")


# Run the examples
if __name__ == "__main__":
    print("LangChain Tutorial with Google Gemini")
    print("====================================")

    # Choose which examples to run
    basic_example()
    prompt_template_example()
    chain_example()
    memory_example()

    # Run interactive chat last
    interactive_chat()
