import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load environment variables from the .env file
load_dotenv()

# Verify that the API key is being read correctly
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found! Make sure it's set in your .env file.")


def test_groq_connection():
    print("Initializing connection to Groq...")

    # 2. Initialize the LLaMA 3 model via Groq
    # We use the 8B model here because it's blazing fast for testing
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
    )

    # 3. Send a test message to the LLM
    print("Sending test prompt to LLaMA 3...")
    response = llm.invoke(
        "Hi LLaMA! nice to meet you, looking forward to working with you!!")

    # 4. Print the output
    print("\n--- Groq Response ---")
    print(response.content)
    print("---------------------\n")


if __name__ == "__main__":
    test_groq_connection()