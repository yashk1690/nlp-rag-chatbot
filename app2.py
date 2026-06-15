import os
import streamlit as st
from dotenv import load_dotenv

# --- NEW: Modern LCEL Imports (Bypassing langchain.chains completely) ---
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load your API Key from the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# 2. Setup Page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("📚 Local PDF Chatbot")
st.markdown("Powered by Docling, Qdrant, FastEmbed, and Groq")


@st.cache_resource
def initialize_rag_pipeline():
    # A. Wake up the embedding model
    embeddings = FastEmbedEmbeddings()

    # B. Connect to your existing Qdrant Database
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="sample_report_2",
        path="qdrant_db"
    )

    # C. Set up the Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    # D. Initialize the Cloud LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    # E. Create the Prompt
    system_prompt = (
        "You are a financial and technical expert. Use the following retrieved context "
        "to answer the user's question accurately. If the answer is not in the context, "
        "simply say that you do not know. Do not guess.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # F. Helper to extract text from the retrieved chunks
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # G. The Modern LCEL Pipeline (No legacy chains needed)
    rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


# 3. Boot up the pipeline
rag_chain = initialize_rag_pipeline()

# 4. Streamlit UI: Chat History setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
user_input = st.chat_input("Ask a question about your PDF...")

if user_input:
    # Show user message instantly
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show a loading spinner while the AI searches Qdrant
    with st.spinner("Searching document and thinking..."):
        # Because we used StrOutputParser, it directly returns the text answer!
        answer = rag_chain.invoke(user_input)

    # Show the AI response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})