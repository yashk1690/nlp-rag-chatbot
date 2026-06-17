import os
import json

from flask import Flask, request, Response, send_from_directory
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__, static_folder="frontend", static_url_path="")


# ── RAG pipeline ──────────────────────────────────────────────────────────────
# Unchanged from the Streamlit version: same embeddings, same vector store,
# same retriever settings, same model, same system prompt, same chain shape.
# It is built once at startup instead of behind @st.cache_resource, which has
# the same effect (one shared instance reused across every request).
def initialize_pipeline():
    embeddings = FastEmbedEmbeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="sample_report",
        path="qdrant_db",
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )

    system_prompt = (
        "You are an expert, highly disciplined document analysis assistant. "
        "Your sole purpose is to help the user understand, analyze, and query "
        "the provided text content.\n\n"

        "GENRE ADAPTABILITY:\n"
        "- Lecture / textbook: clear definitions, conceptual breakdowns, "
        "educational explanations.\n"
        "- Research paper: methodologies, experimental data, citations, "
        "structural conclusions.\n"
        "- Novel / narrative: plot points, character actions, timelines, "
        "narrative themes.\n\n"

        "MATH & SYMBOL FORMATTING — MANDATORY:\n"
        "Whenever the context or your response contains any mathematical formula, "
        "algebraic variable, structural equation, or proof, typeset it using LaTeX. "
        "Use $...$ for inline expressions (e.g., $E=mc^2$) and $$...$$ on its own "
        "line for standalone equations. Plain-text math is NEVER acceptable. "
        "If the document is entirely non-technical, respond in natural prose.\n\n"

        "COMPLIANCE RULES:\n"
        "1. TEXTUAL GROUNDING: Rely strictly on the provided context chunks. "
        "Explain and synthesize, but do not inject outside facts.\n"
        "2. META-QUERIES: If the user requests an overview or asks what the "
        "document is about, synthesize a comprehensive top-down synopsis from "
        "the collective context.\n"
        "3. HONEST CLOSURE: If specific details are absent from the context, "
        "reply: 'The provided context does not contain the specific information "
        "required to answer this query.'\n"
        "4. OUT OF SCOPE: If the user asks something unrelated to document "
        "analysis (e.g., 'write a scraper', 'tell me a joke'), reply: "
        "'OUT OF SCOPE: I can only assist with analysis of the uploaded document.'\n\n"

        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


rag_chain, retriever = initialize_pipeline()


# ── Static frontend ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Chat API ──────────────────────────────────────────────────────────────────
# Streams newline-delimited JSON events so the frontend can render tokens as
# they arrive, the same way the Streamlit version updated its placeholder on
# every chunk. Event shapes:
#   {"type": "sources", "sources": [...]}   sent once, only if show_sources
#   {"type": "chunk", "text": "..."}        sent once per streamed token/chunk
#   {"type": "done"}                        sent once, at the end
#   {"type": "error", "message": "..."}     sent if something goes wrong
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    show_sources = bool(data.get("show_sources"))

    if not user_input:
        return Response(
            json.dumps({"type": "error", "message": "Empty query."}) + "\n",
            mimetype="application/x-ndjson",
        )

    def generate():
        try:
            if show_sources:
                docs = retriever.invoke(user_input)
                sources = [d.page_content for d in docs]
                yield json.dumps({"type": "sources", "sources": sources}) + "\n"

            for chunk in rag_chain.stream(user_input):
                yield json.dumps({"type": "chunk", "text": chunk}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000, use_reloader=False)
