import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.block-container {
    max-width: 780px;
    padding-top: 1.8rem;
    padding-bottom: 5rem;
}

/* ── Title ── */
h1 { letter-spacing: -0.5px; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.35rem;
    border: 1px solid transparent;
    transition: border-color 0.15s;
}

/* Assistant bubble: very faint violet tint + border */
[data-testid="stChatMessage"][data-testid*="assistant"],
div[class*="st-emotion-cache"] [data-testid="stChatMessage"]:has(svg[data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(99, 102, 241, 0.04);
    border-color: rgba(99, 102, 241, 0.12);
}

/* ── Source expander (smaller, muted) ── */
details > summary {
    font-size: 0.78rem;
    color: #9ca3af;
    cursor: pointer;
}
details > summary:hover { color: #6b7280; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption { color: #94a3b8 !important; }
[data-testid="stSidebar"] hr { border-color: #1e293b !important; }

/* Sidebar toggle and button */
[data-testid="stSidebar"] [data-testid="stToggle"] > label span { color: #e2e8f0 !important; }

/* ── Clear button ── */
[data-testid="stSidebar"] button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] button:hover {
    background: #334155 !important;
}

/* ── Info box on empty state ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 0.9rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #6366f1; }
</style>
""", unsafe_allow_html=True)


# ── Math normalizer ───────────────────────────────────────────────────────────
def normalize_math(text: str) -> str:
    """
    Streamlit uses KaTeX for math, triggered by $...$ and $$...$$.
    LLMs sometimes output \\(...\\) for inline and \\[...\\] for display math.
    This converter handles both, plus stray \\[ without a closing \\].
    """
    # Display math:  \[...\]  →  $$...$$
    text = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', text)
    # Inline math:   \(...\)  →  $...$
    text = re.sub(r'\\\(([\s\S]*?)\\\)', r'$\1$', text)
    return text


# ── RAG pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_pipeline():
    embeddings = FastEmbedEmbeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="prob",
        path="qdrant_db",
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )

    system_prompt = (
        "You are a strict, highly disciplined Document Retrieval Assistant. "
        "You have no personal identity, no name, and no outside expertise. "
        "Your ONLY function is to answer questions using the provided context.\n\n"

        "MATH FORMATTING — MANDATORY: Always typeset every formula, symbol, "
        "and equation using LaTeX. "
        "Use $...$ for inline expressions (e.g. $x^2 + y^2 = z^2$) and "
        "$$...$$ on its own line for standalone equations "
        "(e.g. $$\\sum_{{i=1}}^n x_i = \\mu n$$). "
        "Never write maths as plain text.\n\n"

        "Follow these exact routing rules:\n"
        "1. OUT OF SCOPE: If the user asks a conversational question (like 'what is your name'), "
        "or asks you to explain a general concept not mentioned in the text, "
        "you MUST reply ONLY with: 'OUT OF SCOPE: I can only answer questions related to the uploaded document.'\n"
        "2. INSUFFICIENT DATA: If the user asks about a topic that IS in the context, but the specific "
        "data needed to answer is missing, you MUST reply ONLY with: 'INSUFFICIENT DATA.'\n"
        "3. SUCCESS: Otherwise, answer the question accurately using ONLY the provided text.\n\n"

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 DocChat")
    st.markdown("---")

    show_sources = st.toggle("Show source chunks", value=False,
                             help="Reveal the raw document passages used to answer each query.")

    st.markdown("---")
    st.markdown("**Stack**")
    st.caption("🧠  LLaMA 3.1-8B · Groq")
    st.caption("📐  FastEmbed (BAAI/bge-small)")
    st.caption("🗃️  Qdrant · local")
    st.caption("📄  Docling multimodal parser")
    st.markdown("---")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 DocChat")
st.caption("Ask questions about your uploaded documents · math rendered via KaTeX")

if not st.session_state.get("messages"):
    st.info(
        "💡 **Getting started** — ask anything about your document. "
        "Equations like $E = mc^2$ and full display math render automatically."
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helper: render a single source chunk list ─────────────────────────────────
def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source chunks used"):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**Chunk {i}**")
            # Truncate very long chunks for readability
            preview = chunk[:450] + ("…" if len(chunk) > 450 else "")
            st.code(preview, language=None)
            if i < len(sources):
                st.divider()


# ── Replay chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(normalize_math(msg["content"]))
        if show_sources and msg.get("sources"):
            render_sources(msg["sources"])


# ── Live interaction ──────────────────────────────────────────────────────────
user_input = st.chat_input("Ask something about your document…")

if user_input:
    # Save and show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Fetch source chunks (if toggle is on) — runs before streaming
    sources: list[str] = []
    if show_sources:
        with st.spinner("Retrieving chunks…"):
            docs = retriever.invoke(user_input)
            sources = [d.page_content for d in docs]

    # Stream the LLM response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        for chunk in rag_chain.stream(user_input):
            full_text += chunk
            # Render incrementally; trailing cursor gives streaming feel
            placeholder.markdown(normalize_math(full_text) + " ▌")

        # Final render without cursor
        placeholder.markdown(normalize_math(full_text))

        if show_sources:
            render_sources(sources)

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources,
    })