import os
import re
import streamlit as st
import streamlit.components.v1 as components
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
    page_title="DOCCHAT.exe",
    page_icon="🟢",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Font + CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
/* ══ Global: monospace everywhere ══ */
*, *::before, *::after {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* ══ App background ══ */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #050505 !important;
}

/* ══ Layout ══ */
.block-container {
    max-width: 820px !important;
    padding-top: 0.6rem !important;
    padding-bottom: 5rem !important;
}

/* ══ Title glow pulse ══ */
@keyframes matrixGlow {
    0%, 100% {
        text-shadow:
            0 0 6px rgba(0,255,65,0.6),
            0 0 14px rgba(0,255,65,0.2);
    }
    50% {
        text-shadow:
            0 0 14px rgba(0,255,65,1),
            0 0 35px rgba(0,255,65,0.5),
            0 0 60px rgba(0,255,65,0.15);
    }
}

h1 {
    color: #00ff41 !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    animation: matrixGlow 3s ease-in-out infinite !important;
}

h2, h3 {
    color: #00ff41 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    text-shadow: 0 0 6px rgba(0,255,65,0.3) !important;
}

/* ══ Body text ══ */
p, li {
    color: #a8ffb8 !important;
}

/* ══ Caption ══ */
[data-testid="stCaptionContainer"] * {
    color: #2a8040 !important;
    font-size: 0.71rem !important;
    letter-spacing: 1px !important;
}

/* ══ Chat bubbles ══ */
[data-testid="stChatMessage"] {
    background: #060d06 !important;
    border: 1px solid #0d2a0d !important;
    border-radius: 3px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.45rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: #1a4d1a !important;
    box-shadow: 0 0 12px rgba(0,255,65,0.04) !important;
}

/* User: bright green stripe */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid #00ff41 !important;
    background: #040a04 !important;
}

/* Assistant: mid-green stripe */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #00b32c !important;
    background: #050d05 !important;
}

/* Hide user and bot avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* ══ Markdown inside messages ══ */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #a8ffb8 !important;
}

[data-testid="stMarkdownContainer"] strong {
    color: #00ff41 !important;
}

/* ══ Chat input ══ */
[data-testid="stChatInput"] {
    background: #040904 !important;
    border: 1px solid #0d2a0d !important;
    border-radius: 3px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #00ff41 !important;
    box-shadow: 0 0 14px rgba(0,255,65,0.15) !important;
}

[data-testid="stChatInput"] textarea {
    color: #00ff41 !important;
    caret-color: #00ff41 !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #1a4d1a !important;
}

[data-testid="stChatInput"] button svg {
    fill: #00ff41 !important;
}

/* ══ Sidebar ══ */
[data-testid="stSidebar"] {
    background: #020602 !important;
    border-right: 1px solid #0d2a0d !important;
}

[data-testid="stSidebar"] * {
    color: #00ff41 !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #2a8040 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #0d2a0d !important;
}

/* Sidebar button */
[data-testid="stSidebar"] button {
    background: #020602 !important;
    border: 1px solid #00ff41 !important;
    color: #00ff41 !important;
    border-radius: 2px !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-size: 0.68rem !important;
    transition: all 0.2s !important;
}

[data-testid="stSidebar"] button:hover {
    background: #003d00 !important;
    box-shadow: 0 0 14px rgba(0,255,65,0.35) !important;
}

/* ══ Info / alert box ══ */
[data-testid="stAlert"] {
    background: #020a02 !important;
    border: 1px solid #0d2a0d !important;
    border-left: 3px solid #00ff41 !important;
    border-radius: 3px !important;
}

[data-testid="stAlert"] * {
    color: #00ff41 !important;
}

/* ══ Code blocks (source chunks) ══ */
pre {
    background: #020802 !important;
    border: 1px solid #0d2a0d !important;
    border-radius: 2px !important;
    color: #00cc35 !important;
}

code {
    background: #020802 !important;
    color: #00cc35 !important;
}

/* ══ Expander ══ */
details > summary {
    color: #2a7a3a !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
}

details > summary:hover {
    color: #00b32c !important;
}

/* ══ Divider ══ */
hr {
    border-color: #0d2a0d !important;
}

/* ══ Spinner ══ */
[data-testid="stSpinner"] * {
    color: #00b32c !important;
}

/* ══ Scrollbar ══ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #020602; }
::-webkit-scrollbar-thumb { background: #0d3d0d; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00ff41; }

/* ══ Scanlines overlay ══ */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent 0px,
        transparent 2px,
        rgba(0, 0, 0, 0.045) 2px,
        rgba(0, 0, 0, 0.045) 4px
    );
    pointer-events: none;
    z-index: 9997;
}
</style>
""", unsafe_allow_html=True)


# ── Matrix rain banner ────────────────────────────────────────────────────────
components.html("""
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #050505; overflow: hidden; }
  canvas { display: block; width: 100%; }
</style>
<canvas id="rain"></canvas>
<script>
  const c  = document.getElementById('rain');
  const cx = c.getContext('2d');

  const FS    = 12;
  const CHARS = '01アカサタナハABCDEF011010';
  let cols, drops;

  function resize() {
    c.width  = document.documentElement.clientWidth || 800;
    c.height = 64;
    cols  = Math.floor(c.width / FS);
    drops = Array.from({ length: cols }, () => Math.random() * -10);
  }
  resize();
  window.addEventListener('resize', resize);

  function draw() {
    cx.fillStyle = 'rgba(5,5,5,0.14)';
    cx.fillRect(0, 0, c.width, c.height);
    cx.font = FS + 'px "Courier New", monospace';

    for (let i = 0; i < cols; i++) {
      const ch = CHARS[Math.floor(Math.random() * CHARS.length)];
      const y  = Math.floor(drops[i]) * FS;

      if (y >= 0 && y < c.height) {
        // Lead char: near-white flash; rest: matrix green
        cx.fillStyle   = (y < FS) ? '#ccffdd' : '#00ff41';
        cx.shadowColor = '#00ff41';
        cx.shadowBlur  = y < FS ? 10 : 4;
        cx.fillText(ch, i * FS, y + FS);
        cx.shadowBlur  = 0;
      }

      if (y > c.height && Math.random() > 0.96) {
        drops[i] = Math.random() * -8;
      }
      drops[i] += 0.65;
    }
  }

  setInterval(draw, 48);
</script>
""", height=67)


# ── Math normalizer ───────────────────────────────────────────────────────────
def normalize_math(text: str) -> str:
    """
    Streamlit renders math via KaTeX using $...$ and $$...$$.
    LLMs often output \\(...\\) for inline and \\[...\\] for display math.
    This normalizes both forms so KaTeX can pick them up.
    """
    text = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', text)
    text = re.sub(r'\\\(([\s\S]*?)\\\)', r'$\1$', text)
    return text


# ── RAG pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_pipeline():
    embeddings = FastEmbedEmbeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="sample_report_1",
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ DOCCHAT")
    st.markdown("---")

    show_sources = st.toggle(
        "Show source chunks",
        value=False,
        help="Reveal the raw document passages used to answer each query.",
    )

    st.markdown("---")
    st.markdown("**// STACK**")
    st.caption("// LLaMA 3.1-8B · Groq Cloud")
    st.caption("// FastEmbed · BAAI/bge-small")
    st.caption("// Qdrant · local vector store")
    st.caption("// Docling multimodal parser")
    st.markdown("---")

    if st.button("[ CLEAR MEMORY ]", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("▶ DOCCHAT.exe")
st.caption("// RAG DOCUMENT SEARCHER")


# ── Session init ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info(
        "**> SYSTEM ONLINE**  \n"
        "Vector store connected · Awaiting query.  \n"
        "Type below to begin document analysis."
    )


# ── Source chunk renderer ─────────────────────────────────────────────────────
def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander(f"// {len(sources)} CONTEXT CHUNKS RETRIEVED"):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**[CHUNK {i:02d}]**")
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


# ── Live input + streaming response ──────────────────────────────────────────
user_input = st.chat_input("query:// enter search vector…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    sources: list[str] = []
    if show_sources:
        with st.spinner("// retrieving context vectors…"):
            docs = retriever.invoke(user_input)
            sources = [d.page_content for d in docs]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        for chunk in rag_chain.stream(user_input):
            full_text += chunk
            placeholder.markdown(normalize_math(full_text) + " █")

        placeholder.markdown(normalize_math(full_text))

        if show_sources:
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources,
    })