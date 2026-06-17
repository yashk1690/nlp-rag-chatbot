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
    page_title="Research Guide",
    page_icon="🪶",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Fonts + CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,900&family=Newsreader:ital,wght@0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --ink: #11141b;          /* page background — deep ink-navy, not pure black */
    --panel: #181c25;        /* chat bubble / card background */
    --panel-quiet: #0d1016;  /* sidebar / recessed surfaces */
    --hairline: #2a2f3a;     /* hairline borders, like ruled paper */
    --vellum: #e7e1cd;       /* primary text — warm parchment */
    --slate: #8b93a3;        /* secondary / muted text */
    --brass: #b9935c;        /* primary accent — aged brass */
    --oxblood: #8a3140;      /* secondary accent — binding red */
    --sepia: #c9b896;        /* tone for quoted source text */
}

/* ══ Global type system ══ */
*, *::before, *::after {
    font-family: 'Newsreader', Georgia, 'Times New Roman', serif !important;
}

h1, h2, h3 {
    font-family: 'Fraunces', Georgia, serif !important;
}

[data-testid="stCaptionContainer"] *,
code, pre,
[data-testid="stChatInput"] textarea {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

/* Streamlit's built-in icons (chat avatars, expander arrows, toggle marks,
   alert glyphs, etc.) are rendered as ligature text in a Material Symbols
   font. The global serif override above breaks that font, which makes the
   raw icon names ("face", "smart_toy", "keyboard_arrow_right"...) show up
   as literal, overlapping text. This restores the icon font specifically
   for those elements so the icons render as glyphs again. */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

/* ══ App background ══ */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--ink) !important;
}

/* ══ Layout ══ */
.block-container {
    max-width: 760px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 5rem !important;
}

/* ══ Masthead: a single, deliberate load-in — title rises, then a rule draws beneath it ══ */
@keyframes mastheadRise {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes mastheadRule {
    from { width: 0; opacity: 0; }
    to   { width: 200px; opacity: 1; }
}

h1 {
    color: var(--vellum) !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    text-align: center !important;
    animation: mastheadRise 0.7s ease-out both !important;
}

h1::after {
    content: '';
    display: block;
    height: 1px;
    width: 200px;
    margin: 0.55rem auto 0;
    background: linear-gradient(90deg, transparent, var(--brass), transparent);
    animation: mastheadRule 1s 0.35s ease-out both;
}

h2, h3 {
    color: var(--brass) !important;
    letter-spacing: 0.04em !important;
}

.masthead-subtitle {
    text-align: center;
    color: var(--slate) !important;
    font-style: italic;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    margin-top: -0.3rem;
}

/* ══ Body text ══ */
p, li {
    color: var(--vellum) !important;
}

/* ══ Caption ══ */
[data-testid="stCaptionContainer"] * {
    color: var(--slate) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.03em !important;
}

/* ══ Chat bubbles ══ */
[data-testid="stChatMessage"] {
    background: var(--panel) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 4px !important;
    padding: 0.85rem 1.1rem !important;
    margin-bottom: 0.5rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: #3a4150 !important;
    box-shadow: 0 0 0 1px rgba(185, 147, 92, 0.08) !important;
}

/* User: brass margin rule. The testid match is duplicated with a
   case-insensitive substring selector as a fallback, since Streamlit has
   renamed these internal testids across versions before. */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
[data-testid="stChatMessage"]:has([data-testid*="avatar-user" i]),
[data-testid="stChatMessage"]:has([data-testid*="AvatarUser" i]) {
    border-left: 3px solid var(--brass) !important;
    background: #14171e !important;
}

/* Assistant: oxblood margin rule (same fallback strategy as above) */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
[data-testid="stChatMessage"]:has([data-testid*="avatar-assistant" i]),
[data-testid="stChatMessage"]:has([data-testid*="AvatarAssistant" i]) {
    border-left: 3px solid var(--oxblood) !important;
    background: #181b22 !important;
}

/* Hide user and bot avatars. The broad "contains Avatar" rule is a
   catch-all so this keeps working even if Streamlit renames the
   specific testid in a future release. */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessage"] [data-testid*="Avatar" i] {
    display: none !important;
}

/* ══ Markdown inside messages ══ */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--vellum) !important;
}

[data-testid="stMarkdownContainer"] strong {
    color: var(--brass) !important;
}

/* ══ Chat input ══ */
[data-testid="stChatInput"] {
    background: var(--panel-quiet) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 4px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--brass) !important;
    box-shadow: 0 0 0 2px rgba(185, 147, 92, 0.12) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--vellum) !important;
    caret-color: var(--brass) !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--slate) !important;
}

[data-testid="stChatInput"] button svg {
    fill: var(--brass) !important;
}

/* ══ Sidebar ══ */
[data-testid="stSidebar"] {
    background: var(--panel-quiet) !important;
    border-right: 1px solid var(--hairline) !important;
}

[data-testid="stSidebar"] * {
    color: var(--vellum) !important;
}

[data-testid="stSidebar"] h3 {
    color: var(--brass) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: var(--slate) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--hairline) !important;
}

/* Sidebar button */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: 1px solid var(--brass) !important;
    color: var(--brass) !important;
    border-radius: 3px !important;
    letter-spacing: 0.03em !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}

[data-testid="stSidebar"] button:hover {
    background: rgba(185, 147, 92, 0.1) !important;
}

/* ══ Info / alert box ══ */
[data-testid="stAlert"] {
    background: var(--panel) !important;
    border: 1px solid var(--hairline) !important;
    border-left: 3px solid var(--brass) !important;
    border-radius: 4px !important;
}

[data-testid="stAlert"] * {
    color: var(--vellum) !important;
}

/* ══ Code blocks (source excerpts) ══ */
pre {
    background: var(--panel-quiet) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 2px !important;
    color: var(--sepia) !important;
}

code {
    background: var(--panel-quiet) !important;
    color: var(--sepia) !important;
}

/* ══ Expander ══ */
details > summary {
    color: var(--slate) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.02em !important;
    font-style: italic !important;
    cursor: pointer !important;
}

details > summary:hover {
    color: var(--brass) !important;
}

/* ══ Divider ══ */
hr {
    border-color: var(--hairline) !important;
}

/* ══ Spinner ══ */
[data-testid="stSpinner"] * {
    color: var(--brass) !important;
}

/* ══ Scrollbar ══ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--panel-quiet); }
::-webkit-scrollbar-thumb { background: var(--hairline); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--brass); }

/* ══ Subtle vignette ══ */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.25) 100%);
    pointer-events: none;
    z-index: 9997;
}
</style>
""", unsafe_allow_html=True)


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
    st.markdown("### Research Guide")
    st.markdown("---")

    show_sources = st.toggle(
        "Show cited excerpts",
        value=False,
        help="Reveal the source passages the assistant drew on to answer each query.",
    )

    st.markdown("---")
    st.markdown("**Built with**")
    st.caption("LLaMA 3.1-8B · Groq Cloud")
    st.caption("FastEmbed · BAAI/bge-small")
    st.caption("Qdrant · local vector store")
    st.caption("Docling multimodal parser")
    st.markdown("---")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Research Guide")
st.markdown(
    "<p class='masthead-subtitle'>A reading companion for research papers and scholarly texts</p>",
    unsafe_allow_html=True,
)


# ── Session init ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info(
        "**Ready when you are.**  \n"
        "Your document has been indexed and is ready for analysis.  \n"
        "Ask a question below to begin."
    )


# ── Source chunk renderer ─────────────────────────────────────────────────────
def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    label = f"{len(sources)} source excerpt{'s' if len(sources) != 1 else ''} referenced"
    with st.expander(label):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**Excerpt {i}**")
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
user_input = st.chat_input("Ask a question about the document…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    sources: list[str] = []
    if show_sources:
        with st.spinner("Retrieving relevant passages…"):
            docs = retriever.invoke(user_input)
            sources = [d.page_content for d in docs]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        for chunk in rag_chain.stream(user_input):
            full_text += chunk
            placeholder.markdown(normalize_math(full_text) + " ▌")

        placeholder.markdown(normalize_math(full_text))

        if show_sources:
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources,
    })