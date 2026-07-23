"""
streamlit_app.py — TaxLearn (light, card-based conversational chat).

Clean light UI: sidebar with new-chat action, centered greeting with
suggestion cards, docked input with disclaimer. Real conversational memory
(follow-up questions work via query condensation), isolated per browser
session.

RUN:  streamlit run app/streamlit_app.py
"""

import uuid

import streamlit as st
for _key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "LLAMA_CLOUD_API_KEY"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.query_hybrid import (
    load_index_and_collection,
    build_hybrid_retriever_only,
    RERANK_TOP_N,
)

# Benchmarked: reranking (not the LLM call) is the real per-message bottleneck,
# ~8s on bge-reranker-base vs ~1s here — this is a deliberate speed-over-quality
# tradeoff scoped to chat only; app.py/api.py keep the stronger reranker.
CHAT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ASSISTANT_AVATAR = ":material/account_balance:"

# (icon, accent key, card label, question sent when clicked)
SUGGESTION_CARDS = [
    (":material/percent:", "blue", "PIT rates", "What are the new personal income tax rates?"),
    (":material/badge:", "violet", "Who's exempt?", "Who is exempt from personal income tax under the new tax law?"),
    (":material/event_upcoming:", "amber", "Filing deadlines", "What are the deadlines and obligations for filing annual returns?"),
]

st.set_page_config(page_title="TaxLearn", page_icon=ASSISTANT_AVATAR, layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* ---------- light backdrop ---------- */
    .stApp { background: #F6F7F9; color: #14171F; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    /* Hide the hamburger menu + footer, but keep the header itself — it
       holds the sidebar's expand/collapse toggle, which must stay usable. */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; padding-bottom: 8rem; max-width: 780px; }

    section[data-testid="stSidebar"] {
        background: #FFFFFF; border-right: 1px solid #E7E9EE;
    }

    /* ---------- logo mark ---------- */
    .logo-dot {
        width: 40px; height: 40px; border-radius: 50%;
        background: linear-gradient(135deg, #14171F, #2B3040);
        display:flex; align-items:center; justify-content:center;
        color:#FFFFFF; font-size:1.1rem; margin: 0 auto;
    }

    /* ---------- greeting ---------- */
    .greeting-wrap { text-align:center; margin-top: 1rem; margin-bottom: 2.25rem; }
    .greeting-hi { color:#8A8F9C; font-size:1.05rem; margin: 1rem 0 0.15rem 0; }
    .greeting-title {
        font-family:'Space Grotesk',sans-serif; font-weight:700;
        font-size:1.7rem; color:#14171F; letter-spacing:-0.01em;
    }

    /* ---------- suggestion cards (main area only) ---------- */
    div.block-container div[data-testid="stButton"] button[kind="secondary"] {
        background:#FFFFFF; border:1px solid #E7E9EE; border-radius:14px;
        padding: 1rem 1.1rem; text-align:left; white-space:normal;
        min-height: 96px; box-shadow: 0 1px 2px rgba(20,23,31,0.03);
        transition: all 0.18s ease; color:#3B3F4A; font-size:0.9rem;
    }
    div.block-container div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color:#C9CEDA; box-shadow: 0 6px 18px rgba(20,23,31,0.08);
        transform: translateY(-1px); color:#14171F;
    }
    .st-key-card_blue button { border-top: 3px solid #2F6FED; }
    .st-key-card_violet button { border-top: 3px solid #8B5CF6; }
    .st-key-card_amber button { border-top: 3px solid #D97706; }

    /* ---------- sidebar conversation list ---------- */
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
        background:transparent !important; border:none !important; box-shadow:none !important;
        text-align:left; color:#4B5160; font-size:0.86rem; padding:0.45rem 0.6rem;
        min-height:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
        background:#F1F3F7 !important; color:#14171F;
    }
    [data-testid="stSidebar"] [class*="st-key-convwrap_active_"] button[kind="secondary"] {
        background:#EFF4FF !important; color:#2F6FED !important; font-weight:500;
    }
    [data-testid="stSidebar"] .conv-label {
        font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase;
        color:#9AA0AD; margin:1.25rem 0 0.4rem 0.6rem;
    }

    /* ---------- chat messages ---------- */
    [data-testid="stChatMessage"] {
        background: transparent !important; border: none !important;
        padding: 0.35rem 0 1.35rem 0 !important;
    }
    [data-testid="stChatMessageContent"] {
        background: #FFFFFF; border: 1px solid #E7E9EE;
        border-radius: 16px; padding: 1rem 1.25rem !important;
        box-shadow: 0 1px 3px rgba(20,23,31,0.04);
    }
    [data-testid="stChatMessageContent"] p {
        font-size:0.98rem; line-height:1.7; color:#20242E; margin-bottom:0.5rem;
    }
    [data-testid="stChatMessageContent"] p:last-child { margin-bottom:0; }

    /* user messages: tinted */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: #EFF4FF; border-color:#DCE6FB;
    }

    /* ---------- chat input ---------- */
    [data-testid="stChatInput"] {
        background: #FFFFFF !important; border: 1px solid #E7E9EE !important;
        border-radius: 14px !important; box-shadow: 0 2px 10px rgba(20,23,31,0.05);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color:#2F6FED !important;
        box-shadow: 0 0 0 3px rgba(47,111,237,0.12);
    }
    [data-testid="stChatInput"] textarea {
        background:#FFFFFF !important; color:#14171F !important; font-size:0.98rem !important;
        -webkit-text-fill-color:#14171F !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color:#9AA0AD !important; }
    [data-testid="stChatInput"] button {
        background:#2F6FED !important; border:none !important; border-radius:9px !important;
    }
    [data-testid="stChatInput"] button svg { color:#FFFFFF !important; fill:#FFFFFF !important; }
    [data-testid="stBottomBlockContainer"] { background: transparent !important; }

    /* ---------- sidebar primary button (New chat) ---------- */
    [data-testid="stSidebar"] button[kind="primary"] {
        background:#2F6FED !important; border:none !important; color:#FFFFFF !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover { background:#255FD1 !important; }

    /* ---------- source chips ---------- */
    .src-label {
        font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase;
        color:#9AA0AD; margin:1.1rem 0 0.6rem 0;
    }
    .src-chip {
        display:inline-block; background:#F1F3F7; border:1px solid #E7E9EE;
        border-radius:8px; padding:0.4rem 0.75rem; margin:0 0.45rem 0.45rem 0;
        font-size:0.82rem; color:#4B5160;
    }

    .disclaimer { color:#9AA0AD; font-size:0.76rem; text-align:center; margin-top:0.4rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing TaxLearn…")
def get_retriever_and_reranker():
    """Expensive, read-only, stateless — safe to build once and share across
    every user/session in this app process."""
    index, chroma_collection = load_index_and_collection()
    retriever = build_hybrid_retriever_only(index, chroma_collection)
    reranker = SentenceTransformerRerank(model=CHAT_RERANKER_MODEL, top_n=RERANK_TOP_N)
    return retriever, reranker


def get_chat_engine(conv_id):
    """One chat engine per conversation (st.session_state), not shared
    globally — each thread gets its own ChatMemoryBuffer so conversations
    don't leak into each other or between different users/tabs. Built on
    top of the cached retriever/reranker above, so this stays cheap after
    the first call."""
    if conv_id not in st.session_state.chat_engines:
        retriever, reranker = get_retriever_and_reranker()
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        st.session_state.chat_engines[conv_id] = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            node_postprocessors=[reranker],
            system_prompt=(
                "You are TaxLearn, a helpful assistant answering questions about "
                "Nigeria's 2025 tax laws. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't have that "
                "information in the provided documents. Always cite the source "
                "document name for any fact. Answer in clear, plain English."
            ),
        )
    return st.session_state.chat_engines[conv_id]


def new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": None, "messages": []}
    st.session_state.current_id = conv_id
    return conv_id


def render_sources(sources):
    if not sources:
        return
    chips = '<div class="src-label">Sources</div>'
    for s in sources:
        chips += f'<span class="src-chip">{s}</span>'
    st.markdown(chips, unsafe_allow_html=True)


# ---------- conversation state: multiple threads, one active ----------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "chat_engines" not in st.session_state:
    st.session_state.chat_engines = {}
if "current_id" not in st.session_state:
    new_conversation()

# Warm the shared retriever/reranker (~15-30s cold) on page load rather than
# the user's first message — st.cache_resource makes this a no-op after the
# first call, regardless of which conversation triggers it.
get_retriever_and_reranker()

conv = st.session_state.conversations[st.session_state.current_id]

# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="logo-dot">◆</div>', unsafe_allow_html=True)
    st.space("small")
    if st.button("New chat", icon=":material/add:", width="stretch", type="primary"):
        new_conversation()
        st.rerun()

    # history — most recent first, skip empty threads other than the active one
    ordered_ids = list(st.session_state.conversations.keys())[::-1]
    listed = [
        cid for cid in ordered_ids
        if st.session_state.conversations[cid]["messages"] or cid == st.session_state.current_id
    ]
    if len(listed) > 1:
        st.markdown('<div class="conv-label">Recent</div>', unsafe_allow_html=True)
        for cid in listed:
            title = st.session_state.conversations[cid]["title"] or "New conversation"
            wrap_key = f"convwrap_{'active_' if cid == st.session_state.current_id else ''}{cid}"
            with st.container(key=wrap_key):
                if st.button(title, key=f"conv_{cid}", width="stretch"):
                    st.session_state.current_id = cid
                    st.rerun()

    st.caption("TaxLearn · Nigeria 2025 tax law")

# ---------- input (declared early; st.chat_input auto-docks at the bottom) ----------
prompt = st.chat_input("Ask about Nigeria's tax laws…")

# ---------- empty state: greeting + suggestion cards ----------
if not conv["messages"] and not prompt:
    st.markdown(
        '<div class="greeting-wrap">'
        '<div class="logo-dot">◆</div>'
        '<div class="greeting-hi">Hi, there 👋</div>'
        '<div class="greeting-title">How can we help with your taxes?</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, (icon, accent, label, question) in zip(cols, SUGGESTION_CARDS):
        with col:
            if st.button(
                f"{icon}  **{label}**\n\n{question}",
                key=f"card_{accent}",
                width="stretch",
            ):
                prompt = question

# ---------- replay history ----------
for msg in conv["messages"]:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

# ---------- handle new message ----------
if prompt:
    if conv["title"] is None:
        conv["title"] = prompt[:40] + ("…" if len(prompt) > 40 else "")

    conv["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Reading the tax documents…"):
            engine = get_chat_engine(st.session_state.current_id)
            response = engine.chat(prompt)
            answer = str(response)

            sources = []
            for node in getattr(response, "source_nodes", []):
                fname = node.metadata.get("file_name", "unknown")
                if fname not in sources:
                    sources.append(fname)

        st.markdown(answer)
        render_sources(sources)

    conv["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()

st.markdown(
    '<div class="disclaimer">Answers are grounded in Nigeria\'s official 2025 tax documents, with sources cited — always confirm details for your specific situation. Not legal advice.</div>',
    unsafe_allow_html=True,
)
