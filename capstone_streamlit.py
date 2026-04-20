import streamlit as st
import uuid
import os
from agent import build_app, ask_agent

# Set your Groq API key here
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"  # REPLACE THIS

st.set_page_config(page_title="ShopEasy FAQ Bot", page_icon="🛒", layout="wide")


@st.cache_resource
def get_app():
    """Load all heavy resources once and cache them."""
    return build_app()


app = get_app()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 ShopEasy Support")
    st.markdown("**Domain:** E-Commerce Customer Support")
    st.markdown("**Topics I can help with:**")
    st.markdown("""
    - Return & Exchange Policy
    - Shipping & Delivery
    - Order Tracking
    - Payment Methods
    - Order Cancellation
    - Discount & Promo Codes
    - Product Warranty
    - Account & Login Issues
    - Cash on Delivery
    """)
    st.divider()
    st.caption("Built with LangGraph + ChromaDB + Groq")
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Header ────────────────────────────────────────────────────────────────────
st.title("ShopEasy Customer Support Bot")
st.caption("Ask me about returns, shipping, payments, orders and more.")

# ── Display history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_agent(app, prompt, st.session_state.thread_id)
            answer = result.get("answer", "Sorry, I could not process that request.")
            sources = result.get("sources", [])
            faith = result.get("faithfulness", None)

        st.markdown(answer)
        if sources:
            st.caption(f"📚 Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
