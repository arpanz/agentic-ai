# 🛒 ShopEasy Customer Support Bot

> **Agentic AI Capstone Project** — A production-grade, multi-turn FAQ chatbot for an e-commerce platform, built using LangGraph, ChromaDB, Groq LLM, and Streamlit.

---

## 📌 Problem Statement

**Domain:** E-Commerce Customer Support  
**User:** Online shoppers seeking instant answers  
**Problem:** Customer support teams receive hundreds of repetitive daily queries on returns, shipping, payments, and order tracking. This bot handles common queries 24/7 from a structured knowledge base without hallucinating policies or prices.  
**Success:** Agent answers correctly from the KB, admits uncertainty when out-of-scope, maintains context across a multi-turn conversation, and achieves faithfulness score ≥ 0.7.

---

## 🏗️ Architecture

The agent is built as a **LangGraph StateGraph** with 8 nodes connected via conditional and fixed edges:

```
User Question
     │
     ▼
[memory_node]  ← Appends to history, sliding window (last 6), extracts user name
     │
     ▼
[router_node]  ← LLM routes to: retrieve / tool / memory_only
     │
  ┌──┴──────────────┐──────────────┐
  ▼                  ▼              ▼
[retrieve_node]  [tool_node]   [skip_node]
  (ChromaDB RAG)  (datetime)    (greetings)
  └──────────────────┴──────────────┘
                    │
                    ▼
             [answer_node]  ← Grounded system prompt + LLM response
                    │
                    ▼
              [eval_node]   ← Faithfulness score (0.0–1.0), retry if < 0.7
                    │
              ┌─────┴─────┐
         (retry)        (pass)
          answer        [save_node] → END
```

---

## ✅ Six Mandatory Capabilities

| # | Capability | Implementation |
|---|---|---|
| 1 | **LangGraph StateGraph** | `CapstoneState` TypedDict with 11 fields; 8 nodes; conditional routing after `router` and `eval` |
| 2 | **ChromaDB RAG (10 docs)** | `all-MiniLM-L6-v2` embeddings; 10 topic-specific documents (100–400 words each); top-3 retrieval |
| 3 | **MemorySaver + thread_id** | Sliding window of last 6 messages; user name extraction from natural language |
| 4 | **Self-reflection eval node** | LLM-rated faithfulness score; retries answer if score < 0.7; `MAX_EVAL_RETRIES = 2` |
| 5 | **Tool use (beyond retrieval)** | `tool_node` provides current date + standard/express delivery date estimates |
| 6 | **Streamlit deployment** | `st.cache_resource` for one-time init; `st.session_state` for chat history + thread ID; sidebar with topic list |

---

## 📚 Knowledge Base Topics

The KB contains **10 documents**, each covering one specific policy area:

1. Return Policy
2. Shipping Policy
3. Order Tracking
4. Payment Methods
5. Order Cancellation
6. Discount & Promo Codes
7. Product Warranty
8. Account & Login Issues
9. Cash on Delivery
10. Exchange Policy

---

## 🗂️ Project Structure

```
agentic-ai/
├── agent.py                  # Core LangGraph agent (state, nodes, graph)
├── capstone_streamlit.py     # Streamlit UI
├── day13_capstone.ipynb      # Exploration & testing notebook
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Library / Service |
|---|---|
| LLM | `llama3-8b-8192` via **Groq** (`langchain-groq`) |
| Agent Framework | **LangGraph** (`StateGraph`, `MemorySaver`) |
| Vector Store | **ChromaDB** (in-memory) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| UI | **Streamlit** |
| Language | Python 3.10+ |

---

## 🚀 Setup & Running

### 1. Clone the repo

```bash
git clone https://github.com/arpanz/agentic-ai.git
cd agentic-ai
```

### 2. Install dependencies

```bash
pip install streamlit langchain-groq langgraph chromadb sentence-transformers
```

### 3. Set your Groq API key

Open `capstone_streamlit.py` and replace the placeholder:

```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"  # Replace with your key
```

Or export it in your terminal:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the Streamlit app

```bash
streamlit run capstone_streamlit.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💬 Sample Questions to Try

```
"What is the return policy for electronics?"
"How can I track my order?"
"When will my order arrive if I order now?"
"Can I pay via UPI?"
"My name is Arpan. What payment methods do you support?"
"What did I just ask you about?"  ← Memory test
"How do I book a flight?"         ← Out-of-scope test
```

---

## 🧪 Test Coverage

The notebook (`day13_capstone.ipynb`) includes:

- **10 domain-specific test questions** across all KB topics
- **2 red-team tests** — out-of-scope query and false-premise adversarial question
- **Multi-turn memory test** — 3-question sequence to verify context persistence
- **RAGAS baseline evaluation** — faithfulness, answer relevancy, context precision

---

## 📊 State Design

```python
class CapstoneState(TypedDict):
    question: str           # Current user question
    messages: List[dict]    # Conversation history (sliding window)
    route: str              # retrieve | tool | memory_only
    retrieved: str          # RAG context from ChromaDB
    sources: List[str]      # Topic names of retrieved chunks
    tool_result: str        # Output from tool_node
    answer: str             # Final LLM answer
    faithfulness: float     # Eval score (0.0 – 1.0)
    eval_retries: int       # Retry counter (max 2)
    user_name: Optional[str]  # Extracted from conversation
```

---

## 🔒 Grounding Rules

The agent strictly follows these rules in its system prompt:

1. Answer **only** from the knowledge base or tool result
2. If information is not in context: *"I don't have that info. Contact support@shopeasy.in or call 1800-XXX-XXXX."*
3. Never fabricate policies, prices, or products
4. On eval retry, instructions are escalated to be even more strictly grounded

---

## 🔮 Future Improvements

- **Persistent ChromaDB** — Replace in-memory with a disk-backed collection so the KB survives restarts
- **Live web search tool** — Integrate a search API for real-time order status and inventory lookup
- **Multilingual support** — Hindi/regional language queries via translation pre-processing
- **FastAPI backend** — Expose the agent as a REST API for integration with a mobile/web frontend
- **WhatsApp integration** — Connect via Twilio for WhatsApp-native customer support

---

## 👤 Author

**Arpan Singh**  
Agentic AI Hands-On Course — Capstone Project 2026  
Batch: [Your Batch/Roll Number]  
GitHub: [@arpanz](https://github.com/arpanz)

---

> Built with LangGraph + ChromaDB + Groq 🤖
