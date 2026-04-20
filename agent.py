import os
from typing import TypedDict, List, Optional
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ── State ─────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: Optional[str]

MAX_EVAL_RETRIES = 2

# ── Knowledge Base ────────────────────────────────────────────────────────────
DOCUMENTS = [
    {"id": "doc_001", "topic": "Return Policy",
     "text": "ShopEasy allows returns within 30 days of delivery for most items. To initiate a return, visit My Orders, select the item, and click Return. Items must be unused, unwashed, and in original packaging with all tags intact. Electronics must be returned within 10 days of delivery. Perishable goods, digital downloads, and customised items are non-returnable. Once the returned item is received and inspected, a refund is processed within 5-7 business days. Refunds are credited to the original payment method. For Cash on Delivery orders, the refund is issued as store credit or a bank transfer within 7 business days."},
    {"id": "doc_002", "topic": "Shipping Policy",
     "text": "ShopEasy offers free standard shipping on orders above Rs 499. Standard delivery takes 5-7 business days. Express delivery (1-2 business days) is available for an additional charge of Rs 99. Same-day delivery is available in select metro cities including Bangalore, Mumbai, Delhi, Hyderabad, Chennai, and Pune for orders placed before 12 PM. Orders are not shipped on Sundays and public holidays. International shipping is currently not available. Shipping charges for orders below Rs 499 are Rs 49 for standard and Rs 149 for express."},
    {"id": "doc_003", "topic": "Order Tracking",
     "text": "You can track your order in real time by visiting the My Orders section after logging in. A tracking link is also sent to your registered email and SMS within 24 hours of dispatch. Tracking information may take up to 24 hours to update after the order is shipped. If tracking shows delivered but you have not received the package, raise a complaint within 48 hours via the Help section. ShopEasy will investigate and resolve within 3 business days. Courier partners include BlueDart, Delhivery, Ekart, and DTDC depending on your location."},
    {"id": "doc_004", "topic": "Payment Methods",
     "text": "ShopEasy accepts multiple payment methods: UPI (PhonePe, GPay, Paytm), Credit Cards (Visa, Mastercard, Amex), Debit Cards, Net Banking (all major banks), EMI on credit cards for orders above Rs 3000, and Cash on Delivery (COD) for orders up to Rs 10,000. Buy Now Pay Later (BNPL) is available via LazyPay and Simpl. All transactions are secured with 256-bit SSL encryption. Payment failures are auto-reversed within 3-5 business days. ShopEasy Wallet is available for faster checkout with cashback benefits."},
    {"id": "doc_005", "topic": "Order Cancellation",
     "text": "Orders can be cancelled before they are dispatched. To cancel, go to My Orders, select the order, and click Cancel Order. If the order has already been dispatched, cancellation is not possible and you must wait for delivery and then initiate a return. Refunds for cancelled orders are processed within 3-5 business days. Prepaid orders are refunded to the original payment method. COD orders that are cancelled before dispatch incur no charges."},
    {"id": "doc_006", "topic": "Discount and Promo Codes",
     "text": "ShopEasy offers promo codes during seasonal sales like Big Billion Days, End of Season Sale, and Festive Sales. Promo codes can be applied at checkout in the Apply Coupon field. Only one promo code can be applied per order. First-time users get a flat 10% off using code WELCOME10 with a maximum discount of Rs 200. Referral codes give both referrer and referee Rs 100 ShopEasy Wallet credit once the referee places their first order above Rs 500."},
    {"id": "doc_007", "topic": "Product Warranty",
     "text": "Warranty terms depend on the product category and brand. Electronics carry a standard 1-year manufacturer warranty. Large appliances such as refrigerators, washing machines, and ACs carry a 1-year comprehensive warranty and up to 5-year warranty on specific parts like the compressor. Warranty claims must be raised directly with the brand service centre. Warranty does not cover physical damage, water damage, or damage from misuse. Extended warranty plans are available for purchase on select electronics."},
    {"id": "doc_008", "topic": "Account and Login Issues",
     "text": "If you cannot log in, first try resetting your password using Forgot Password on the login page. A reset link will be sent to your registered email within 2 minutes. Accounts are locked after 5 failed login attempts for security and you must wait 30 minutes or contact support. To change your registered mobile number, go to Profile then Edit then Mobile and verify with OTP. For account deletion requests, email privacy@shopeasy.in with the subject Account Deletion Request."},
    {"id": "doc_009", "topic": "Cash on Delivery",
     "text": "Cash on Delivery is available for orders up to Rs 10,000. COD is not available for digital products, pre-order items, and certain high-value electronics. COD orders incur an additional handling fee of Rs 25. If a COD order is refused at delivery without a valid reason more than twice in 6 months, COD option may be temporarily disabled for the account. Refunds for returned COD items are issued as ShopEasy Wallet credit or bank transfer after providing bank details to the support team."},
    {"id": "doc_010", "topic": "Exchange Policy",
     "text": "Exchanges are available for clothing and footwear within 30 days of delivery for size or colour issues. To request an exchange, go to My Orders and select Exchange. Exchange is subject to availability of the requested size or colour. If the desired variant is unavailable, a full refund will be issued instead. Electronics and appliances are not eligible for exchange and only return and refund applies. Only one exchange per order item is allowed."}
]


def build_app():
    """Build and return the compiled LangGraph app. Call once and cache."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="shopeasy_faq")
    texts = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    metadatas = [{"topic": d["topic"]} for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)

    def _retrieve(question, n=3):
        q_emb = embedder.encode([question]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=n)
        return [{"topic": m["topic"], "text": d}
                for d, m in zip(results["documents"][0], results["metadatas"][0])]

    # ── Nodes ─────────────────────────────────────────────────────────────────
    def memory_node(state: CapstoneState) -> CapstoneState:
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        msgs = msgs[-6:]
        uname = state.get("user_name")
        lq = state["question"].lower()
        if "my name is" in lq:
            parts = lq.split("my name is")
            if len(parts) > 1:
                uname = parts[1].strip().split()[0].capitalize()
        return {**state, "messages": msgs, "user_name": uname,
                "eval_retries": state.get("eval_retries", 0)}

    def router_node(state: CapstoneState) -> CapstoneState:
        prompt = (
            f"Route this e-commerce support question into ONE word:\n"
            f"- retrieve: return/shipping/payment/tracking/cancellation/discount/warranty/account/COD/exchange\n"
            f"- tool: needs today's date or delivery date calculation\n"
            f"- memory_only: greeting/thanks/smalltalk/user asking their own name\n\n"
            f"Question: {state['question']}\nReply ONE word only."
        )
        r = llm.invoke(prompt).content.strip().lower().split()[0]
        if r not in ["retrieve", "tool", "memory_only"]:
            r = "retrieve"
        return {**state, "route": r}

    def retrieval_node(state: CapstoneState) -> CapstoneState:
        chunks = _retrieve(state["question"])
        context = "\n\n".join(f"[{c['topic']}]\n{c['text']}" for c in chunks)
        sources = [c["topic"] for c in chunks]
        return {**state, "retrieved": context, "sources": sources, "tool_result": ""}

    def skip_node(state: CapstoneState) -> CapstoneState:
        return {**state, "retrieved": "", "sources": [], "tool_result": ""}

    def tool_node(state: CapstoneState) -> CapstoneState:
        try:
            today = datetime.now()
            q = state["question"].lower()
            parts = [f"Today is {today.strftime('%A, %d %B %Y')}."]
            if any(w in q for w in ["deliver", "arrive", "reach", "when will"]):
                std = (today + timedelta(days=7)).strftime("%d %B %Y")
                exp = (today + timedelta(days=2)).strftime("%d %B %Y")
                parts.append(
                    f"Standard delivery estimate: {std} (5-7 business days). "
                    f"Express delivery estimate: {exp} (1-2 business days)."
                )
            result = " ".join(parts)
        except Exception as e:
            result = f"Could not compute date: {str(e)}"
        return {**state, "tool_result": result, "retrieved": "", "sources": []}

    def answer_node(state: CapstoneState) -> CapstoneState:
        uname = state.get("user_name")
        name_part = f" Address customer as {uname}." if uname else ""
        retry_instr = "\nBe strictly grounded. Use ONLY provided context." if state.get("eval_retries", 0) > 0 else ""

        ctx = ""
        if state.get("retrieved"):
            ctx += f"\n\nKNOWLEDGE BASE:\n{state['retrieved']}"
        if state.get("tool_result"):
            ctx += f"\n\nTOOL RESULT:\n{state['tool_result']}"

        history = "".join(
            f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
            for m in state.get("messages", [])[:-1]
        )

        system = (
            f"You are ShopEasy customer support assistant.{name_part}\n"
            f"RULES: 1. Answer ONLY from KNOWLEDGE BASE or TOOL RESULT below. "
            f"2. If not in context, say: 'I don't have that info. Contact support@shopeasy.in or call 1800-XXX-XXXX.' "
            f"3. Do NOT fabricate policies, prices, or products.{retry_instr}\n\n"
            f"CONVERSATION HISTORY:\n{history or 'None'}\n{ctx}"
        )
        ans = llm.invoke(f"{system}\n\nCustomer: {state['question']}\nAssistant:").content.strip()
        return {**state, "answer": ans}

    def eval_node(state: CapstoneState) -> CapstoneState:
        if not state.get("retrieved", "").strip():
            return {**state, "faithfulness": 1.0}
        try:
            resp = llm.invoke(
                f"Rate faithfulness 0.0-1.0 (how well grounded is this answer in context?).\n"
                f"Context: {state['retrieved'][:800]}\nAnswer: {state['answer']}\nReply decimal only."
            )
            score = max(0.0, min(1.0, float(resp.content.strip().split()[0])))
        except Exception:
            score = 0.75
        retries = state.get("eval_retries", 0)
        if score < 0.7 and retries < MAX_EVAL_RETRIES:
            return {**state, "faithfulness": score, "eval_retries": retries + 1}
        return {**state, "faithfulness": score}

    def save_node(state: CapstoneState) -> CapstoneState:
        msgs = state.get("messages", []) + [{"role": "assistant", "content": state.get("answer", "")}]
        return {**state, "messages": msgs}

    # ── Routing functions ──────────────────────────────────────────────────────
    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        return "tool" if r == "tool" else "skip" if r == "memory_only" else "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) <= MAX_EVAL_RETRIES:
            return "answer"
        return "save"

    # ── Graph assembly ─────────────────────────────────────────────────────────
    g = StateGraph(CapstoneState)
    for name, fn in [
        ("memory", memory_node), ("router", router_node),
        ("retrieve", retrieval_node), ("skip", skip_node),
        ("tool", tool_node), ("answer", answer_node),
        ("eval", eval_node), ("save", save_node)
    ]:
        g.add_node(name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges("router", route_decision,
                             {"retrieve": "retrieve", "tool": "tool", "skip": "skip"})
    for src in ["retrieve", "tool", "skip"]:
        g.add_edge(src, "answer")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    g.add_edge("save", END)

    return g.compile(checkpointer=MemorySaver())


def ask_agent(app, question: str, thread_id: str) -> dict:
    """Invoke the agent with a question."""
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "question": question, "messages": [], "route": "",
        "retrieved": "", "sources": [], "tool_result": "",
        "answer": "", "faithfulness": 0.0, "eval_retries": 0, "user_name": None
    }
    return app.invoke(initial, config=config)
