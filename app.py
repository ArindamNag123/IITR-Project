import os
import tempfile
import uuid
import logging
from datetime import datetime

import pandas as pd
import streamlit as st
import redis
from PIL import Image
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage as LCMessage
from dotenv import load_dotenv

from config import DATA_PATH, IMAGE_FOLDER
from similarity_engine import search_by_image, search_by_text

load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FalkorDB Connection ---
r = redis.Redis(
    host=os.getenv("FALKORDB_HOST"),
    port=int(os.getenv("FALKORDB_PORT", 6379)),
    username=os.getenv("FALKORDB_USER"),
    password=os.getenv("FALKORDB_PASS"),
    decode_responses=True
)

try:
    r.ping()
    logger.info("✅ Connected to FalkorDB")
except Exception as e:
    logger.error(f"❌ FalkorDB Connection Error: {e}")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Smart Retail App")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

st.session_state.setdefault("cart", [])
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("chat_history_en", [])
st.session_state.setdefault("last_user_language", "en")
st.session_state.setdefault("invoice", None)
st.session_state.setdefault("show_invoice", False)

# 🔐 Secure API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Cached resources (loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_chat_graph():
    from chatbot.chatbot_controller import retail_graph
    return retail_graph


@st.cache_data
def load_product_catalog():
    return pd.read_csv(DATA_PATH)


graph = load_chat_graph()
catalog = load_product_catalog()


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def apply_styles():
    st.markdown("""
    <style>
    /* ── Launcher (bottom-right), like typical web chat widgets ── */
    .chat-fab {
        position: fixed; bottom: 1.5rem; right: 1.5rem;
        width: 56px; height: 56px; border-radius: 50%;
        background: linear-gradient(145deg, #4f46e5 0%, #7c3aed 100%);
        color: white; font-size: 1.5rem; line-height: 1;
        text-decoration: none; display: flex;
        align-items: center; justify-content: center;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.45), 0 4px 6px -2px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        z-index: 99999;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .chat-fab:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px -6px rgba(79, 70, 229, 0.5);
        color: white; text-decoration: none;
    }
    .chat-fab.open {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
        font-size: 1.35rem;
    }

    /* ── Single bordered chat shell (container with border=True) ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        background: #f8fafc !important;
        box-shadow:
            0 4px 6px -1px rgba(15, 23, 42, 0.06),
            0 10px 24px -8px rgba(15, 23, 42, 0.12) !important;
        overflow: hidden !important;
        min-height: min(72vh, 620px) !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Header strip inside the widget */
    .chat-widget-header-bar {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 55%, #8b5cf6 100%);
        color: #fff;
        margin: -1px -1px 0 -1px;
        padding: 0.85rem 1rem 0.75rem 1rem;
        border-radius: 13px 13px 0 0;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }
    .chat-widget-header-bar .chat-title {
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.01em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .chat-widget-header-bar .chat-sub {
        font-size: 0.78rem;
        opacity: 0.92;
        margin: 0.2rem 0 0 0;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .chat-status-dot {
        width: 7px; height: 7px;
        background: #4ade80;
        border-radius: 50%;
        box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.35);
        flex-shrink: 0;
    }

    /* Scrollable message region feel */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        flex: 1 1 auto;
        min-height: 0;
    }

    /* Message list — breathing room inside the card */
    [data-testid="stChatMessage"] {
        padding: 0.35rem 0.5rem !important;
    }

    /* Input area — footer strip inside the widget */
    [data-testid="stChatInput"] {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        background: #fff !important;
    }
    [data-testid="stChatInputContainer"] {
        padding: 0.65rem 0.75rem 0.85rem !important;
        background: #eef2f7 !important;
        border-top: 1px solid #e2e8f0 !important;
        margin: 0 -1px -1px -1px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — search controls + cart + bill
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.header("Search")
    mode = st.sidebar.radio("Mode", ["Text", "Image"])
    gender_filter = st.sidebar.selectbox("Gender Filter", ["None", "Men", "Women"])
    gender_filter = None if gender_filter == "None" else gender_filter

    results = []

    if mode == "Text":
        query = st.sidebar.text_input("Search")
        if st.sidebar.button("Search"):
            results = search_by_text(query, gender_filter=gender_filter)

    elif mode == "Image":
        uploaded = st.sidebar.file_uploader("Upload image")
        if uploaded:
            img = Image.open(uploaded)
            st.sidebar.image(img)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                results = search_by_image(tmp.name, gender_filter=gender_filter)

    render_cart()

    return results


def render_cart():
    st.sidebar.subheader("Cart")
    total = 0
    for i, item in enumerate(st.session_state.cart):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.write(f"{item['name']} — ₹{item['price']}")
            total += item["price"]
        with col2:
            if st.sidebar.button("❌", key=f"remove_{item['id']}_{i}"):
                st.session_state.cart.pop(i)
                st.rerun()
    st.sidebar.write(f"**Total: ₹{total}**")

    if st.sidebar.button("🛍️ Buy Now"):
        if not st.session_state.cart:
            st.sidebar.warning("Cart is empty")
        else:
            order = create_order(st.session_state.cart)
            with st.spinner("Generating AI invoice..."):
                invoice = generate_invoice(order)
            
            save_order(order)
            st.session_state.invoice = invoice
            st.session_state.show_invoice = True
            st.session_state.cart = []
            st.sidebar.success("Order placed!")

def create_order(cart, user="Guest"):
    order_id = str(uuid.uuid4())[:8]
    invoice_no = f"INV-{int(datetime.now().timestamp())}"
    timestamp = datetime.now()
    products = [i["name"] for i in cart]
    prices = [i["price"] for i in cart]
    total = sum(prices)
    return {
        "user_name": user,
        "order_id": order_id,
        "invoice_no": invoice_no,
        "date": timestamp,
        "products_list": products,
        "prices_list": prices,
        "total_price": total
    }

def generate_invoice(order):
    items_table = "\n".join([
        f"{i+1}. {p} - ₹{pr}"
        for i, (p, pr) in enumerate(zip(order["products_list"], order["prices_list"]))
    ])
    gst = int(order["total_price"] * 0.05)
    final = int(order["total_price"] * 1.05)
    prompt = f"""
    Create a professional retail invoice.
    Include:
    - Store Name: Smart Retail AI
    - Invoice Number
    - Order ID
    - Date
    - Customer Name
    - Itemized list
    - Subtotal, GST (5%), Final Total
    - Friendly thank-you message
    DATA:
    Invoice No: {order['invoice_no']}
    Order ID: {order['order_id']}
    Date: {order['date']}
    Customer: {order['user_name']}
    Items:
    {items_table}
    Subtotal: ₹{order['total_price']}
    GST: ₹{gst}
    Final: ₹{final}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def save_order(order):
    logger.info(f"FETCHING DATA BEFORE SAVE: Order ID: {order['order_id']}, Total: ₹{order['total_price']}")
    logger.info(f"Attempting to save Order {order['invoice_no']} to FalkorDB...")
    
    try:
        # Create a graph query for the Invoice node
        query = f"""
        CREATE (:Invoice {{
            invoiceNumber: '{order['invoice_no']}',
            orderID: '{order['order_id']}',
            date: '{order['date'].strftime('%Y-%m-%d')}',
            customerName: '{order['user_name']}',
            itemizedList: '{str(order["products_list"]).replace("'", '"')}',
            subtotal: {order['total_price']},
            gst: {int(order['total_price'] * 0.05)}, 
            finalTotal: {int(order['total_price'] * 1.05)}
        }})
        """
        r.execute_command("GRAPH.QUERY", "products", query)
        logger.info("✅ Data saved successfully to FalkorDB!")

        # Log check: Retrieve the data we just saved
        check_query = f"MATCH (i:Invoice {{invoiceNumber: '{order['invoice_no']}'}}) RETURN i.invoiceNumber, i.customerName, i.finalTotal"
        retrieved = r.execute_command("GRAPH.QUERY", "products", check_query)
        logger.info(f"🔍 READ CHECK (Retrieved from DB): {retrieved}")

    except Exception as e:
        logger.error(f"❌ FalkorDB Error during save: {e}")

@st.dialog("🧾 Invoice")
def show_invoice_popup():
    st.text_area("Invoice", st.session_state.invoice, height=400)
    st.download_button(
        "⬇️ Download Invoice",
        st.session_state.invoice,
        file_name="invoice.txt"
    )
    if st.button("Close"):
        st.session_state.show_invoice = False
        st.rerun()


def build_bill(cart: list) -> str:
    subtotal = sum(item["price"] for item in cart)
    line_items = "\n".join(f"• {item['name']} — ₹{item['price']}" for item in cart)
    return (
        f"Invoice\n\nItems:\n{line_items}\n\n"
        f"{'─' * 25}\n"
        f"Subtotal : ₹{subtotal}\n"
        f"GST (5%) : ₹{int(subtotal * 0.05)}\n"
        f"Total    : ₹{int(subtotal * 1.05)}\n\n"
        f"Thank you for shopping!"
    )


# ---------------------------------------------------------------------------
# Product catalog panel
# ---------------------------------------------------------------------------

def render_search_results(results: list, chat_open: bool):
    st.subheader("Top Matches")
    cols = st.columns(min(len(results), 2))
    for i, item in enumerate(results):
        with cols[i % 2]:
            st.image(item["image_path"], width=180)
            st.write(item["name"])
            st.write(f"₹ {item['price']}")
            score = max(0.0, min(1.0, item["score"]))
            st.progress(score)
            st.caption(f"{int(score * 100)}% match")
            if st.button("Add to Cart", key=f"res_{item['id']}_{i}"):
                add_to_cart(item)


def render_full_catalog(chat_open: bool):
    st.subheader("All Products")
    n_cols = 2 if chat_open else 4
    cols = st.columns(n_cols)
    for i, row in catalog.iterrows():
        with cols[i % n_cols]:
            img_path = os.path.join(IMAGE_FOLDER, f"{row['id']}.jpg")
            if os.path.exists(img_path):
                st.image(img_path, width=120)
            st.write(row["productDisplayName"][:38])
            st.write(f"₹ {row['price']}")
            if st.button("Add to Cart", key=f"cat_{row['id']}"):
                add_to_cart({
                    "id": int(row["id"]),
                    "name": row["productDisplayName"],
                    "price": row["price"],
                    "image_path": img_path,
                })


def add_to_cart(item: dict):
    existing_ids = [i["id"] for i in st.session_state.cart]
    if item["id"] not in existing_ids:
        cart_item = {k: item[k] for k in ("id", "name", "price", "image_path")}
        st.session_state.cart.append(cart_item)
        st.toast("Added to cart ✅")


# ---------------------------------------------------------------------------
# Chat panel
# ---------------------------------------------------------------------------

def render_chat_button(chat_open: bool):
    if chat_open:
        st.markdown(
            '<a class="chat-fab open" href="/?chat=" title="Close chatbot">✕</a>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<a class="chat-fab" href="/?chat=open" title="Open Smart Retail Chatbot">💬</a>',
            unsafe_allow_html=True,
        )


def render_chat_panel():
    """Contained chat widget: card shell + header bar + messages + input (support-chat style)."""
    with st.container(border=True):
        st.markdown(
            """
            <div class="chat-widget-header-bar">
                <p class="chat-title">🛍️ Smart Retail Assistant</p>
                <p class="chat-sub">
                    <span class="chat-status-dot" aria-hidden="true"></span>
                    We’re here to help — products, orders, billing &amp; more
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _head, _clear = st.columns([4, 2])
        with _clear:
            if st.button("Clear conversation", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.chat_history_en = []
                st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Message…")
        if user_input:
            handle_user_message(user_input)


def handle_user_message(user_input: str):
    # Display history stays in the user's language.
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Internal history is kept in English so the supervisor + all agents work in one language.
    from chatbot.translator_module import ChatbotTranslator

    translator = ChatbotTranslator()
    # STEP 1: translate user → English (best-effort; never block the graph)
    try:
        tr = translator.translate_to_english(user_input)
        st.session_state.last_user_language = tr.detected_language or "en"
        user_input_en = tr.english
    except Exception:
        st.session_state.last_user_language = "en"
        user_input_en = user_input

    st.session_state.chat_history_en.append({"role": "user", "content": user_input_en})

    lc_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else LCMessage(content=m["content"])
        for m in st.session_state.chat_history_en
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = graph.invoke({
                    "messages": lc_messages,
                    "next_agent": "",
                    "intent": "",
                    # We pass what we detected from the *original* user input.
                    # Agents/supervisor can use it for optional language-aware behavior.
                    "detected_language": st.session_state.last_user_language,
                    "rag_context": None,
                    "metadata": {},
                })
                agent_reply_en = result["messages"][-1].content
                routed_to = result.get("next_agent", "?")

                # Store the English reply for next-turn context.
                st.session_state.chat_history_en.append({"role": "assistant", "content": agent_reply_en})

                # STEP 3: translate English reply → user's language (best-effort).
                try:
                    agent_reply = translator.translate_from_english(
                        agent_reply_en,
                        target_language=st.session_state.last_user_language,
                        reference_text=user_input,
                    )
                except Exception:
                    agent_reply = agent_reply_en

                st.markdown(agent_reply)
                st.caption(f"→ **{routed_to} agent**")

            except Exception as exc:
                agent_reply = f"Something went wrong: {exc}"
                st.error(agent_reply)

    st.session_state.chat_history.append({"role": "assistant", "content": agent_reply})
    st.rerun()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    apply_styles()

    chat_open = st.query_params.get("chat", "") == "open"
    render_chat_button(chat_open)

    st.title("Smart Retail App")

    if chat_open:
        col_products, col_chat = st.columns([6, 4], gap="medium")
    else:
        col_products = st.container()
        col_chat = None

    with col_products:
        results = render_sidebar()
        if results:
            render_search_results(results, chat_open)
        else:
            render_full_catalog(chat_open)

    if chat_open and col_chat is not None:
        with col_chat:
            render_chat_panel()


    if st.session_state.show_invoice:
        show_invoice_popup()

main()
