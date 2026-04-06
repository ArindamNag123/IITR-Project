import os
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage as LCMessage

from config import DATA_PATH, IMAGE_FOLDER
from similarity_engine import search_by_image, search_by_text

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Smart Retail App")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

st.session_state.setdefault("cart", [])
st.session_state.setdefault("chat_history", [])


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
    /* Floating chat button (bottom-right) */
    .chat-fab {
        position: fixed; bottom: 2rem; right: 2rem;
        width: 62px; height: 62px; border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-size: 1.75rem; line-height: 1;
        text-decoration: none; display: flex;
        align-items: center; justify-content: center;
        box-shadow: 0 4px 20px rgba(102,126,234,0.55);
        z-index: 99999; transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .chat-fab:hover {
        transform: scale(1.12);
        box-shadow: 0 6px 28px rgba(102,126,234,0.75);
        color: white; text-decoration: none;
    }
    .chat-fab.open { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }

    /* Chat panel header */
    .chat-panel-title {
        font-size: 1.05rem; font-weight: 700; color: #a78bfa;
        margin: 0 0 0.6rem 0; display: flex; align-items: center; gap: 0.4rem;
        border-bottom: 1px solid #0f3460; padding-bottom: 0.6rem;
    }

    /* Compact message bubbles */
    [data-testid="stChatMessage"] { padding: 0.4rem 0.6rem !important; }
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
    for item in st.session_state.cart:
        st.sidebar.write(f"{item['name']} — ₹{item['price']}")
        total += item["price"]
    st.sidebar.write(f"**Total: ₹{total}**")

    if st.sidebar.button("Generate Bill"):
        st.sidebar.text_area("Bill", build_bill(st.session_state.cart), height=260)


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
    cart_item = {k: item[k] for k in ("id", "name", "price", "image_path")}
    if cart_item not in st.session_state.cart:
        st.session_state.cart.append(cart_item)


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
    st.markdown(
        '<div class="chat-panel-title">🛍️ Smart Retail Assistant</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.chat_history:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    user_input = st.chat_input("Ask about products, orders, billing…")
    if user_input:
        handle_user_message(user_input)


def handle_user_message(user_input: str):
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    lc_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else LCMessage(content=m["content"])
        for m in st.session_state.chat_history
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = graph.invoke({
                    "messages": lc_messages,
                    "next_agent": "",
                    "intent": "",
                    "detected_language": "en",
                    "rag_context": None,
                    "metadata": {},
                })
                agent_reply = result["messages"][-1].content
                routed_to = result.get("next_agent", "?")

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


main()
