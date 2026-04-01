
import streamlit as st
import pandas as pd
import os
import tempfile
from PIL import Image

from similarity_engine import search_by_image, search_by_text
from config import IMAGE_FOLDER, DATA_PATH

st.set_page_config(layout="wide")
st.title("🛒 Smart Retail App")

if "cart" not in st.session_state:
    st.session_state.cart = []

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔍 Search")

mode = st.sidebar.radio("Mode", ["Text", "Image"])

gender_filter = st.sidebar.selectbox(
    "Gender Filter",
    ["None", "Men", "Women"]
)

if gender_filter == "None":
    gender_filter = None

results = []

if mode == "Text":
    query = st.sidebar.text_input("Search")
    if st.sidebar.button("Search"):
        results = search_by_text(query, gender_filter=gender_filter)

elif mode == "Image":
    uploaded = st.sidebar.file_uploader("Upload")

    if uploaded:
        img = Image.open(uploaded)
        st.sidebar.image(img)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = search_by_image(tmp.name, gender_filter=gender_filter)

# =========================
# RESULTS OR CATALOG
# =========================
if results:
    st.subheader("🔍 Top Matches")

    cols = st.columns(2)

    for i, item in enumerate(results):
        with cols[i]:
            st.image(item['image_path'], width=200)
            st.write(item['name'])
            st.write(f"₹ {item['price']}")

            score = max(0, min(1, item['score']))
            st.progress(score)
            st.caption(f"{int(score*100)}% match")

            if st.button("Add to Cart", key=f"res_{item['id']}_{i}"):
                cart_item = {
                    "id": item['id'],
                    "name": item['name'],
                    "price": item['price'],
                    "image_path": item['image_path']
                }

                if cart_item not in st.session_state.cart:
                    st.session_state.cart.append(cart_item)

else:
    st.subheader("🛍️ All Products")

    cols = st.columns(4)

    for i, row in df.iterrows():
        with cols[i % 4]:
            img_path = os.path.join(IMAGE_FOLDER, f"{row['id']}.jpg")

            if os.path.exists(img_path):
                st.image(img_path, width=120)

            st.write(row['productDisplayName'][:40])
            st.write(f"₹ {row['price']}")

            if st.button("Add to Cart", key=f"cat_{row['id']}"):
                cart_item = {
                    "id": int(row['id']),
                    "name": row['productDisplayName'],
                    "price": row['price'],
                    "image_path": img_path
                }

                if cart_item not in st.session_state.cart:
                    st.session_state.cart.append(cart_item)

# =========================
# CART
# =========================
st.sidebar.subheader("🛒 Cart")

total = 0
for item in st.session_state.cart:
    st.sidebar.write(f"{item['name']} - ₹{item['price']}")
    total += item['price']

st.sidebar.write(f"Total: ₹{total}")

# =========================
# BILL
# =========================
def generate_bill(cart):
    total = sum(i['price'] for i in cart)

    items = "\n".join(
        [f"• {i['name']} - ₹{i['price']}" for i in cart]
    )

    return f"""
🧾 Invoice

Items:
{items}

---------------------
Total: ₹{total}
GST: ₹{int(total*0.05)}

Final: ₹{int(total*1.05)}

Thank you! 🛍️
"""

if st.sidebar.button("Generate Bill"):
    st.sidebar.text_area("Bill", generate_bill(st.session_state.cart), height=250)
