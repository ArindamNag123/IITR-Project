
import os
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from config import IMAGE_FOLDER, DATA_PATH

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH).reset_index(drop=True)

df['image_path'] = df['id'].astype(str) + ".jpg"
df['image_path'] = df['image_path'].apply(
    lambda x: os.path.join(IMAGE_FOLDER, x)
)

df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

# =========================
# TEXT PREPROCESSING
# =========================
def build_text(row):
    return " ".join([
        str(row['productDisplayName']) * 3,
        str(row['articleType']) * 3,
        str(row['masterCategory']) * 2,
        str(row['subCategory']) * 2,
        str(row['baseColour']) * 2,
        str(row['gender'])
    ])

df['text'] = df.apply(build_text, axis=1).str.lower()

# =========================
# QUERY NORMALIZATION
# =========================
def normalize_query(q):
    q = q.lower()

    synonyms = {
        "shoe": "shoes footwear sneakers",
        "shirt": "shirt tshirt tee",
        "pant": "pants trousers jeans",
        "kurta": "kurta ethnic"
    }

    for k, v in synonyms.items():
        if k in q:
            q += " " + v

    return q

# =========================
# GENDER MATCHING
# =========================
def gender_match(row_gender, query_gender):
    row_gender = str(row_gender).lower()
    query_gender = str(query_gender).lower()

    if query_gender == "men":
        return any(x in row_gender for x in ["men", "male", "boys", "unisex"])

    if query_gender == "women":
        return any(x in row_gender for x in ["women", "female", "girls", "unisex"])

    return True

# =========================
# AUTO GENDER DETECTION
# =========================
def detect_gender_from_query(query):
    query = query.lower()

    if any(x in query for x in ["women", "female", "girl", "ladies"]):
        return "women"

    if any(x in query for x in ["men", "male", "boy", "gents"]):
        return "men"

    return None

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['text'])

text_embeddings = tfidf_matrix.toarray().astype('float32')

norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
text_embeddings /= norms

# =========================
# IMAGE MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def encode_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img)

        feat = feat.squeeze().cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        return feat.astype('float32')
    except:
        return None

# =========================
# BUILD INDEX
# =========================
def build_index(emb):
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

print("Rebuilding text index...")
text_index = build_index(text_embeddings)

print("Building image embeddings...")
image_embeddings = np.array([
    encode_image(p) for p in df['image_path']
])

print("Rebuilding image index...")
image_index = build_index(image_embeddings)

# =========================
# TEXT SEARCH
# =========================
def search_by_text(query, top_k=2, gender_filter=None):

    if not query:
        return []

    query = normalize_query(query)

    auto_gender = detect_gender_from_query(query)
    if not gender_filter:
        gender_filter = auto_gender

    query_vec = vectorizer.transform([query]).toarray().astype('float32')

    norm = np.linalg.norm(query_vec)
    if norm != 0:
        query_vec /= norm

    scores, indices = text_index.search(query_vec, 20)

    results = []
    for i, score in zip(indices[0], scores[0]):
        row = df.iloc[i]

        if gender_filter and not gender_match(row['gender'], gender_filter):
            continue

        if "shoe" in query and "shoe" not in row['articleType'].lower():
            continue

        if "shirt" in query and "shirt" not in row['articleType'].lower():
            continue

        results.append({
            "id": int(row['id']),
            "name": row['productDisplayName'],
            "price": row['price'],
            "image_path": row['image_path'],
            "score": float(score)
        })

        if len(results) == top_k:
            break

    return results

# =========================
# IMAGE SEARCH
# =========================
def search_by_image(image_path, top_k=2, gender_filter=None):

    emb = encode_image(image_path)
    if emb is None:
        return []

    emb = emb.reshape(1, -1)

    scores, indices = image_index.search(emb, 20)

    results = []
    for i, score in zip(indices[0], scores[0]):
        row = df.iloc[i]

        if gender_filter and not gender_match(row['gender'], gender_filter):
            continue

        results.append({
            "id": int(row['id']),
            "name": row['productDisplayName'],
            "price": row['price'],
            "image_path": row['image_path'],
            "score": float(score)
        })

        if len(results) == top_k:
            break

    return results
