
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import IMAGE_FOLDER, DATA_PATH

_df: Optional[pd.DataFrame] = None

_vectorizer: Optional[TfidfVectorizer] = None
_text_index = None

_device = None
_model = None
_transform = None
_image_index = None
_image_embeddings = None


def _load_df() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df

    df = pd.read_csv(DATA_PATH).reset_index(drop=True)
    df["image_path"] = df["id"].astype(str) + ".jpg"
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(IMAGE_FOLDER, x))
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    _df = df
    return _df

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

def _ensure_text_corpus():
    df = _load_df()
    if "text" not in df.columns:
        df["text"] = df.apply(build_text, axis=1).str.lower()
    return df

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
def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


def _build_faiss_index(emb: np.ndarray):
    faiss = _try_import_faiss()
    if faiss is None:
        raise ImportError("faiss is not installed (required for similarity search).")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def _init_text():
    global _vectorizer, _text_index
    if _vectorizer is not None and _text_index is not None:
        return

    df = _ensure_text_corpus()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    text_embeddings = tfidf_matrix.toarray().astype("float32")

    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    text_embeddings /= norms

    _vectorizer = vectorizer
    _text_index = _build_faiss_index(text_embeddings)


def _init_image():
    """
    Lazy init so the app can still start even if torch/torchvision aren't installed.
    """
    global _device, _model, _transform, _image_index, _image_embeddings
    if _image_index is not None and _image_embeddings is not None:
        return

    try:
        import torch  # type: ignore
        import torchvision.models as models  # type: ignore
        import torchvision.transforms as transforms  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Image search dependencies missing. Install torch + torchvision + pillow to enable image search."
        ) from exc

    df = _load_df()
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(_device)
    _model = model

    _transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def encode_image(path: str):
        try:
            img = Image.open(path).convert("RGB")
            img = _transform(img).unsqueeze(0).to(_device)
            with torch.no_grad():
                feat = _model(img)
            feat = feat.squeeze().cpu().numpy()
            feat = feat / np.linalg.norm(feat)
            return feat.astype("float32")
        except Exception:
            return None

    image_embeddings = np.array([encode_image(p) for p in df["image_path"]])
    image_embeddings = image_embeddings.astype("float32")

    _image_embeddings = image_embeddings
    _image_index = _build_faiss_index(image_embeddings)

# =========================
# TEXT SEARCH
# =========================
def search_by_text(query, top_k=2, gender_filter=None):

    if not query:
        return []

    _init_text()

    query = normalize_query(query)

    auto_gender = detect_gender_from_query(query)
    if not gender_filter:
        gender_filter = auto_gender

    query_vec = _vectorizer.transform([query]).toarray().astype("float32")

    norm = np.linalg.norm(query_vec)
    if norm != 0:
        query_vec /= norm

    scores, indices = _text_index.search(query_vec, 20)

    results = []
    for i, score in zip(indices[0], scores[0]):
        row = _load_df().iloc[i]

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

    try:
        _init_image()
    except ImportError:
        # App should not crash if image deps aren't installed; just return no results.
        return []

    # Re-encode the query image using the initialized model/transform.
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        img = _transform(img).unsqueeze(0).to(_device)
        with torch.no_grad():
            feat = _model(img)
        emb = feat.squeeze().cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        emb = emb.astype("float32").reshape(1, -1)
    except Exception:
        return []

    scores, indices = _image_index.search(emb, 20)

    results = []
    for i, score in zip(indices[0], scores[0]):
        row = _load_df().iloc[i]

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
