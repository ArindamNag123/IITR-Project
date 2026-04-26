import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Catalog (CSV is named styles.csv under data/)
DATA_PATH = os.path.join(BASE_DIR, "data", "styles.csv")

# Product images: {id}.jpg — repo ships under Images/Train (and Images/Test for some ids)
IMAGE_FOLDER = os.path.join(BASE_DIR, "Images", "Train")
