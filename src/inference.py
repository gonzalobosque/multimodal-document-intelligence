import os
from pathlib import Path

# Prevent Transformers from loading its TensorFlow integration.
os.environ["USE_TF"] = "0"

import json

import joblib
import numpy as np
import pandas as pd
import pytesseract
import torch

from PIL import Image
from sentence_transformers import SentenceTransformer
from tensorflow import keras
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_DIR, EMBEDDING_MODEL_ID


TEXT_MODEL_DIR = DATA_DIR / "text_model"
IMAGE_MODEL_PATH = DATA_DIR / "image_model" / "model.keras"
FUSION_MODEL_PATH = DATA_DIR / "fusion_logreg.joblib"
CLASS_ORDER_PATH = DATA_DIR / "class_order.json"
CATALOG_PATH = DATA_DIR / "documents.parquet"


def _require_path(path: Path) -> None:
    """Fail early with a clear message when a required artifact is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact not found: {path}"
        )


for required_path in (
    TEXT_MODEL_DIR,
    IMAGE_MODEL_PATH,
    FUSION_MODEL_PATH,
    CLASS_ORDER_PATH,
):
    _require_path(required_path)


tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)

text_model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL_DIR
)
text_model.eval()

image_model = keras.models.load_model(
    IMAGE_MODEL_PATH,
    compile=False,
)

fusion_model = joblib.load(FUSION_MODEL_PATH)


# Preserve the exact class order used during training and late fusion.
with CLASS_ORDER_PATH.open(encoding="utf-8") as class_file:
    class_names = json.load(class_file)


# The embedding model is downloaded and cached automatically when necessary.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

if CATALOG_PATH.exists():
    catalog = pd.read_parquet(CATALOG_PATH)
else:
    catalog = pd.DataFrame(
        columns=[
            "document_id",
            "filename",
            "source_path",
            "s3_key",
            "text",
            "label_name",
            "predicted_class",
            "confidence",
            "embedding",
        ]
    )

# Catalog rows and embedding rows must always remain aligned.
if len(catalog) > 0:
    embeddings = np.stack(catalog["embedding"].to_numpy())
else:
    embeddings = np.empty(
        (
            0,
            embedding_model.get_sentence_embedding_dimension(),
        ),
        dtype=np.float32,
    )


def classify_document(
    document_path: str | Path,
) -> tuple[str, str, float]:
    """Run OCR and multimodal classification for a document image."""

    document_path = Path(document_path)

    with Image.open(document_path) as document_image:
        document_text = pytesseract.image_to_string(
            document_image,
            lang="eng",
        ).strip()

    # Match the DistilBERT preprocessing used during training and evaluation.
    tokens = tokenizer(
        document_text,
        truncation=True,
        max_length=512,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = text_model(**tokens).logits
        text_probabilities = (
            torch.softmax(logits, dim=1)
            .cpu()
            .numpy()
        )

    model_image = keras.utils.load_img(
        document_path,
        target_size=(384, 384),
        color_mode="rgb",
    )

    # EfficientNetB0 already includes the preprocessing expected by the model.
    image_array = keras.utils.img_to_array(model_image)
    image_batch = np.expand_dims(image_array, axis=0)

    image_probabilities = image_model.predict(
        image_batch,
        verbose=False,
    )

    # Late fusion expects text probabilities followed by image probabilities.
    fusion_features = np.concatenate(
        [text_probabilities, image_probabilities],
        axis=1,
    )

    predicted_id = int(
        fusion_model.predict(fusion_features)[0]
    )

    fusion_probabilities = fusion_model.predict_proba(
        fusion_features
    )[0]

    # predict_proba columns follow fusion_model.classes_, not array indices.
    class_positions = np.flatnonzero(
        fusion_model.classes_ == predicted_id
    )

    if len(class_positions) != 1:
        raise RuntimeError(
            f"Predicted class {predicted_id} is missing from "
            "fusion_model.classes_."
        )

    confidence = float(
        fusion_probabilities[class_positions[0]]
    )

    predicted_class = class_names[predicted_id]

    return document_text, predicted_class, confidence