import logging
import shutil
import tarfile
import threading
from pathlib import Path
from uuid import uuid4

import boto3
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import (
    ARTIFACT_SOURCE,
    AWS_REGION,
    CATALOG_KEY,
    CLASS_ORDER_KEY,
    DATA_DIR,
    DOCUMENTS_PREFIX,
    FUSION_MODEL_KEY,
    IMAGE_MODEL_KEY,
    PROJECT_ROOT,
    S3_BUCKET,
    TEXT_MODEL_KEY,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TEXT_MODEL_DIR = DATA_DIR / "text_model"
IMAGE_MODEL_DIR = DATA_DIR / "image_model"
FUSION_MODEL_PATH = DATA_DIR / "fusion_logreg.joblib"
CLASS_ORDER_PATH = DATA_DIR / "class_order.json"
CATALOG_PATH = DATA_DIR / "documents.parquet"

UPLOADS_DIR = DATA_DIR / "uploads"
LOCAL_DOCUMENTS_DIR = DATA_DIR / "documents"

for directory in (
    DATA_DIR,
    TEXT_MODEL_DIR,
    IMAGE_MODEL_DIR,
    UPLOADS_DIR,
    LOCAL_DOCUMENTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


s3 = None


def _download_and_extract_model(
    s3_key: str,
    target_dir: Path,
    marker_file: Path,
) -> None:
    """Download and extract a model artifact when it is not cached locally."""

    if marker_file.exists():
        return

    archive_path = DATA_DIR / f"{target_dir.name}.tar.gz"

    logger.info("Downloading model artifact from S3: %s", s3_key)
    s3.download_file(
        S3_BUCKET,
        s3_key,
        str(archive_path),
    )

    try:
        with tarfile.open(archive_path, "r:gz") as model_archive:
            model_archive.extractall(
                target_dir,
                filter="data",
            )
    finally:
        archive_path.unlink(missing_ok=True)


def _prepare_artifacts() -> None:
    """Prepare model and catalog artifacts for the selected storage mode."""

    global s3

    if ARTIFACT_SOURCE == "s3":
        if not S3_BUCKET:
            raise RuntimeError(
                "S3_BUCKET must be configured when "
                "ARTIFACT_SOURCE='s3'."
            )

        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
        )

        _download_and_extract_model(
            TEXT_MODEL_KEY,
            TEXT_MODEL_DIR,
            TEXT_MODEL_DIR / "config.json",
        )

        _download_and_extract_model(
            IMAGE_MODEL_KEY,
            IMAGE_MODEL_DIR,
            IMAGE_MODEL_DIR / "model.keras",
        )

        s3.download_file(
            S3_BUCKET,
            FUSION_MODEL_KEY,
            str(FUSION_MODEL_PATH),
        )

        s3.download_file(
            S3_BUCKET,
            CLASS_ORDER_KEY,
            str(CLASS_ORDER_PATH),
        )

        # Refresh the catalog so S3 deployments start from its latest version.
        s3.download_file(
            S3_BUCKET,
            CATALOG_KEY,
            str(CATALOG_PATH),
        )

    else:
        logger.info(
            "Using local model and catalog artifacts from %s",
            DATA_DIR,
        )

    required_artifacts = (
        TEXT_MODEL_DIR / "config.json",
        IMAGE_MODEL_DIR / "model.keras",
        FUSION_MODEL_PATH,
        CLASS_ORDER_PATH,
    )

    missing_artifacts = [
        path for path in required_artifacts if not path.exists()
    ]

    if missing_artifacts:
        missing = "\n".join(
            f"- {path}" for path in missing_artifacts
        )
        raise RuntimeError(
            "Required application artifacts are missing:\n"
            f"{missing}"
        )


_prepare_artifacts()


# Import only after all inference artifacts are available.
from src.inference import (
    catalog,
    classify_document,
    embedding_model,
    embeddings,
)


app = FastAPI(title="Multimodal Document Intelligence")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

_catalog_lock = threading.Lock()


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Return a lightweight service health response."""

    return {"status": "ok"}


def _store_document(
    document_path: Path,
    document_id: str,
    filename: str,
):
    """Store an uploaded document using the selected storage backend."""

    if ARTIFACT_SOURCE == "s3":
        s3_key = (
            f"{DOCUMENTS_PREFIX.rstrip('/')}/"
            f"{document_id}/{filename}"
        )

        s3.upload_file(
            str(document_path),
            S3_BUCKET,
            s3_key,
        )

        return pd.NA, s3_key

    destination = (
        LOCAL_DOCUMENTS_DIR
        / document_id
        / filename
    )
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        document_path,
        destination,
    )

    relative_path = destination.relative_to(
        PROJECT_ROOT
    )

    return str(relative_path), pd.NA


def _persist_catalog(updated_catalog: pd.DataFrame) -> None:
    """Persist the catalog locally and optionally synchronize it with S3."""

    updated_catalog.to_parquet(
        CATALOG_PATH,
        index=False,
    )

    if ARTIFACT_SOURCE == "s3":
        s3.upload_file(
            str(CATALOG_PATH),
            S3_BUCKET,
            CATALOG_KEY,
        )


@app.post("/classify")
def classify(file: UploadFile = File(...)):
    """Classify an uploaded TIFF document and add it to the search catalog."""

    global catalog, embeddings

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in {".tif", ".tiff"}:
        raise HTTPException(
            status_code=400,
            detail="Only TIFF documents are supported.",
        )

    document_id = f"app-{uuid4().hex}"
    document_path = UPLOADS_DIR / f"{document_id}{suffix}"

    try:
        with document_path.open("wb") as uploaded_file:
            shutil.copyfileobj(
                file.file,
                uploaded_file,
            )

        document_text, predicted_class, confidence = classify_document(
            document_path
        )

        document_embedding = embedding_model.encode(
            document_text,
            normalize_embeddings=True,
        )

        with _catalog_lock:
            source_path, s3_key = _store_document(
                document_path,
                document_id,
                filename,
            )

            new_document = pd.DataFrame(
                [{
                    "document_id": document_id,
                    "filename": filename,
                    "source_path": source_path,
                    "s3_key": s3_key,
                    "text": document_text,
                    "label_name": pd.NA,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "embedding": document_embedding,
                }]
            )

            new_catalog = pd.concat(
                [catalog, new_document],
                ignore_index=True,
            )

            new_embeddings = np.stack(
                new_catalog["embedding"].to_numpy()
            )

            _persist_catalog(new_catalog)

            catalog = new_catalog
            embeddings = new_embeddings

        return {
            "document_id": document_id,
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "text_excerpt": document_text[:500],
            "catalog_size": len(catalog),
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Document classification failed."
        )
        raise HTTPException(
            status_code=500,
            detail="Document classification failed.",
        ) from None

    finally:
        document_path.unlink(missing_ok=True)


@app.post("/search")
def search(query: str = Form(...)):
    """Search by explicit class/text filter or semantic similarity."""

    with _catalog_lock:
        catalog_snapshot = catalog
        embeddings_snapshot = embeddings

    if query.startswith("(") and ")" in query:
        closing_parenthesis = query.index(")")

        class_filter = query[1:closing_parenthesis].strip()
        text_filter = query[
            closing_parenthesis + 1:
        ].strip()

        # Corpus documents use ground-truth labels; uploads use predictions.
        document_classes = catalog_snapshot["label_name"].fillna(
            catalog_snapshot["predicted_class"]
        ).fillna("")

        matches = catalog_snapshot[
            (document_classes.str.lower() == class_filter.lower())
            & (
                catalog_snapshot["text"].str.contains(
                    text_filter,
                    case=False,
                    na=False,
                    regex=False,
                )
            )
        ]

        results = [
            _format_search_result(document)
            for _, document in matches.iterrows()
        ]

        return {"results": results}

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    )

    similarities = np.dot(
        embeddings_snapshot,
        query_embedding,
    )

    positions = np.argsort(-similarities)[:5]

    results = [
        _format_search_result(
            catalog_snapshot.iloc[position],
            similarity=float(similarities[position]),
        )
        for position in positions
    ]

    return {"results": results}


def _format_search_result(
    document: pd.Series,
    similarity: float | None = None,
) -> dict:
    """Convert a catalog row into the response format used by the frontend."""

    # Local/corpus documents use source_path; AWS uploads use an S3 key.
    reference = document["source_path"]

    if pd.isna(reference):
        reference = document["s3_key"]

    label_name = document["label_name"]
    if pd.isna(label_name):
        label_name = None

    predicted_class = document["predicted_class"]
    if pd.isna(predicted_class):
        predicted_class = None

    return {
        "filename": document["filename"],
        "reference": reference,
        "label_name": label_name,
        "predicted_class": predicted_class,
        "similarity": similarity,
        "text_excerpt": document["text"][:300],
    }