import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
S3_BUCKET = os.getenv("S3_BUCKET")

TEXT_MODEL_KEY = os.getenv(
    "TEXT_MODEL_KEY",
    "tfm-aws/jupyterlab/models/text/model.tar.gz",
)

IMAGE_MODEL_KEY = os.getenv(
    "IMAGE_MODEL_KEY",
    "tfm-aws/jupyterlab/models/image/model.tar.gz",
)

FUSION_MODEL_KEY = os.getenv(
    "FUSION_MODEL_KEY",
    "tfm-aws/jupyterlab/models/fusion/fusion_logreg.joblib",
)

CLASS_ORDER_KEY = os.getenv(
    "CLASS_ORDER_KEY",
    "tfm-aws/jupyterlab/data/full/class_order.json",
)

CATALOG_KEY = os.getenv(
    "CATALOG_KEY",
    "tfm-aws/jupyterlab/semantic/documents.parquet",
)

DOCUMENTS_PREFIX = os.getenv(
    "DOCUMENTS_PREFIX",
    "tfm-aws/jupyterlab/app/documents",
)

EMBEDDING_MODEL_ID = os.getenv(
    "EMBEDDING_MODEL_ID",
    "BAAI/bge-small-en-v1.5",
)

ARTIFACT_SOURCE = os.getenv(
    "ARTIFACT_SOURCE",
    "local",
).strip().lower()

if ARTIFACT_SOURCE not in {"local", "s3"}:
    raise RuntimeError(
        "ARTIFACT_SOURCE must be either 'local' or 's3'."
    )
