from __future__ import annotations

import json
from pathlib import Path

import zvec
from zvec import (
    Collection,
    CollectionOption,
    CollectionSchema,
    DataType,
    FieldSchema,
    HnswIndexParam,
    IVFIndexParam,
    MetricType,
    VectorSchema,
)

_ZVEC_INITIALIZED = False
_OPEN_COLLECTIONS: dict[str, Collection] = {}

NAME_FIELD = "name"
TEXT_FIELD = "text"
METADATA_FIELD = "metadata"
NAME_EMBEDDING_FIELD = "name_embedding"
TEXT_EMBEDDING_FIELD = "text_embedding"
TEXT_SPARSE_EMBEDDING_FIELD = "text_sparse_embedding"
EMBED_CONFIG_FILENAME = "_embed_config.json"


def _ensure_zvec_initialized() -> None:
    global _ZVEC_INITIALIZED

    if _ZVEC_INITIALIZED:
        return

    try:
        zvec.init()
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise

    _ZVEC_INITIALIZED = True


def serialize_metadata(metadata: object) -> str:
    return json.dumps(metadata, ensure_ascii=False)


def _collection_path(collection_name: str, zvec_uri: str | Path) -> Path:
    return Path(zvec_uri) / collection_name


def write_embed_config(
    collection_name: str,
    zvec_uri: str | Path,
    config: dict[str, object],
) -> None:
    config_path = _collection_path(collection_name, zvec_uri) / EMBED_CONFIG_FILENAME
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def read_embed_config(
    collection_name: str,
    zvec_uri: str | Path,
) -> dict[str, object] | None:
    config_path = _collection_path(collection_name, zvec_uri) / EMBED_CONFIG_FILENAME
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


def get_or_create_collection(
    collection_name: str,
    zvec_uri: str,
    dim: int,
) -> Collection:
    _ensure_zvec_initialized()

    collection_path = _collection_path(collection_name, zvec_uri)
    collection_path_str = str(collection_path)

    collection = _OPEN_COLLECTIONS.get(collection_path_str)
    if collection is not None:
        return collection

    if collection_path.exists():
        collection = zvec.open(collection_path_str, option=CollectionOption())
        _OPEN_COLLECTIONS[collection_path_str] = collection
        return collection

    collection_path.parent.mkdir(parents=True, exist_ok=True)

    schema = CollectionSchema(
        name=collection_name,
        fields=[
            FieldSchema(name=NAME_FIELD, data_type=DataType.STRING),
            FieldSchema(name=TEXT_FIELD, data_type=DataType.STRING),
            FieldSchema(name=METADATA_FIELD, data_type=DataType.STRING),
        ],
        vectors=[
            VectorSchema(
                name=NAME_EMBEDDING_FIELD,
                data_type=DataType.VECTOR_FP32,
                dimension=dim,
                index_param=IVFIndexParam(
                    metric_type=MetricType.COSINE,
                    n_list=1024,
                ),
            ),
            VectorSchema(
                name=TEXT_EMBEDDING_FIELD,
                data_type=DataType.VECTOR_FP32,
                dimension=dim,
                index_param=IVFIndexParam(
                    metric_type=MetricType.COSINE,
                    n_list=1024,
                ),
            ),
            VectorSchema(
                name=TEXT_SPARSE_EMBEDDING_FIELD,
                data_type=DataType.SPARSE_VECTOR_FP32,
                index_param=HnswIndexParam(
                    ef_construction=100,
                    m=8,
                ),
            ),
        ],
    )

    collection = zvec.create_and_open(
        collection_path_str,
        schema=schema,
        option=CollectionOption(),
    )
    _OPEN_COLLECTIONS[collection_path_str] = collection
    return collection
