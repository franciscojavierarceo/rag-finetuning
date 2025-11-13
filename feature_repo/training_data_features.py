#!/usr/bin/env python3
"""
Feast Feature Definitions for Embedding Training Data

Defines feature views for:
1. Training pairs (offline store) - for main training dataset
2. Query embeddings (online store) - for dynamic hard negative sampling
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import Array, Float32, String

# Training sample entity (for offline training data)
training_sample = Entity(
    name="training_sample_id",
    join_keys=["training_sample_id"],
    value_type=ValueType.STRING,
    description="Unique ID for each training sample (query-document pair)",
)

# Query entity (for online query embeddings)
query_entity = Entity(
    name="query_id",
    join_keys=["query_id"],
    value_type=ValueType.STRING,
    description="Unique ID for queries",
)

# Training pairs data source (offline store)
training_pairs_source = FileSource(
    name="embedding_training_pairs_source",
    file_format=ParquetFormat(),
    path="data/embedding_training_data.parquet",
    timestamp_field="event_timestamp",
)

# Training pairs feature view (OFFLINE ONLY)
embedding_training_pairs = FeatureView(
    name="embedding_training_pairs",
    entities=[training_sample],
    ttl=timedelta(days=30),
    schema=[
        Field(
            name="query_text",
            dtype=String,
            description="Query text (title, question, etc.)",
        ),
        Field(
            name="document_text",
            dtype=String,
            description="Document text content",
        ),
        Field(
            name="label",
            dtype=String,
            description="Label: positive, negative, hard_negative",
        ),
        Field(
            name="similarity_score",
            dtype=Float32,
            description="Similarity score between query and document",
        ),
        Field(
            name="query_id", dtype=String, description="Source query identifier"
        ),
        Field(
            name="document_id",
            dtype=String,
            description="Source document identifier",
        ),
        Field(
            name="metadata",
            dtype=String,
            description="JSON metadata about the training pair",
        ),
    ],
    online=False,  # Offline only - no need for online serving of training data
    source=training_pairs_source,
    description="Training pairs for embedding fine-tuning (positive/negative/hard negative)",
)

# Query embeddings data source (online store for dynamic sampling)
query_embeddings_source = FileSource(
    name="query_embeddings_source",
    file_format=ParquetFormat(),
    path="data/query_embeddings.parquet",
    timestamp_field="event_timestamp",
)

# Query embeddings feature view (ONLINE for dynamic sampling)
query_embeddings_view = FeatureView(
    name="query_embeddings",
    entities=[query_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="query_text", dtype=String, description="Query text"),
        Field(
            name="query_embedding",
            dtype=Array(Float32),
            description="Query embedding vector for similarity search",
            vector_index=True,
            vector_length=384,  # all-MiniLM-L6-v2 dimension
            vector_search_metric="COSINE",
        ),
        Field(
            name="query_type",
            dtype=String,
            description="Type of query (title, question, etc.)",
        ),
    ],
    online=True,  # Online for vector search during dynamic sampling
    source=query_embeddings_source,
    description="Query embeddings for dynamic hard negative sampling during training",
)
