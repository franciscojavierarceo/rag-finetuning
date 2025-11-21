from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import Array, Float32, String

# parquet_file_path = "data/wiki_dpr.parquet"
# parquet_file_path = "data/train-00000-of-00157_sample_with_timestamp.parquet"
parquet_file_path = "data/train-00000-of-00157_sample_with_timestamp_chunked.parquet"

wiki_passage = Entity(
    name="id",
    join_keys=["id"],
    value_type=ValueType.STRING,
    description="Unique ID of a Wikipedia passage",
)

wiki_dpr_source = FileSource(
    name="wiki_dpr_source",
    file_format=ParquetFormat(),
    path=parquet_file_path,
    timestamp_field="event_timestamp",
)

wiki_passage_feature_view = FeatureView(
    name="wiki_passages",
    entities=[wiki_passage],
    ttl=timedelta(days=1),
    schema=[
        Field(
            name="text",
            dtype=String,
            description="Content of the Wikipedia passage",
        ),
        Field(
            name="embeddings",
            dtype=Array(Float32),
            description="vectors",
            vector_index=True,
            vector_length=384,
            vector_search_metric="COSINE",
        ),
    ],
    online=True,  # Enabled for online vector search with Milvus
    source=wiki_dpr_source,
    description="Content features of Wikipedia passages",
)
