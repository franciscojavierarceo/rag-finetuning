# Feast feature repository definitions

# Import feature views from existing modules
from wiki_features import wiki_passage_feature_view
from training_data_features import embedding_training_pairs, query_embeddings_view

# Import new fine-tuned feature definitions
from fine_tuned_wiki_features import fine_tuned_wiki_passages
from fine_tuned_embedding_odfv import fine_tuned_query_embeddings

# Export all feature definitions for Feast to discover
__all__ = [
    # Original feature views
    "wiki_passage_feature_view",
    "embedding_training_pairs",
    "query_embeddings_view",

    # New fine-tuned feature definitions
    "fine_tuned_wiki_passages",
    "fine_tuned_query_embeddings",
]