#!/usr/bin/env python3
"""
Prepare Training Data for Embedding Fine-tuning

This script creates positive/negative/hard negative training pairs and stores them
in Feast's offline store for efficient training.
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import random
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataPreparer:
    """Prepare training data for embedding fine-tuning"""

    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        source_data_path: str = "feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet",
    ):
        self.base_model_name = base_model_name
        self.source_data_path = source_data_path

        # Load base model for hard negative generation
        logger.info(f"Loading base model: {base_model_name}")
        self.model = SentenceTransformer(base_model_name)

        # Load source data
        logger.info(f"Loading source data from: {source_data_path}")
        self.source_df = pd.read_parquet(source_data_path)
        logger.info(f"Loaded {len(self.source_df)} source documents")

    def create_positive_pairs(self) -> List[Dict]:
        """Create positive training pairs from title-text relationships"""

        positive_pairs = []

        for idx, row in self.source_df.iterrows():
            if pd.isna(row["title"]) or pd.isna(row["text"]):
                continue

            # Title-Text positive pair
            pair = {
                "training_sample_id": f"pos_{row['id']}_{idx}",
                "query_text": row["title"],
                "document_text": row["text"],
                "label": "positive",
                "similarity_score": 1.0,  # Perfect match
                "query_id": f"query_{row['id']}",
                "document_id": row["id"],
                "metadata": json.dumps({"pair_type": "title_text", "source_row": idx}),
                "event_timestamp": datetime.now(),
            }
            positive_pairs.append(pair)

        logger.info(f"Created {len(positive_pairs)} positive pairs")
        return positive_pairs

    def create_random_negative_pairs(self, num_negatives: int = None) -> List[Dict]:
        """Create random negative pairs"""

        if num_negatives is None:
            num_negatives = len(self.source_df) // 2  # 50% of positive pairs

        negative_pairs = []

        for i in range(num_negatives):
            # Get two random different documents
            idx1, idx2 = random.sample(range(len(self.source_df)), 2)
            row1, row2 = self.source_df.iloc[idx1], self.source_df.iloc[idx2]

            if pd.isna(row1["title"]) or pd.isna(row2["text"]):
                continue

            pair = {
                "training_sample_id": f"neg_{row1['id']}_{row2['id']}_{i}",
                "query_text": row1["title"],
                "document_text": row2["text"],
                "label": "negative",
                "similarity_score": 0.0,  # Assumed no relevance
                "query_id": f"query_{row1['id']}",
                "document_id": row2["id"],
                "metadata": json.dumps(
                    {
                        "pair_type": "random_negative",
                        "query_row": idx1,
                        "doc_row": idx2,
                    }
                ),
                "event_timestamp": datetime.now(),
            }
            negative_pairs.append(pair)

        logger.info(f"Created {len(negative_pairs)} random negative pairs")
        return negative_pairs

    def create_hard_negative_pairs(
        self, num_hard_negatives_per_query: int = 2
    ) -> List[Dict]:
        """Create hard negative pairs using embedding similarity"""

        logger.info("Computing embeddings for hard negative generation...")

        # Get all valid titles and texts
        valid_rows = self.source_df.dropna(subset=["title", "text"])
        titles = valid_rows["title"].tolist()
        texts = valid_rows["text"].tolist()

        # Encode titles and texts
        title_embeddings = self.model.encode(titles, show_progress_bar=True)
        text_embeddings = self.model.encode(texts, show_progress_bar=True)

        # Calculate cross-similarities (title vs all texts)
        similarities = cos_sim(
            torch.tensor(title_embeddings), torch.tensor(text_embeddings)
        )

        hard_negative_pairs = []

        for i, row in enumerate(valid_rows.itertuples()):
            query_title = row.title
            query_id = f"query_{row.id}"

            # Get similarity scores for this query against all documents
            query_similarities = similarities[i]

            # Get indices sorted by similarity (excluding the correct document)
            sorted_indices = torch.argsort(query_similarities, descending=True)

            # Find the correct document index
            correct_doc_idx = i  # Assuming same indexing

            # Get hard negatives (similar but not correct)
            hard_neg_count = 0
            for doc_idx in sorted_indices:
                doc_idx = doc_idx.item()

                # Skip if it's the correct document
                if doc_idx == correct_doc_idx:
                    continue

                # Skip if similarity is too low (not "hard")
                sim_score = query_similarities[doc_idx].item()
                if sim_score < 0.3:  # Threshold for "hard" negatives
                    continue

                doc_row = valid_rows.iloc[doc_idx]

                pair = {
                    "training_sample_id": f"hard_neg_{row.id}_{doc_row.id}_{hard_neg_count}",
                    "query_text": query_title,
                    "document_text": doc_row.text,
                    "label": "hard_negative",
                    "similarity_score": sim_score,
                    "query_id": query_id,
                    "document_id": doc_row.id,
                    "metadata": json.dumps(
                        {
                            "pair_type": "hard_negative",
                            "base_similarity": sim_score,
                            "query_row": i,
                            "doc_row": doc_idx,
                        }
                    ),
                    "event_timestamp": datetime.now(),
                }
                hard_negative_pairs.append(pair)

                hard_neg_count += 1
                if hard_neg_count >= num_hard_negatives_per_query:
                    break

        logger.info(f"Created {len(hard_negative_pairs)} hard negative pairs")
        return hard_negative_pairs

    def create_complete_training_dataset(
        self,
        output_path: str = "feature_repo/data/embedding_training_data.parquet",
        num_hard_negatives_per_query: int = 2,
        random_negative_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """Create complete training dataset with all pair types"""

        logger.info("Creating complete training dataset...")

        # Create positive pairs
        positive_pairs = self.create_positive_pairs()

        # Create random negative pairs
        num_random_negatives = int(len(positive_pairs) * random_negative_ratio)
        negative_pairs = self.create_random_negative_pairs(num_random_negatives)

        # Create hard negative pairs
        hard_negative_pairs = self.create_hard_negative_pairs(
            num_hard_negatives_per_query
        )

        # Combine all pairs
        all_pairs = positive_pairs + negative_pairs + hard_negative_pairs

        # Create DataFrame
        training_df = pd.DataFrame(all_pairs)

        # Shuffle the dataset
        training_df = training_df.sample(frac=1).reset_index(drop=True)

        # Add Feast-compatible fields for proper integration
        self._add_feast_compatibility_fields(training_df)

        # Save to parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        training_df.to_parquet(output_path, index=False)

        logger.info(f"Training dataset saved to: {output_path}")
        logger.info("Dataset composition:")
        logger.info(f"  - Positive pairs: {len(positive_pairs)}")
        logger.info(f"  - Random negative pairs: {len(negative_pairs)}")
        logger.info(f"  - Hard negative pairs: {len(hard_negative_pairs)}")
        logger.info(f"  - Total pairs: {len(all_pairs)}")
        logger.info("✅ Feast-compatible fields added for offline store integration")

        return training_df

    def _add_feast_compatibility_fields(self, training_df: pd.DataFrame) -> None:
        """
        Add fields required for Feast offline store compatibility.

        Adds:
        - training_sample_id: Unique identifier for each training sample (entity key)
        - event_timestamp: Timestamp for point-in-time correctness
        """
        from datetime import datetime

        # Add training sample IDs (entity key for Feast)
        training_df["training_sample_id"] = [
            f"sample_{i}" for i in range(len(training_df))
        ]

        # Add event timestamps for point-in-time correctness
        # Use current timestamp for all samples in this dataset
        current_timestamp = datetime.now()
        training_df["event_timestamp"] = current_timestamp

        logger.info(f"Added Feast compatibility fields:")
        logger.info(f"  - training_sample_id: {len(training_df)} unique IDs")
        logger.info(f"  - event_timestamp: {current_timestamp}")

    def create_query_embeddings_dataset(
        self, output_path: str = "feature_repo/data/query_embeddings.parquet"
    ) -> pd.DataFrame:
        """Create query embeddings dataset for dynamic negative sampling"""

        logger.info("Creating query embeddings dataset...")

        # Get unique queries (titles)
        unique_queries = self.source_df.dropna(subset=["title"]).drop_duplicates(
            "title"
        )

        queries_data = []

        for idx, row in unique_queries.iterrows():
            query_embedding = self.model.encode([row["title"]])[0].tolist()

            query_data = {
                "query_id": f"query_{row['id']}",
                "query_text": row["title"],
                "query_embedding": query_embedding,
                "query_type": "title",
                "event_timestamp": datetime.now(),
            }
            queries_data.append(query_data)

        # Create DataFrame
        queries_df = pd.DataFrame(queries_data)

        # Save to parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        queries_df.to_parquet(output_path, index=False)

        logger.info(f"Query embeddings dataset saved to: {output_path}")
        logger.info(f"Created {len(queries_df)} query embeddings")

        return queries_df


def main():
    """Main function for command-line usage"""

    parser = argparse.ArgumentParser(
        description="Prepare training data for embedding fine-tuning"
    )
    parser.add_argument(
        "--source-data",
        default="feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet",
        help="Path to source data parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="feature_repo/data",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base model for hard negative generation",
    )
    parser.add_argument(
        "--hard-negatives-per-query",
        type=int,
        default=2,
        help="Number of hard negatives per query",
    )
    parser.add_argument(
        "--random-negative-ratio",
        type=float,
        default=0.5,
        help="Ratio of random negatives to positive pairs",
    )

    args = parser.parse_args()

    # Initialize preparer
    preparer = TrainingDataPreparer(
        base_model_name=args.base_model, source_data_path=args.source_data
    )

    # Create training dataset
    training_output = Path(args.output_dir) / "embedding_training_data.parquet"
    training_df = preparer.create_complete_training_dataset(
        output_path=str(training_output),
        num_hard_negatives_per_query=args.hard_negatives_per_query,
        random_negative_ratio=args.random_negative_ratio,
    )

    # Create query embeddings dataset
    queries_output = Path(args.output_dir) / "query_embeddings.parquet"
    queries_df = preparer.create_query_embeddings_dataset(
        output_path=str(queries_output)
    )

    logger.info("Training data preparation completed successfully!")

    return {
        "training_data_path": str(training_output),
        "training_data_size": len(training_df),
        "query_embeddings_path": str(queries_output),
        "query_embeddings_size": len(queries_df),
    }


if __name__ == "__main__":
    result = main()
    print(f"Training data preparation completed: {result}")
