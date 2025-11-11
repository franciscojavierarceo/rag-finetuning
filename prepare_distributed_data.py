#!/usr/bin/env python3
"""
Large-Scale Data Preparation with Sharding
Prepares massive datasets for distributed training across multiple nodes
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
import shutil

# Add current directory for imports
sys.path.insert(0, os.getcwd())

from sentence_transformers import SentenceTransformer


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("data_preparation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def chunk_large_dataset(input_path: str, chunk_size: int = 100000) -> List[str]:
    """
    Split large dataset into manageable chunks

    Args:
        input_path: Path to large parquet file
        chunk_size: Number of rows per chunk

    Returns:
        List of chunk file paths
    """
    logger = logging.getLogger(__name__)
    logger.info(f"📊 Chunking large dataset: {input_path}")

    # Read dataset info
    df_sample = pd.read_parquet(input_path, nrows=1000)
    total_rows = len(pd.read_parquet(input_path))

    logger.info("📈 Dataset statistics:")
    logger.info(f"   - Total rows: {total_rows:,}")
    logger.info(f"   - Columns: {list(df_sample.columns)}")
    logger.info(f"   - Chunk size: {chunk_size:,}")
    logger.info(f"   - Expected chunks: {(total_rows // chunk_size) + 1}")

    # Create chunks directory
    input_name = Path(input_path).stem
    chunks_dir = Path(f"data_chunks_{input_name}")
    chunks_dir.mkdir(exist_ok=True)

    chunk_paths = []

    # Process in chunks
    for chunk_idx, chunk_df in enumerate(
        pd.read_parquet(input_path, chunksize=chunk_size)
    ):
        chunk_path = chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
        chunk_df.to_parquet(chunk_path, index=False)
        chunk_paths.append(str(chunk_path))

        if (chunk_idx + 1) % 10 == 0:
            logger.info(f"   📦 Created {chunk_idx + 1} chunks...")

    logger.info(f"✅ Created {len(chunk_paths)} data chunks in {chunks_dir}")
    return chunk_paths


def process_chunk_for_training(
    chunk_path: str,
    base_model: str,
    hard_negatives_per_query: int,
    random_negative_ratio: float,
    chunk_idx: int,
) -> str:
    """
    Process a single chunk to create training data

    Args:
        chunk_path: Path to data chunk
        base_model: Model for embedding generation
        hard_negatives_per_query: Number of hard negatives
        random_negative_ratio: Ratio of random negatives
        chunk_idx: Chunk index for logging

    Returns:
        Path to processed training chunk
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Processing chunk {chunk_idx}: {chunk_path}")

    # Load chunk
    df = pd.read_parquet(chunk_path)

    # Initialize model for this process
    model = SentenceTransformer(base_model)

    # Create positive pairs (title-text matches)
    positive_pairs = []
    for _, row in df.iterrows():
        if pd.notna(row.get("title")) and pd.notna(row.get("text")):
            positive_pairs.append(
                {
                    "anchor": str(row["title"]),
                    "positive": str(row["text"])[:512],  # Truncate long texts
                    "label": 1,
                }
            )

    # Generate embeddings for hard negative mining
    texts = [pair["positive"] for pair in positive_pairs]
    if texts:
        logger.info(f"   📝 Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=False)

        # Hard negative mining within chunk
        hard_negatives = []
        for i, pair in enumerate(positive_pairs):
            anchor_emb = model.encode(
                [pair["anchor"]], show_progress_bar=False
            )[0]

            # Calculate similarities to find hard negatives
            similarities = np.dot(embeddings, anchor_emb) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(anchor_emb)
            )

            # Get indices of most similar (but not the exact match)
            similar_indices = np.argsort(similarities)[::-1]
            hard_neg_indices = [
                idx
                for idx in similar_indices[: hard_negatives_per_query + 5]
                if idx != i
            ][:hard_negatives_per_query]

            # Add hard negatives
            for neg_idx in hard_neg_indices:
                hard_negatives.append(
                    {
                        "anchor": pair["anchor"],
                        "positive": texts[neg_idx],
                        "label": 0,
                    }
                )

        # Add random negatives
        random_negatives = []
        num_random = int(len(positive_pairs) * random_negative_ratio)
        for _ in range(num_random):
            anchor_idx = np.random.randint(0, len(positive_pairs))
            negative_idx = np.random.randint(0, len(texts))

            if anchor_idx != negative_idx:
                random_negatives.append(
                    {
                        "anchor": positive_pairs[anchor_idx]["anchor"],
                        "positive": texts[negative_idx],
                        "label": 0,
                    }
                )

        # Combine all training examples
        training_examples = positive_pairs + hard_negatives + random_negatives
        training_df = pd.DataFrame(training_examples)

        # Save processed chunk
        output_path = chunk_path.replace(".parquet", "_training.parquet")
        training_df.to_parquet(output_path, index=False)

        logger.info(
            f"✅ Chunk {chunk_idx} processed: {len(training_examples)} training examples"
        )
        logger.info(f"   - Positives: {len(positive_pairs)}")
        logger.info(f"   - Hard negatives: {len(hard_negatives)}")
        logger.info(f"   - Random negatives: {len(random_negatives)}")

        return output_path

    else:
        logger.warning(f"⚠️ Chunk {chunk_idx} had no valid text pairs")
        return None


def parallel_chunk_processing(
    chunk_paths: List[str],
    base_model: str,
    hard_negatives_per_query: int = 3,
    random_negative_ratio: float = 0.3,
    max_workers: int = 4,
) -> List[str]:
    """
    Process chunks in parallel for training data preparation

    Args:
        chunk_paths: List of chunk file paths
        base_model: Model for embedding generation
        hard_negatives_per_query: Number of hard negatives per query
        random_negative_ratio: Ratio of random negatives
        max_workers: Number of parallel processes

    Returns:
        List of processed training chunk paths
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"🚀 Processing {len(chunk_paths)} chunks with {max_workers} workers"
    )

    processed_chunks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk processing tasks
        futures = []
        for i, chunk_path in enumerate(chunk_paths):
            future = executor.submit(
                process_chunk_for_training,
                chunk_path,
                base_model,
                hard_negatives_per_query,
                random_negative_ratio,
                i,
            )
            futures.append(future)

        # Collect results
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=3600)  # 1 hour timeout per chunk
                if result:
                    processed_chunks.append(result)
                logger.info(f"📊 Completed chunk {i + 1}/{len(chunk_paths)}")
            except Exception as e:
                logger.error(f"❌ Chunk {i} failed: {e}")

    logger.info(f"✅ Processed {len(processed_chunks)} chunks successfully")
    return processed_chunks


def combine_processed_chunks(
    processed_chunks: List[str], output_path: str, shard_size: int = 50000
) -> List[str]:
    """
    Combine processed chunks into training shards for distributed training

    Args:
        processed_chunks: List of processed chunk paths
        output_path: Output directory for training shards
        shard_size: Number of examples per shard

    Returns:
        List of training shard paths
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"📦 Combining {len(processed_chunks)} chunks into training shards"
    )

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)

    # Read and combine all processed chunks
    all_examples = []
    for chunk_path in processed_chunks:
        try:
            chunk_df = pd.read_parquet(chunk_path)
            all_examples.append(chunk_df)
            logger.info(f"   📄 Loaded chunk: {len(chunk_df)} examples")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load chunk {chunk_path}: {e}")

    if not all_examples:
        logger.error("❌ No valid chunks to combine")
        return []

    # Combine all chunks
    combined_df = pd.concat(all_examples, ignore_index=True)
    logger.info("📊 Combined dataset statistics:")
    logger.info(f"   - Total examples: {len(combined_df):,}")
    logger.info(
        f"   - Label distribution: {dict(combined_df['label'].value_counts())}"
    )

    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Create training shards
    shard_paths = []
    num_shards = (len(combined_df) // shard_size) + 1

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(combined_df))

        if start_idx < len(combined_df):
            shard_df = combined_df.iloc[start_idx:end_idx]
            shard_path = output_dir / f"training_shard_{shard_idx:04d}.parquet"
            shard_df.to_parquet(shard_path, index=False)
            shard_paths.append(str(shard_path))

            logger.info(f"   💾 Shard {shard_idx}: {len(shard_df)} examples")

    # Save combined dataset
    main_path = output_dir / "embedding_training_data.parquet"
    combined_df.to_parquet(main_path, index=False)

    # Create query embeddings for dynamic hard negative mining
    query_texts = combined_df[combined_df["label"] == 1]["anchor"].unique()[
        :1000
    ]
    if len(query_texts) > 0:
        logger.info(
            f"🔍 Creating query embeddings for {len(query_texts)} unique queries..."
        )
        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Fast model for query embeddings
        query_embeddings = model.encode(query_texts)

        query_df = pd.DataFrame(
            {
                "query": query_texts,
                "embedding": [emb.tolist() for emb in query_embeddings],
            }
        )
        query_path = output_dir / "query_embeddings.parquet"
        query_df.to_parquet(query_path, index=False)
        logger.info(f"💾 Query embeddings saved: {query_path}")

    logger.info(f"✅ Created {len(shard_paths)} training shards")
    logger.info(f"📁 Main dataset: {main_path}")
    logger.info(f"📁 Training shards: {output_dir}/training_shard_*.parquet")

    return shard_paths


def main():
    """Main function for distributed data preparation"""
    parser = argparse.ArgumentParser(
        description="Prepare large-scale data for distributed training"
    )
    parser.add_argument(
        "--input-data", required=True, help="Path to large input dataset"
    )
    parser.add_argument(
        "--output-dir", default="feature_repo/data", help="Output directory"
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base model for embeddings",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for processing",
    )
    parser.add_argument(
        "--shard-size", type=int, default=50000, help="Training shard size"
    )
    parser.add_argument(
        "--hard-negatives", type=int, default=3, help="Hard negatives per query"
    )
    parser.add_argument(
        "--random-ratio", type=float, default=0.3, help="Random negative ratio"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("🚀 Starting large-scale distributed data preparation")
    logger.info("📊 Configuration:")
    logger.info(f"   - Input data: {args.input_data}")
    logger.info(f"   - Output directory: {args.output_dir}")
    logger.info(f"   - Base model: {args.base_model}")
    logger.info(f"   - Chunk size: {args.chunk_size:,}")
    logger.info(f"   - Shard size: {args.shard_size:,}")
    logger.info(f"   - Max workers: {args.max_workers}")

    start_time = datetime.now()

    try:
        # Step 1: Chunk large dataset
        chunk_paths = chunk_large_dataset(args.input_data, args.chunk_size)

        # Step 2: Process chunks in parallel
        processed_chunks = parallel_chunk_processing(
            chunk_paths,
            args.base_model,
            args.hard_negatives,
            args.random_ratio,
            args.max_workers,
        )

        # Step 3: Combine into training shards
        shard_paths = combine_processed_chunks(
            processed_chunks, args.output_dir, args.shard_size
        )

        # Cleanup intermediate files
        logger.info("🧹 Cleaning up intermediate files...")
        for chunk_path in chunk_paths + processed_chunks:
            try:
                os.remove(chunk_path)
            except:
                pass

        # Remove chunks directories
        chunk_dirs = set(Path(p).parent for p in chunk_paths)
        for chunk_dir in chunk_dirs:
            try:
                shutil.rmtree(chunk_dir)
            except:
                pass

        total_time = datetime.now() - start_time
        logger.info("✅ Distributed data preparation completed!")
        logger.info(f"⏱️  Total time: {total_time}")
        logger.info(f"📊 Created {len(shard_paths)} training shards")
        logger.info("📁 Ready for distributed training with multiple nodes")

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()
