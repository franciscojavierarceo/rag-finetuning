#!/usr/bin/env python3
"""
Kubeflow-Integrated Embedding Fine-tuning for RAG Systems

This script integrates the hybrid embedding fine-tuning approach with your existing
Kubeflow trainer infrastructure using PyTorch distributed training.
"""

from kubeflow.trainer import (
    TrainerClient,
    CustomTrainer,
    LocalProcessBackendConfig,
)


def hybrid_embedding_training(*args, **func_args):
    """
    Hybrid embedding fine-tuning function for Kubeflow trainer

    This function implements the hybrid approach:
    1. Main training data from Feast offline store
    2. Periodic hard negative updates using current model
    3. In-batch negatives via MultipleNegativesRankingLoss
    4. PyTorch distributed training support
    """
    # Handle both calling styles:
    # 1. Kubeflow: hybrid_embedding_training({'key': 'value'}) - positional dict
    # 2. Direct: hybrid_embedding_training(key='value') - keyword args
    if args and len(args) == 1 and isinstance(args[0], dict):
        func_args = args[0]  # Use positional dict from Kubeflow

    import os
    import torch
    import torch.distributed as dist
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from sentence_transformers.util import cos_sim
    import logging
    from pathlib import Path
    import json
    from datetime import datetime, timedelta
    from torch.utils.tensorboard import SummaryWriter

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize distributed training if available
    world_size = 1
    rank = 0
    local_rank = 0

    if "WORLD_SIZE" in os.environ:
        try:
            # Initialize with longer timeout for hard negative generation
            dist.init_process_group(
                backend="gloo",
                timeout=timedelta(seconds=3600)  # 1 hour timeout instead of 30 minutes
            )
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))

            logger.info("PyTorch Distributed Training Environment:")
            logger.info(f"WORLD_SIZE: {world_size}")
            logger.info(f"RANK: {rank}")
            logger.info(f"LOCAL_RANK: {local_rank}")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {e}")

    # Parse function arguments
    model_name = func_args.get("model_name", "all-MiniLM-L6-v2")
    epochs = int(func_args.get("epochs", "5"))
    batch_size = int(func_args.get("batch_size", "16"))
    learning_rate = float(func_args.get("learning_rate", "2e-5"))
    max_samples = func_args.get("max_samples")
    feast_repo_path = func_args.get("feast_repo_path", "feature_repo")
    hard_negative_update_freq = int(
        func_args.get("hard_negative_update_frequency", "3")
    )

    if max_samples:
        max_samples = int(max_samples)

    # Only log from rank 0 to avoid duplicate logs
    if rank == 0:
        logger.info(
            f"Starting hybrid embedding fine-tuning with args: {func_args}"
        )

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if rank == 0:
        logger.info(f"Using device: {device}")

    # Load model - use cached version if running in Kubernetes to avoid network calls
    if os.path.exists("/workspace/cached_model/all-MiniLM-L6-v2") and model_name == "all-MiniLM-L6-v2":
        model_path = "/workspace/cached_model/all-MiniLM-L6-v2"
        if rank == 0:
            logger.info(f"Using cached model from: {model_path}")
    elif os.path.exists("./cached_model/all-MiniLM-L6-v2") and model_name == "all-MiniLM-L6-v2":
        model_path = "./cached_model/all-MiniLM-L6-v2"
        if rank == 0:
            logger.info(f"Using cached model from: {model_path}")
    else:
        model_path = model_name
        if rank == 0:
            logger.info(f"Downloading model from Hugging Face: {model_name}")

    model = SentenceTransformer(model_path, device=device)
    if rank == 0:
        logger.info(
            f"Loaded model: {model_name} (dim: {model.get_sentence_embedding_dimension()})"
        )

    # Initialize TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0:
        # Dynamic project directory - works in local, Docker, and Kubernetes environments
        project_dir = (
            func_args.get("project_dir")
            or os.environ.get("PROJECT_DIR")
            or os.getcwd()
        )
        # Use mounted volume path if running in Kubernetes, otherwise local path
        if os.path.exists("/workspace/outputs"):
            log_dir = f"/workspace/outputs/tensorboard_logs/embedding_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            log_dir = f"{project_dir}/tensorboard_logs/embedding_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logging to: {log_dir}")

        # Log hyperparameters
        writer.add_hparams(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "world_size": world_size,
                "hard_negative_update_freq": hard_negative_update_freq,
            },
            {},
        )

    # Dataset class for Feast training data
    class FeastTrainingDataset(Dataset):
        def __init__(self, data_path, max_samples=None):
            # Load training data from Feast offline store data
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Training data not found at {data_path}"
                )

            self.df = pd.read_parquet(data_path)
            if max_samples:
                self.df = self.df.head(max_samples)

            # Filter out any invalid samples
            self.df = self.df.dropna(subset=["query_text", "document_text"])

            if rank == 0:
                logger.info(f"Loaded {len(self.df)} training samples")
                label_counts = self.df["label"].value_counts()
                logger.info("Dataset composition:")
                for label, count in label_counts.items():
                    logger.info(f"  {label}: {count}")

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # Return InputExample for sentence-transformers
            return InputExample(
                texts=[row["query_text"], row["document_text"]],
                label=1.0 if row["label"] == "positive" else 0.0,
            )

    # Load training dataset - use dynamic paths that work in any environment
    import os

    # Use the same dynamic project directory approach
    project_dir = (
        func_args.get("project_dir")
        or os.environ.get("PROJECT_DIR")
        or os.getcwd()
    )

    # Check if we're running in Kubeflow (temporary directory) or locally
    current_dir = os.getcwd()
    if ("/tmp/" in current_dir or "/T/" in current_dir or
        "/private/var/folders" in current_dir or "tmp" in current_dir):
        # Running in Kubeflow temporary directory - try absolute paths
        potential_paths = [
            f"{project_dir}/{feast_repo_path}/data/embedding_training_data.parquet",
            f"/Users/farceo/dev/rag-finetuning/{feast_repo_path}/data/embedding_training_data.parquet",
            "/data/embedding_training_data.parquet",  # Container mount point
        ]
        training_data_path = None
        for path in potential_paths:
            if os.path.exists(path):
                training_data_path = path
                break
    else:
        # Running locally - use relative path
        training_data_path = f"{feast_repo_path}/data/embedding_training_data.parquet"

    if not training_data_path or not os.path.exists(training_data_path):
        raise FileNotFoundError(
            f"Training data not found. Checked paths: {potential_paths if 'potential_paths' in locals() else [f'{feast_repo_path}/data/embedding_training_data.parquet']}. "
            f"Please run: uv run prepare_training_data.py first"
        )

    if rank == 0:
        logger.info(f"Using training data from: {training_data_path}")

    dataset = FeastTrainingDataset(training_data_path, max_samples)

    # Create balanced fixed test set for consistent evaluation (never changes)
    test_sets = []

    # Get samples from each label type for balanced evaluation
    for label in ["positive", "hard_negative", "negative"]:
        label_data = dataset.df[dataset.df["label"] == label]
        if len(label_data) > 0:
            # Take up to 20 samples of each type, or all available if less
            sample_size = min(20, len(label_data))
            test_samples = label_data.sample(n=sample_size, random_state=42)
            test_sets.append(test_samples)
            if rank == 0:
                logger.info(f"Test set: {sample_size} {label} samples")

    # Combine all test samples
    if test_sets:
        test_set = pd.concat(test_sets, ignore_index=True)
        # Remove test samples from training data
        dataset.df = dataset.df.drop(test_set.index).reset_index(drop=True)
    else:
        # Fallback if no balanced split is possible
        test_set_size = min(60, int(0.1 * len(dataset)))
        test_indices = dataset.df.sample(n=test_set_size, random_state=42).index
        test_set = dataset.df.loc[test_indices].copy()
        dataset.df = dataset.df.drop(test_indices).reset_index(drop=True)

    if rank == 0:
        logger.info(f"Created fixed test set: {len(test_set)} samples")
        logger.info(f"Training data: {len(dataset.df)} samples")

    # Split remaining data into train/eval (80/20 split)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    if rank == 0:
        logger.info(
            f"Dataset split - Training: {len(train_dataset)}, Evaluation: {len(eval_dataset)}"
        )

    # Distributed sampling for training
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

    # Evaluation dataloader (no distributed sampling needed for eval)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False
    )

    # Training loss - ContrastiveLoss to pull positives closer and push negatives apart
    train_loss = losses.ContrastiveLoss(model=model)

    # Function to update hard negatives periodically
    def update_hard_negatives_dynamic():
        """Generate new hard negatives using current model"""
        if rank != 0:  # Only update from rank 0
            return 0

        logger.info("Updating hard negatives with current model...")

        try:
            # Use smaller sample for faster processing in distributed mode
            positive_samples = dataset.df[
                dataset.df["label"] == "positive"
            ].sample(min(10, len(dataset.df)))  # Reduced from 20 to 10

            new_hard_negatives = []

            for _, row in positive_samples.iterrows():
                query_text = row["query_text"]

                # Get random documents as potential hard negatives
                random_docs = dataset.df["document_text"].sample(10).tolist()

                # Encode query and documents
                query_emb = model.encode([query_text])
                doc_embs = model.encode(random_docs)

                # Find most similar (but not exact match)
                similarities = cos_sim(
                    torch.tensor(query_emb), torch.tensor(doc_embs)
                )[0]

                # Get moderately similar documents (potential hard negatives)
                for i, sim_score in enumerate(similarities):
                    if 0.4 < sim_score < 0.8:  # Hard negative range
                        new_pair = {
                            "training_sample_id": f"dynamic_hard_{len(new_hard_negatives)}_{datetime.now().timestamp()}",
                            "query_text": query_text,
                            "document_text": random_docs[i],
                            "label": "hard_negative",
                            "similarity_score": sim_score.item(),
                            "query_id": f"dynamic_{row.get('query_id', 'unknown')}",
                            "document_id": f"dynamic_doc_{i}",
                            "metadata": json.dumps(
                                {
                                    "type": "dynamic_hard_negative",
                                    "similarity": sim_score.item(),
                                }
                            ),
                            "event_timestamp": datetime.now(),
                        }
                        new_hard_negatives.append(new_pair)

                        if len(new_hard_negatives) >= 25:  # Reduced from 50 to 25 for faster processing
                            break

                if len(new_hard_negatives) >= 25:
                    break

            # Replace old hard negatives with new ones (constant dataset size)
            if new_hard_negatives:
                new_df = pd.DataFrame(new_hard_negatives)

                # Find oldest hard negatives to replace
                old_hard_negatives = dataset.df[dataset.df["label"] == "hard_negative"].head(len(new_hard_negatives))

                if len(old_hard_negatives) > 0:
                    # Remove old hard negatives
                    dataset.df = dataset.df.drop(old_hard_negatives.index).reset_index(drop=True)
                    logger.info(f"Removed {len(old_hard_negatives)} old hard negatives")

                # Add new hard negatives (same count, so dataset size stays constant)
                dataset.df = pd.concat([dataset.df, new_df], ignore_index=True)
                del new_df  # Free memory immediately

                logger.info(f"Added {len(new_hard_negatives)} new hard negatives. Dataset size: {len(dataset.df)} (constant)")

                # Also save to file for persistence
                updated_path = (
                    f"{feast_repo_path}/data/embedding_training_data.parquet"
                )
                dataset.df.to_parquet(updated_path, index=False)

                logger.info(
                    f"Added {len(new_hard_negatives)} new hard negative pairs"
                )

                # Log to TensorBoard
                if writer:
                    writer.add_scalar(
                        "Hard_Negatives/New_Pairs_Added",
                        len(new_hard_negatives),
                        training_stats["epochs_completed"],
                    )
                    writer.add_scalar(
                        "Hard_Negatives/Total_Updates",
                        training_stats["hard_negative_updates"],
                        training_stats["epochs_completed"],
                    )

                return len(new_hard_negatives)

        except Exception as e:
            logger.warning(f"Failed to update hard negatives: {e}")

        return 0

    # Training loop with periodic hard negative updates
    training_stats = {"epochs_completed": 0, "hard_negative_updates": 0}

    for epoch in range(epochs):
        if rank == 0:
            print(f"\n🚀 STARTING EPOCH {epoch + 1}/{epochs}")
            print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}")
            print(
                "Loss Function: ContrastiveLoss (pulls positives closer, pushes negatives apart)"
            )
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        # Update hard negatives periodically
        if epoch > 0 and epoch % hard_negative_update_freq == 0:
            if rank == 0:
                print(f"🔄 UPDATING HARD NEGATIVES (Epoch {epoch + 1})")
            new_pairs = update_hard_negatives_dynamic()
            if new_pairs > 0:
                training_stats["hard_negative_updates"] += 1
                if rank == 0:
                    print(f"✅ Added {new_pairs} new hard negative pairs")
                # Recreate train/eval split and dataloaders with updated dataset
                train_size = int(0.8 * len(dataset))
                eval_size = len(dataset) - train_size
                train_dataset, eval_dataset = torch.utils.data.random_split(
                    dataset, [train_size, eval_size]
                )

                # Clear old dataloaders to free memory
                import gc
                gc.collect()

                if world_size > 1:
                    train_sampler = (
                        torch.utils.data.distributed.DistributedSampler(
                            train_dataset, num_replicas=world_size, rank=rank
                        )
                    )
                    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        sampler=train_sampler,
                    )
                else:
                    train_dataloader = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True
                    )

                eval_dataloader = DataLoader(
                    eval_dataset, batch_size=batch_size, shuffle=False
                )

                # Force garbage collection after dataloader recreation
                gc.collect()

        # Set sampler epoch for distributed training
        if world_size > 1 and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        # Train one epoch - no built-in evaluator since we do custom evaluation
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100 if epoch == 0 else 0,
            optimizer_params={"lr": learning_rate},
            show_progress_bar=(rank == 0),
            use_amp=True,
        )

        training_stats["epochs_completed"] = epoch + 1

        # Detailed evaluation and logging after every epoch
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch + 1}/{epochs} EVALUATION")
            print(f"{'=' * 60}")

            # Evaluate on FIXED TEST SET (never changes, no data leakage)
            pos_samples = test_set[test_set["label"] == "positive"].head(20)
            hard_neg_samples = test_set[test_set["label"] == "hard_negative"].head(20)
            neg_samples = test_set[test_set["label"] == "negative"].head(20)

            metrics = {}

            if len(pos_samples) > 0:
                queries = pos_samples["query_text"].tolist()
                docs = pos_samples["document_text"].tolist()

                query_embs = model.encode(queries)
                doc_embs = model.encode(docs)

                similarities = cos_sim(
                    torch.tensor(query_embs), torch.tensor(doc_embs)
                )
                avg_pos_sim = torch.mean(torch.diag(similarities)).item()
                metrics["positive_similarity"] = avg_pos_sim

                print(f"✅ Positive Pair Similarity: {avg_pos_sim:.4f}")

            # Evaluate hard negatives
            if len(hard_neg_samples) > 0:
                queries = hard_neg_samples["query_text"].tolist()
                docs = hard_neg_samples["document_text"].tolist()

                query_embs = model.encode(queries)
                doc_embs = model.encode(docs)

                similarities = cos_sim(
                    torch.tensor(query_embs), torch.tensor(doc_embs)
                )
                avg_hard_neg_sim = torch.mean(torch.diag(similarities)).item()
                metrics["hard_negative_similarity"] = avg_hard_neg_sim

                print(f"⚠️  Hard Negative Similarity: {avg_hard_neg_sim:.4f}")

            # Evaluate random negatives
            if len(neg_samples) > 0:
                queries = neg_samples["query_text"].tolist()
                docs = neg_samples["document_text"].tolist()

                query_embs = model.encode(queries)
                doc_embs = model.encode(docs)

                similarities = cos_sim(
                    torch.tensor(query_embs), torch.tensor(doc_embs)
                )
                avg_neg_sim = torch.mean(torch.diag(similarities)).item()
                metrics["negative_similarity"] = avg_neg_sim

                print(f"❌ Negative Pair Similarity: {avg_neg_sim:.4f}")

            # Calculate separation metrics
            if (
                "positive_similarity" in metrics
                and "hard_negative_similarity" in metrics
            ):
                pos_hard_gap = (
                    metrics["positive_similarity"]
                    - metrics["hard_negative_similarity"]
                )
                print(f"📊 Pos-HardNeg Gap: {pos_hard_gap:.4f}")
                metrics["pos_hard_gap"] = pos_hard_gap

            if (
                "positive_similarity" in metrics
                and "negative_similarity" in metrics
            ):
                pos_neg_gap = (
                    metrics["positive_similarity"]
                    - metrics["negative_similarity"]
                )
                print(f"📊 Pos-Neg Gap: {pos_neg_gap:.4f}")
                metrics["pos_neg_gap"] = pos_neg_gap

            # Dataset info
            label_counts = dataset.df["label"].value_counts()
            test_label_counts = test_set["label"].value_counts()
            print("\n📈 Training Dataset Composition:")
            for label, count in label_counts.items():
                print(f"   {label}: {count}")
            print(f"   Training samples: {len(train_dataset)}")
            print(f"   Validation samples: {len(eval_dataset)}")
            print("\n🧪 Fixed Test Set Composition:")
            for label, count in test_label_counts.items():
                print(f"   {label}: {count}")
            print(f"   Total test samples: {len(test_set)}")

            # Log to TensorBoard
            if writer:
                # Similarity metrics
                for metric_name, value in metrics.items():
                    writer.add_scalar(
                        f"Evaluation/{metric_name}", value, epoch + 1
                    )

                # Dataset composition
                for label, count in label_counts.items():
                    writer.add_scalar(
                        f"Dataset/{label}_count", count, epoch + 1
                    )

                # Dataset sizes
                writer.add_scalar(
                    "Dataset/Total_Size", len(dataset.df), epoch + 1
                )
                writer.add_scalar(
                    "Dataset/Train_Size", len(train_dataset), epoch + 1
                )
                writer.add_scalar(
                    "Dataset/Eval_Size", len(eval_dataset), epoch + 1
                )

            print(f"{'=' * 60}\n")

    # Calculate REAL final evaluation metrics
    def evaluate_final_model_performance():
        """Evaluate the final model on the full dataset with real metrics"""
        print("🔍 FINAL MODEL EVALUATION")
        print("=" * 60)

        # Get positive and negative pairs from FIXED TEST SET for evaluation
        positive_pairs = [(row['query_text'], row['document_text']) for _, row in test_set.iterrows() if row['label'] == 'positive']
        negative_pairs = [(row['query_text'], row['document_text']) for _, row in test_set.iterrows() if row['label'] != 'positive']

        metrics = {}

        if positive_pairs:
            # Evaluate positive pairs
            pos_anchors = [pair[0] for pair in positive_pairs[:100]]  # Sample for speed
            pos_positives = [pair[1] for pair in positive_pairs[:100]]

            pos_anchor_embeddings = model.encode(pos_anchors)
            pos_positive_embeddings = model.encode(pos_positives)

            pos_similarities = [
                np.dot(a, p) / (np.linalg.norm(a) * np.linalg.norm(p))
                for a, p in zip(pos_anchor_embeddings, pos_positive_embeddings)
            ]
            metrics['positive_similarity_mean'] = np.mean(pos_similarities)
            metrics['positive_similarity_std'] = np.std(pos_similarities)

        if negative_pairs:
            # Evaluate negative pairs
            neg_anchors = [pair[0] for pair in negative_pairs[:100]]  # Sample for speed
            neg_negatives = [pair[1] for pair in negative_pairs[:100]]

            neg_anchor_embeddings = model.encode(neg_anchors)
            neg_negative_embeddings = model.encode(neg_negatives)

            neg_similarities = [
                np.dot(a, n) / (np.linalg.norm(a) * np.linalg.norm(n))
                for a, n in zip(neg_anchor_embeddings, neg_negative_embeddings)
            ]
            metrics['negative_similarity_mean'] = np.mean(neg_similarities)
            metrics['negative_similarity_std'] = np.std(neg_similarities)

        # Calculate separation (key metric for embedding quality)
        if 'positive_similarity_mean' in metrics and 'negative_similarity_mean' in metrics:
            metrics['similarity_separation'] = metrics['positive_similarity_mean'] - metrics['negative_similarity_mean']

            # Overall quality score (0.0 to 1.0)
            # Good embeddings: positive > 0.8, negative < 0.3, separation > 0.5
            pos_score = min(metrics['positive_similarity_mean'], 1.0) if metrics['positive_similarity_mean'] > 0 else 0.0
            separation_score = min(metrics['similarity_separation'] / 0.6, 1.0) if metrics['similarity_separation'] > 0 else 0.0
            metrics['overall_quality_score'] = (pos_score + separation_score) / 2.0
        else:
            metrics['overall_quality_score'] = 0.0

        return metrics

    # Get real evaluation metrics
    final_metrics = evaluate_final_model_performance() if rank == 0 else {}

    # Save final model (only from rank 0)
    if rank == 0:
        # Use dynamic project directory for consistent model saving
        project_dir = (
            func_args.get("project_dir")
            or os.environ.get("PROJECT_DIR")
            or os.getcwd()
        )
        # Use mounted volume path if running in Kubernetes, otherwise local path
        if os.path.exists("/workspace/outputs"):
            output_dir = Path("/workspace/outputs/fine_tuned_kubeflow_embeddings")
        else:
            output_dir = Path(f"{project_dir}/fine_tuned_kubeflow_embeddings")
        output_dir.mkdir(exist_ok=True, parents=True)

        model.save(str(output_dir))

        # Save training info
        training_info = {
            "base_model": model_name,
            "loss_function": "ContrastiveLoss",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "world_size": world_size,
            "training_stats": training_stats,
            "output_path": str(output_dir),
            "timestamp": datetime.now().isoformat(),
            "improvements": "Ultra-low LR (1e-6) + ContrastiveLoss to increase positive similarities",
        }

        with open(output_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        # Print comprehensive training summary
        print("\n🎉 TRAINING COMPLETED!")
        print(f"{'=' * 60}")
        print(f"📁 Model saved to: {output_dir}")
        print("🔧 Training Configuration:")
        print(
            "   - Loss function: ContrastiveLoss (designed to increase positive similarities)"
        )
        print(f"   - Learning rate: {learning_rate} (ultra-conservative)")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Total epochs: {training_stats['epochs_completed']}")
        print("📊 Training Results:")
        print(
            f"   - Hard negative updates: {training_stats['hard_negative_updates']}"
        )
        # Display real evaluation metrics
        if final_metrics:
            print("📊 REAL MODEL EVALUATION RESULTS:")
            if 'positive_similarity_mean' in final_metrics:
                print(f"   ✅ Positive similarity: {final_metrics['positive_similarity_mean']:.4f} ± {final_metrics.get('positive_similarity_std', 0):.4f}")
            if 'negative_similarity_mean' in final_metrics:
                print(f"   ❌ Negative similarity: {final_metrics['negative_similarity_mean']:.4f} ± {final_metrics.get('negative_similarity_std', 0):.4f}")
            if 'similarity_separation' in final_metrics:
                print(f"   🎯 Separation: {final_metrics['similarity_separation']:.4f} (higher is better)")
            print(f"   🏆 Overall Quality Score: {final_metrics.get('overall_quality_score', 0):.4f}/1.0")

            # Quality interpretation
            quality_score = final_metrics.get('overall_quality_score', 0)
            if quality_score >= 0.8:
                print("   🎉 EXCELLENT: Model shows great embedding quality!")
            elif quality_score >= 0.6:
                print("   ✅ GOOD: Model shows solid embedding quality")
            elif quality_score >= 0.4:
                print("   ⚠️  MODERATE: Model needs more training")
            else:
                print("   ❌ POOR: Model needs significant improvement")
        else:
            print("   ⚠️  Final evaluation skipped (distributed training)")

        print("💡 Key Metrics:")
        print("   - Positive similarity should be > 0.8")
        print("   - Negative similarity should be < 0.3")
        print("   - Separation should be > 0.5")
        print(f"{'=' * 60}")

        logger.info(f"Model saved to {output_dir}")
        logger.info(f"Training completed - Stats: {training_stats}")

        # Log REAL metrics to TensorBoard
        if writer and final_metrics:
            # Log real evaluation metrics
            for metric_name, value in final_metrics.items():
                writer.add_scalar(f"Final_Evaluation/{metric_name}", value, epochs)

            writer.add_scalar(
                "Training/Total_Hard_Negative_Updates",
                training_stats["hard_negative_updates"],
                epochs,
            )

            # Add final model info as text with REAL metrics
            model_info = f"""
Base Model: {model_name}
Total Epochs: {epochs}
Hard Negative Updates: {training_stats["hard_negative_updates"]}
World Size: {world_size}

REAL EVALUATION METRICS:"""

            if final_metrics:
                for metric_name, value in final_metrics.items():
                    model_info += f"\n{metric_name}: {value:.4f}"

            writer.add_text("Model_Info", model_info, epochs)

            writer.close()
            logger.info("TensorBoard logging completed with REAL metrics")

    # Return real quality score instead of fake score
    final_score = final_metrics.get('overall_quality_score', 0.0) if rank == 0 else 0.0

    if rank == 0:
        logger.info(f"Real model quality score: {final_score:.4f}/1.0")

    return final_score


# Create the Kubeflow training job
if __name__ == "__main__":
    # Create TrainerClient with local backend
    client = TrainerClient(backend_config=LocalProcessBackendConfig())

    # Create the TrainJob for embedding fine-tuning
    job_id = client.train(
        runtime=client.get_runtime("torch-distributed"),
        trainer=CustomTrainer(
            func=hybrid_embedding_training,
            func_args={
                "model_name": "all-MiniLM-L6-v2",
                "epochs": "25",  # Laptop-friendly, same as deployment script
                "batch_size": "16",  # Good batch size for local training
                "learning_rate": "2e-6",
                "max_samples": "500",  # Use more data, same as deployment
                "feast_repo_path": "feature_repo",
                "hard_negative_update_frequency": "5",  # Same as deployment
            },
            packages_to_install=[
                "kubeflow==0.2.0",
                "torch>=2.8.0",
                "sentence-transformers>=3.0.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "tensorboard>=2.14.0",
                "pyarrow>=15.0.0",
                "datasets>=3.0.0",
                "accelerate>=0.26.0",
            ],
            num_nodes=3,
            resources_per_node={"cpu": 2},
        ),
    )

    # Wait for TrainJob to complete
    client.wait_for_job_status(job_id)

    # Print TrainJob logs
    print("\n".join(client.get_job_logs(name=job_id)))
