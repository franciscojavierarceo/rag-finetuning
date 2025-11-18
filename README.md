# RAG Embedding Fine-tuning with Kubeflow & Feast

A production-ready, fully integrated pipeline for fine-tuning embedding models for Retrieval-Augmented Generation (RAG) using Kubeflow distributed training and Feast feature store management.

## 🎯 Overview

This project implements a complete RAG fine-tuning pipeline with:
- **Kubeflow Trainer** for distributed training on KIND clusters
- **Feast Feature Store** for data management and real-time serving
- **On Demand Feature Views (ODFV)** for real-time inference
- **Consistent 384-dim embeddings** throughout the system
- **Dynamic hard negative sampling** during training
- **Local TensorBoard** with bind-mounted logs
- **End-to-end validation** and testing

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   Feast Feature     │
                    │   Store (384-dim)   │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      ▼                      │
┌───────────────┐    ┌────────────────────┐    ┌─────────────────┐
│ Training Data │───▶│   KIND Cluster     │───▶│  Local Machine  │
│ (Feast ODFV)  │    │   (2 nodes)        │    │   TensorBoard   │
└───────────────┘    └────────────────────┘    └─────────────────┘
        │                      │                          │
        │                      ▼                          ▼
        │              ┌────────────────────┐    ┌─────────────────┐
        │              │ Bind Mount         │    │ ./training_     │
        │              │ /tmp/rag-training  │───▶│ outputs/        │
        │              │ -output            │    │                 │
        │              └────────────────────┘    └─────────────────┘
        │                                                 │
        └─────────────────────────────────────────────────┘
                     Real-time Query Embeddings
                      (Fine-tuned Model ODFV)
```

### Key Components

- **Feast Offline Store**: Training data with point-in-time correctness
- **Feast Online Store**: Vector search with Milvus (384-dim embeddings)
- **ODFV**: Real-time inference using fine-tuned model
- **Consistent Embeddings**: 384-dimensional throughout the pipeline
- **Re-embedded Documents**: Wikipedia corpus with fine-tuned embeddings

## 📋 Prerequisites

- **Python 3.11+** with UV package manager
- **Docker** (for KIND and training containers)
- **8GB+ RAM** (for distributed training)
- **kubectl** and **kind** (auto-installed by setup script)

## 🚀 Quick Start

### 1. Prepare Training Data

```bash
# Prepare training data with Feast-compatible fields
uv run prepare_training_data.py \
    --source-data feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet \
    --output-dir feature_repo/data \
    --hard-negatives-per-query 2 \
    --random-negative-ratio 0.3
```

This creates:
- `embedding_training_data.parquet` with training pairs + Feast fields (`training_sample_id`, `event_timestamp`)
- `query_embeddings.parquet` for dynamic negative sampling

### 1.5. Setup Feast Feature Store

```bash
# Apply Feast configuration (offline-only for now)
cd feature_repo && uv run feast apply
```

This registers:
- ✅ Feature views for training data and Wikipedia passages
- ✅ On Demand Feature View (ODFV) for real-time inference
- ✅ Consistent 384-dimensional embedding configuration

### 2. Setup Kubernetes Cluster

```bash
# Creates KIND cluster with bind mounts and Kubeflow Trainer
./setup-kind-kubeflow.sh all
```

This will:
✅ Create KIND cluster with 3 nodes (1 control-plane, 2 workers)
✅ Install Kubeflow Trainer operator v2.1.0
✅ Setup local Docker registry
✅ Configure bind mount: `./training_outputs` ↔ `/tmp/rag-training-output`
✅ Configure registry access for all nodes

### 3. Deploy Training

```bash
# Creates PVC for training storage
kubectl apply -f training-storage.yaml

# Deploy distributed training job
uv run deploy_kubeflow_training.py
```

### 4. Monitor Training

```bash
# Start TensorBoard locally (in separate terminal)
uv run tensorboard --logdir=./training_outputs/tensorboard_logs --host=0.0.0.0 --port=6006

# Open TensorBoard in browser
open http://localhost:6006
```

### 5. Check Training Status

```bash
# View training jobs
kubectl get trainjobs

# Find training pods
kubectl get pods | grep node

# View logs (use rank 0 for evaluation metrics)
kubectl logs <job-id>-node-0-0-<suffix>
```

## 📊 Understanding Results

### Training Configuration
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Nodes**: 2 (distributed PyTorch training)
- **Batch size**: 8 per node (16 global)
- **Learning rate**: 2e-6 (ultra-conservative)
- **Loss function**: ContrastiveLoss

### Key Metrics (in TensorBoard)

1. **`Evaluation/Positive_Pair_Similarity`**
   - Should increase from ~0.6 → 0.85+
   - **Target**: >0.8 for good performance

2. **`Hard_Negatives/New_Pairs_Added`**
   - Shows dynamic sampling activity
   - Non-zero every N epochs

3. **`Dataset/Total_Size`**
   - Dataset growth from dynamic sampling

### Distributed Training Logs

**Multiple log files**: Each training node creates logs, but **only rank 0** runs evaluation.

**Look for rank 0 logs**: Search for `RANK: 0` in pod logs for authoritative metrics.

**Final performance**: Check final log output for model quality score (target: >80%).

## 🔧 Configuration

### Training Parameters

Edit `deploy_kubeflow_training.py`:

```python
func_args={
    "model_name": "all-MiniLM-L6-v2",
    "epochs": "20",                    # More epochs for better results
    "batch_size": "16",               # Larger batch size
    "learning_rate": "2e-6",
    "max_samples": None,              # Use full dataset
    "hard_negative_update_frequency": "5"
}
```

### Scaling Up

```python
# In deploy_kubeflow_training.py
num_nodes=4,                          # More distributed nodes
resources_per_node={
    "cpu": "4",
    "memory": "8Gi",
    # "nvidia.com/gpu": "1",           # GPU support if available
}
```

## 📁 Project Structure

```
rag-finetuning/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── Dockerfile                       # Training container
│
├── setup-kind-kubeflow.sh           # ✅ Cluster setup with bind mounts
├── deploy_kubeflow_training.py      # ✅ Main deployment script
├── kubeflow_embedding_training.py   # ✅ Training logic with Feast integration
├── prepare_training_data.py         # ✅ Data preparation with Feast fields
├── training-storage.yaml            # ✅ PVC configuration
│
├── re_embed_wikipedia.py            # 🆕 Re-embed corpus with fine-tuned model
├── test_end_to_end.py               # 🆕 End-to-end pipeline validation
│
├── feature_repo/                    # 🔄 Feast feature store
│   ├── feature_store.yaml           #     Feast config (384-dim)
│   ├── wiki_features.py            #     Wikipedia feature views
│   ├── training_data_features.py   #     Training data feature views
│   ├── fine_tuned_wiki_features.py # 🆕 Re-embedded documents
│   ├── fine_tuned_embedding_odfv.py# 🆕 Real-time inference ODFV
│   ├── __init__.py                  #     Feature registry
│   └── data/                        #
│       ├── registry.db              #     Feast registry
│       ├── embedding_training_data.parquet        # Training pairs + Feast fields
│       ├── query_embeddings.parquet              # Query embeddings for sampling
│       ├── wiki_fine_tuned_embeddings.parquet   # 🆕 Re-embedded Wikipedia (384-dim)
│       └── train-00000-of-00157_sample_with_timestamp_chunked.parquet
│
├── training_outputs/               # ✅ Local output (bind mounted)
│   ├── tensorboard_logs/           # ← TensorBoard reads from here
│   └── fine_tuned_kubeflow_embeddings/ # 🚀 Fine-tuned model (384-dim)
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── training_info.json      # Training metadata
│
├── cached_model/                   # Pre-downloaded model cache
└── .venv/                         # UV virtual environment
```

### Key New Files

- **🆕 `re_embed_wikipedia.py`**: Re-embeds Wikipedia corpus with fine-tuned model
- **🆕 `fine_tuned_embedding_odfv.py`**: Real-time inference On Demand Feature View
- **🆕 `fine_tuned_wiki_features.py`**: Feature view for re-embedded documents
- **🆕 `test_end_to_end.py`**: Comprehensive pipeline validation
- **🔄 Updated training files**: Now integrated with Feast offline store

## 🎯 Training Strategy

### Hybrid Negative Sampling
1. **In-batch negatives**: Zero overhead (MultipleNegativesRankingLoss)
2. **Pre-computed hard negatives**: High quality challenging examples
3. **Dynamic hard negatives**: Updated every N epochs using current model

### Data Composition
- **~30% Positive pairs**: Title-text matches (ground truth)
- **~60% Hard negatives**: Similar but incorrect matches
- **~10% Random negatives**: Completely unrelated pairs

## 🔧 Advanced Usage

### Alternative Training Methods

**Docker-only training** (no Kubernetes):
```bash
./simple-docker.sh all
```

**Native local training** (development):
```bash
uv run kubeflow_embedding_training.py
```

### Custom Models

```python
# In training configuration
"model_name": "sentence-transformers/all-mpnet-base-v2",  # Larger model
"model_name": "paraphrase-multilingual-MiniLM-L12-v2",   # Multilingual
```

## 🐛 Troubleshooting

### Common Issues

1. **Pods stuck in Pending**: Check PVC exists (`kubectl get pvc`)
2. **ImagePullBackOff**: Verify registry setup (`./setup-kind-kubeflow.sh cluster`)
3. **No TensorBoard metrics**: Check bind mount (`ls ./training_outputs/`)
4. **Training fails**: Check pod logs (`kubectl logs <pod-name>`)

### Clean Reset

```bash
# Delete everything and start fresh
kind delete cluster --name rag-kubeflow
rm -rf ./training_outputs/
./setup-kind-kubeflow.sh all
```

### Logs and Debugging

```bash
# Check cluster status
kubectl get nodes
kubectl get pods -A

# Training job status
kubectl get trainjobs -w

# Pod logs (find rank 0 for evaluation)
kubectl logs <job-id>-node-0-0-<suffix> | grep "RANK: 0" -A 20
```

## 🔄 Post-Training: Re-embedding & Feast Integration

After training completes, integrate the fine-tuned model with Feast:

### 1. Re-embed Wikipedia Corpus

```bash
# Re-embed Wikipedia data using fine-tuned model (ensures 384-dim consistency)
uv run re_embed_wikipedia.py
```

This:
- ✅ Loads your fine-tuned model from `./training_outputs/fine_tuned_kubeflow_embeddings`
- ✅ Re-embeds Wikipedia passages with 384-dimensional vectors
- ✅ Saves to `feature_repo/data/wiki_fine_tuned_embeddings.parquet`
- ✅ Validates embedding dimensions and consistency

### 2. Test End-to-End Pipeline

```bash
# Validate complete pipeline functionality
uv run test_end_to_end.py
```

This tests:
- ✅ Feast feature store initialization
- ✅ ODFV real-time inference with fine-tuned model
- ✅ Training data integration
- ✅ Embedding dimension consistency

### 3. Enable Online Vector Store (Optional)

For production RAG with vector similarity search:

```bash
# Start Milvus vector database
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.3.0

# Re-enable online stores in feature_store.yaml
# Set online=True in feature views

# Apply and materialize
cd feature_repo
uv run feast apply
uv run feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

## 📚 Integration

### Using Fine-tuned Model

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer('./training_outputs/fine_tuned_kubeflow_embeddings')

# Use for encoding
query_embedding = model.encode("What is machine learning?")
```

### Real-time RAG with Feast ODFV

```python
from feast import FeatureStore

# Initialize Feast store
store = FeatureStore("./feature_repo")

# Generate query embedding via ODFV (uses fine-tuned model)
query_features = store.get_online_features(
    features=["fine_tuned_query_embeddings:query_embedding"],
    entity_rows=[{"query_text": "What is machine learning?"}]
).to_df()

query_embedding = query_features["query_embedding"].iloc[0]
print(f"Generated {len(query_embedding)}-dimensional embedding")  # 384-dim
```

### Production RAG Pipeline

```python
from feast import FeatureStore
import numpy as np

def rag_query(question: str, top_k: int = 10):
    store = FeatureStore("./feature_repo")

    # Step 1: Generate query embedding with ODFV
    query_features = store.get_online_features(
        features=["fine_tuned_query_embeddings:query_embedding"],
        entity_rows=[{"query_text": question}]
    ).to_df()

    query_embedding = np.array(query_features["query_embedding"].iloc[0])

    # Step 2: Vector similarity search (requires Milvus online store)
    # This would use the re-embedded Wikipedia documents
    similar_docs = store.get_online_features(
        features=["fine_tuned_wiki_passages:text", "fine_tuned_wiki_passages:embeddings"],
        entity_rows=[{"id": f"doc_{i}"} for i in range(top_k)]
    )

    return {
        "query": question,
        "query_embedding": query_embedding,
        "similar_documents": similar_docs
    }

# Example usage
result = rag_query("Explain neural networks", top_k=5)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with your data
4. Submit a pull request

## 🎉 What's New: Feast Integration

This updated version transforms the original standalone training pipeline into a **production-ready, fully integrated Feast-based RAG system**:

### 🔧 **Fixed Critical Issues**
- ✅ **Embedding Dimension Consistency**: Fixed 768-dim → 384-dim mismatch throughout system
- ✅ **Feast Integration**: Training now uses `FeatureStore.get_historical_features()` instead of static files
- ✅ **Real-time Inference**: Created ODFV for serving fine-tuned model in production
- ✅ **Re-embedded Corpus**: Wikipedia documents now use consistent fine-tuned embeddings

### 🚀 **New Capabilities**
- **On Demand Feature Views**: Real-time query embedding generation with cached model loading
- **Feature Versioning**: Proper lineage and point-in-time correctness for training data
- **End-to-end Validation**: Comprehensive testing of the complete pipeline
- **Production-Ready**: Ready for online vector stores (Milvus) and production RAG workloads

### 🧪 **Validation**
All components are tested and validated:
```bash
uv run test_end_to_end.py
# 🎉 4/4 tests passed - Pipeline is ready!
```

### 📈 **Impact**
- **Before**: Standalone training with embedding dimension mismatches
- **After**: Complete feature store-based RAG system with consistent embeddings and real-time inference

---

## 📄 License

This project is licensed under the Apache License 2.0.