# RAG Embedding Fine-tuning with Kubeflow

A production-ready pipeline for fine-tuning embedding models for Retrieval-Augmented Generation (RAG) using Kubeflow distributed training with local TensorBoard monitoring.

## 🎯 Overview

This project implements distributed PyTorch training for embedding fine-tuning with:
- **Kubeflow Trainer** for distributed training on KIND clusters
- **Dynamic hard negative sampling** during training
- **Local TensorBoard** with bind-mounted logs
- **Feast integration** for data management
- **Cross-platform Docker support**

## 🏗️ Architecture

```
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│   Training Data │───▶│   KIND Cluster     │───▶│  Local Machine  │
│   (Parquet)     │    │   (2 nodes)        │    │   TensorBoard   │
└─────────────────┘    └────────────────────┘    └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌────────────────────┐    ┌─────────────────┐
                        │ Bind Mount         │    │ ./training_     │
                        │ /tmp/rag-training  │───▶│ outputs/        │
                        │ -output            │    │                 │
                        └────────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python 3.11+** with UV package manager
- **Docker** (for KIND and training containers)
- **8GB+ RAM** (for distributed training)
- **kubectl** and **kind** (auto-installed by setup script)

## 🚀 Quick Start

### 1. Prepare Training Data

```bash
uv run prepare_training_data.py \
    --source-data feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet \
    --output-dir feature_repo/data \
    --hard-negatives-per-query 2 \
    --random-negative-ratio 0.3
```

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
├── kubeflow_embedding_training.py   # ✅ Training logic
├── prepare_training_data.py         # ✅ Data preparation
├── training-storage.yaml            # ✅ PVC configuration
│
├── feature_repo/                    # Feast data and features
├── training_outputs/               # ✅ Local output (bind mounted)
│   ├── tensorboard_logs/           # ← TensorBoard reads from here
│   └── fine_tuned_kubeflow_embeddings/
├── cached_model/                   # Pre-downloaded model cache
└── .venv/                         # UV virtual environment
```

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

## 📚 Integration

### Using Fine-tuned Model

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer('./training_outputs/fine_tuned_kubeflow_embeddings')

# Use for encoding
query_embedding = model.encode("What is machine learning?")
```

### Feast RAG Integration

```python
from feast.rag_retriever import FeastRAGRetriever

retriever = FeastRAGRetriever(
    question_encoder_model_name='./training_outputs/fine_tuned_kubeflow_embeddings',
    generator_model="your-llm-model",
    feast_repo_path="feature_repo",
    feature_view=wiki_passage_feature_view
)

answer = retriever.generate_answer("Your question", top_k=10)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with your data
4. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0.