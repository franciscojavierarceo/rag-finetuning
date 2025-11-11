# RAG Embedding Fine-tuning with Feast and Kubeflow

A complete pipeline for fine-tuning embedding models for Retrieval-Augmented Generation (RAG) systems using Feast for data management and Kubeflow for distributed training. Now with **Docker-based local development** for reliable cross-platform training.

## 🎯 Overview

This project implements a hybrid approach to embedding fine-tuning that:
- Uses **Feast offline store** for efficient training data storage
- Implements **dynamic hard negative sampling** during training
- Leverages **Kubeflow trainer** for distributed PyTorch training
- Provides **TensorBoard monitoring** for training metrics
- **Docker containerization** for reliable local and cloud deployment
- **Dynamic path resolution** for Kubernetes environments
- Integrates seamlessly with existing **Feast RAG pipelines**

## 🏗️ Architecture

```
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│   Source Data   │───▶│  Training Data     │───▶│   Kubeflow      │
│  (Wikipedia)    │    │  Preparation       │    │   Training      │
└─────────────────┘    └────────────────────┘    └─────────────────┘
                                │                          │
                                ▼                          ▼
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│ Feast Offline   │    │ Positive/Negative/ │    │  Fine-tuned     │
│ Store           │    │ Hard Negative      │    │  Model +        │
│ (Training Data) │    │ Pairs (660 total)  │    │  TensorBoard    │
└─────────────────┘    └────────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- UV package manager
- Docker (for containerized training)
- Kubeflow SDK 2.1.0+
- 8GB+ RAM (for model training)

## 🚀 Quick Start

### Option A: Simple Docker Training (Recommended)

The fastest way to get started with reliable containerized training:

```bash
# 1. Prepare training data
uv run prepare_training_data.py \
    --source-data feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet \
    --output-dir feature_repo/data \
    --hard-negatives-per-query 2 \
    --random-negative-ratio 0.3

# 2. Build training image and run everything
./simple-docker.sh all
```

This will:
✅ Build Docker training image
✅ Start local registry
✅ Start TensorBoard
✅ Run complete training pipeline
✅ Open TensorBoard at http://localhost:6006

### Option B: Docker Compose (Advanced)

For more complex setups with persistent volumes:

```bash
# Setup all services
./docker-run.sh setup

# Run training
./docker-run.sh train

# Open TensorBoard
./docker-run.sh tensorboard
```

### Option C: Kubernetes Distributed Training (Production)

For distributed training using KIND cluster with Kubeflow Trainer operator:

```bash
# 1. Prepare training data
uv run prepare_training_data.py \
    --source-data feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet \
    --output-dir feature_repo/data \
    --hard-negatives-per-query 2 \
    --random-negative-ratio 0.3

# 2. Setup KIND cluster with Kubeflow Trainer
./setup-kind-kubeflow.sh all

# 3. Deploy distributed training (Python - more reliable)
uv run deploy_kubeflow_training.py

# 4. Monitor training progress
kubectl get trainjobs -w
kubectl logs -f -l trainer.kubeflow.org/trainjob-ancestor-step=trainer
```

This will:
✅ Create KIND cluster with 3 nodes (1 control-plane, 2 workers)
✅ Install Kubeflow Trainer operator v2.1.0
✅ Deploy distributed PyTorch training (2 nodes)
✅ Run training with dynamic hard negative sampling
✅ Save models and logs to local filesystem

### Option D: Native Local Training

For development without Docker or Kubernetes (direct Python execution):

```bash
# 1. Prepare training data with custom parameters
uv run prepare_training_data.py \
    --source-data feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet \
    --output-dir feature_repo/data \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --hard-negatives-per-query 3 \
    --random-negative-ratio 0.3

# 2. Run standalone training locally (recommended for local development)
uv run standalone_embedding_training.py

# Alternative: Run with Kubeflow wrapper (may have path issues)
uv run kubeflow_embedding_training.py

# 3. Start TensorBoard for monitoring (in separate terminal)
uv run tensorboard --logdir=tensorboard_logs --host=0.0.0.0 --port=6006

# 4. Open TensorBoard in browser
open http://localhost:6006
```

**Advanced Local Training Options:**

```bash
# For longer training with more epochs (recommended for best results)
# Edit kubeflow_embedding_training.py to modify:
# func_args = {
#     "model_name": "all-MiniLM-L6-v2",
#     "epochs": "20",                    # Increase epochs for better convergence
#     "batch_size": "16",               # Larger batch size for stability
#     "learning_rate": "2e-6",          # Ultra-low LR for ContrastiveLoss
#     "max_samples": None,              # Use full dataset (remove limit)
#     "hard_negative_update_frequency": "3"  # Update negatives every 3 epochs
# }

# Alternative models to try:
# "sentence-transformers/all-mpnet-base-v2"           # Larger, more powerful
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
```

This will:
✅ Run complete hybrid training pipeline locally
✅ Generate dynamic hard negatives during training
✅ Use ContrastiveLoss for optimal positive similarity learning
✅ Save fine-tuned model to `fine_tuned_kubeflow_embeddings/`
✅ Log detailed metrics to TensorBoard
✅ Work on any machine with Python 3.11+ and UV

### Training Output

**Generated Files:**
- `feature_repo/data/embedding_training_data.parquet` - Main training dataset
- `feature_repo/data/query_embeddings.parquet` - Query embeddings for dynamic sampling
- `fine_tuned_kubeflow_embeddings/` - Fine-tuned model
- `tensorboard_logs/` - Training metrics and logs

**TensorBoard Metrics:**
- `Evaluation/Positive_Pair_Similarity` - Model performance on positive pairs
- `Hard_Negatives/New_Pairs_Added` - Dynamic negative sampling activity
- `Dataset/*_count` - Training dataset composition over time

## 📁 Project Structure

```
rag-finetuning/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Dockerfile                            # Container build configuration
├── docker-compose.yml                    # Docker Compose services
│
├── prepare_training_data.py               # Generate training data
├── kubeflow_embedding_training.py         # Main training script (dynamic paths)
│
├── simple-docker.sh                      # Simple Docker training (recommended)
├── docker-run.sh                         # Docker Compose wrapper
├── build-and-deploy.sh                   # Local deployment script
│
├── feature_repo/
│   ├── wiki_features.py                   # Feast feature definitions (inference)
│   ├── training_data_features.py          # Training data feature definitions
│   └── data/
│       ├── train-00000-of-00157_*.parquet # Source data
│       ├── embedding_training_data.parquet # Training pairs
│       └── query_embeddings.parquet       # Query embeddings
│
├── tensorboard_logs/                      # TensorBoard logs
└── fine_tuned_kubeflow_embeddings/        # Output models
```

### 🐳 Docker & Kubernetes Files

**Essential for Local Development:**
- `simple-docker.sh` - **Recommended**: Single-command Docker training
- `docker-run.sh` - Docker Compose with persistent volumes
- `Dockerfile` - Container build configuration
- `docker-compose.yml` - Multi-service setup
- `requirements.txt` - Python dependencies with pinned versions

**Production Kubernetes Training (Distributed):**
- `setup-kind-kubeflow.sh` - **Working**: KIND cluster with Kubeflow Trainer operator
- `deploy_kubeflow_training.py` - **Working**: Python deployment script (cross-platform)
- `deploy-kubeflow-training.sh` - Legacy bash deployment (deprecated)
- `setup-kind-cluster.sh` - Legacy KIND setup (deprecated)
- `quick-start.sh` - All-in-one KIND setup (deprecated)
- `local-deploy.sh` - Legacy deployment scripts

**Legacy/Optional:**
- `standalone_embed.sh` - Milvus setup (optional vector DB)

## ⚙️ Configuration Options

### Training Parameters

Edit `kubeflow_embedding_training.py` to customize training:

```python
job_id = client.train(
    trainer=CustomTrainer(
        func=hybrid_embedding_training,
        func_args={
            "model_name": "all-MiniLM-L6-v2",           # Base model
            "epochs": "5",                              # Training epochs
            "batch_size": "16",                         # Batch size
            "learning_rate": "2e-5",                    # Learning rate
            "max_samples": "200",                       # Limit samples (testing)
            "hard_negative_update_frequency": "3"       # Update hard negatives every N epochs
        },
        num_nodes=3,                                    # Distributed training nodes
        resources_per_node={"cpu": 2},                  # Resources per node
    ),
)
```

### Hardware Optimization (Desktop/Server)

**For high-performance desktop training, optimize Colima and training resources:**

```bash
# Optimize Colima for desktop hardware (Ubuntu/Linux)
colima stop
colima start --cpu 6 --memory 28 --disk 100 --mount-type virtiofs

# Example for different system sizes:
# High-end (32GB RAM, 16 cores): --cpu 14 --memory 24
# Medium (16GB RAM, 8 cores):    --cpu 6  --memory 12
# Your setup (62GB RAM, 8 cores): --cpu 6  --memory 28
```

**Then edit `deploy_kubeflow_training.py` for better performance:**

```python
# In deploy_kubeflow_training.py, update resources_per_node:
resources_per_node={
    "cpu": "3",              # 3 CPUs per node (6 total)
    "memory": "12Gi",        # 12GB per node (24GB total)
    # "nvidia.com/gpu": "1", # GPU support if available
},

func_args={
    "epochs": "20",          # More epochs with better hardware
    "batch_size": "32",      # Larger batches with more memory
    "max_samples": None,     # Use full dataset
}
```

**For maximum performance, consider native training:**
```bash
# Native training uses full desktop resources automatically
uv run kubeflow_embedding_training.py
```

### Data Generation Parameters

Customize training data generation:

```bash
uv run prepare_training_data.py \
    --source-data your_data.parquet \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --hard-negatives-per-query 3 \      # More hard negatives
    --random-negative-ratio 0.5          # More random negatives
```

## 🎯 Training Strategy

### Hybrid Negative Sampling

1. **In-batch Negatives** (Zero Overhead)
   - Uses `MultipleNegativesRankingLoss`
   - Other examples in batch serve as negatives
   - No additional computation required

2. **Pre-computed Hard Negatives** (High Quality)
   - Generated using base model similarity
   - Stored in Feast offline store
   - Provides challenging examples for learning

3. **Dynamic Hard Negatives** (Adaptive)
   - Periodically updated using current model
   - Every N epochs (configurable)
   - Keeps negatives relevant as model improves

### Data Composition

Typical training dataset composition:
- **~30% Positive pairs**: Title-text matches (ground truth)
- **~10% Random negatives**: Completely unrelated pairs
- **~60% Hard negatives**: Similar but incorrect matches

## 📊 Results and Evaluation

### Training Metrics

Monitor these key metrics in TensorBoard:

1. **`Evaluation/Positive_Pair_Similarity`**
   - Should increase over epochs
   - Target: >0.8 for good performance

2. **`Hard_Negatives/New_Pairs_Added`**
   - Shows dynamic sampling activity
   - Non-zero every N epochs

3. **`Dataset/Total_Size`**
   - Shows dataset growth from dynamic sampling

### Model Output

Fine-tuned model saved to:
```
fine_tuned_kubeflow_embeddings/
├── config.json                 # Model configuration
├── pytorch_model.bin            # Model weights
├── tokenizer.json              # Tokenizer
└── training_info.json          # Training metadata
```

## 🔌 Integration with Existing RAG Systems

### Option 1: Direct Model Loading

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer('./fine_tuned_kubeflow_embeddings')

# Use for encoding queries/documents
query_embedding = model.encode("What is machine learning?")
doc_embeddings = model.encode(["ML is...", "AI involves..."])
```

### Option 2: Feast RAG Integration

Update your existing Feast RAG pipeline:

```python
from feast.rag_retriever import FeastRAGRetriever

# Use with Feast RAG retriever
retriever = FeastRAGRetriever(
    question_encoder_model_name='./fine_tuned_kubeflow_embeddings',  # Your fine-tuned model
    generator_model="your-llm-model",
    feast_repo_path="feature_repo",
    feature_view=wiki_passage_feature_view
)

answer = retriever.generate_answer("Your question", top_k=10)
```

### Option 3: Update Feast Embeddings

Re-encode your corpus with the fine-tuned model:

```python
# Re-encode documents with fine-tuned model
from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('./fine_tuned_kubeflow_embeddings')

# Load your documents
df = pd.read_parquet('feature_repo/data/your_documents.parquet')

# Re-encode with fine-tuned model
new_embeddings = model.encode(df['text'].tolist())
df['embeddings'] = new_embeddings.tolist()

# Save updated embeddings
df.to_parquet('feature_repo/data/updated_embeddings.parquet')

# Update Feast online store
from feast import FeatureStore
fs = FeatureStore(repo_path="feature_repo")
fs.write_to_online_store("your_feature_view", df)
```

## 🐛 Troubleshooting

### Docker Issues

1. **TensorBoard not accessible at localhost:6006**
   ```bash
   # Stop Docker TensorBoard (often has architecture issues)
   docker stop rag-tensorboard && docker rm rag-tensorboard

   # Use native TensorBoard instead
   uv run tensorboard --logdir=tensorboard_logs --host=0.0.0.0 --port=6006
   ```

2. **KIND cluster kubelet issues (macOS)**
   ```
   Error: timed out waiting for node kind-worker to be ready
   Solution: Use Docker-based training instead (more reliable)
   ./simple-docker.sh all
   ```

3. **Docker container name conflicts**
   ```bash
   # Error: container name already in use
   docker stop rag-tensorboard rag-training rag-registry
   docker rm rag-tensorboard rag-training rag-registry
   ```

4. **Architecture mismatch warnings (Apple Silicon)**
   ```
   WARNING: platform (linux/amd64) does not match detected host platform (linux/arm64/v8)
   # This is expected and usually works fine, or use native training
   ```

### Native Local Training Issues

1. **ModuleNotFoundError or ImportError**
   ```bash
   # Install dependencies with UV
   uv sync

   # Or install manually if UV isn't working
   pip install -r requirements.txt
   ```

2. **Training data not found**
   ```bash
   # Make sure you generated training data first
   uv run prepare_training_data.py \
       --source-data feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet \
       --output-dir feature_repo/data

   # Verify the file exists
   ls -la feature_repo/data/embedding_training_data.parquet
   ```

3. **CUDA out of memory (if using GPU)**
   ```python
   # Edit kubeflow_embedding_training.py and reduce batch size:
   func_args = {
       "batch_size": "8",  # Reduce from 16 to 8
       # ... other parameters
   }
   ```

4. **Positive similarities decreasing instead of increasing**
   ```bash
   # This is fixed in the current version using ContrastiveLoss
   # Check TensorBoard at http://localhost:6006
   # Look for "Evaluation/positive_similarity" - should increase over epochs
   ```

5. **TensorBoard not showing metrics**
   ```bash
   # Make sure TensorBoard is pointing to correct log directory
   uv run tensorboard --logdir=tensorboard_logs --host=0.0.0.0 --port=6006

   # Check if logs directory exists and has content
   ls -la tensorboard_logs/
   ```

6. **Training taking too long**
   ```python
   # For faster testing, edit kubeflow_embedding_training.py:
   func_args = {
       "epochs": "5",        # Reduce epochs for testing
       "max_samples": "100", # Limit dataset size
       "batch_size": "32",   # Increase batch size if you have memory
   }
   ```

### Common Issues

5. **ImportError: cannot import name 'TrainJobTemplate'**
   ```
   Solution: Using correct Kubeflow SDK API (v2.1.0+)
   ```

6. **FileNotFoundError: Training data not found**
   ```bash
   # Run data preparation first
   uv run prepare_training_data.py --source-data your_data.parquet
   ```

7. **CUDA out of memory**
   ```python
   # Reduce batch size in training config
   "batch_size": "8",  # Instead of 16
   ```

8. **Training job exits immediately**
   ```bash
   # Check if training data exists
   ls feature_repo/data/embedding_training_data.parquet

   # Check logs for errors
   docker logs rag-training
   ```

### Debugging Training

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check training data composition:
```python
import pandas as pd
df = pd.read_parquet('feature_repo/data/embedding_training_data.parquet')
print(df['label'].value_counts())
print(f"Total samples: {len(df)}")
```

## 🔧 Advanced Configuration

### Custom Models

Use different base models:
```python
"model_name": "sentence-transformers/all-mpnet-base-v2",  # Larger model
"model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
```

### Distributed Training

Scale up training:
```python
trainer=CustomTrainer(
    # ... other params ...
    num_nodes=8,                          # More nodes
    resources_per_node={"cpu": 4, "gpu": 1},  # GPU support
)
```

### Custom Loss Functions

Modify training loss in `kubeflow_embedding_training.py`:
```python
# Use different loss functions
from sentence_transformers import losses
train_loss = losses.CosineSimilarityLoss(model=model)  # Alternative loss
```

### Native Local Training Performance Tips

**Optimal Configuration for Local Training:**

```python
# Edit kubeflow_embedding_training.py for best local performance
func_args = {
    "model_name": "all-MiniLM-L6-v2",    # Good balance of speed and quality
    "epochs": "50",                       # More epochs for convergence
    "batch_size": "32",                   # Increase if you have sufficient RAM/GPU
    "learning_rate": "2e-6",              # Ultra-low LR proven to work
    "max_samples": None,                  # Use full dataset for best results
    "hard_negative_update_frequency": "5" # Update every 5 epochs for efficiency
}
```

**Hardware Recommendations:**
- **RAM**: 8GB+ recommended (16GB+ for larger datasets)
- **GPU**: Optional but speeds up training significantly
- **Storage**: SSD recommended for faster data loading
- **CPU**: 4+ cores for efficient processing

**Training Time Estimates (on typical hardware):**
- **Small dataset (100 samples)**: 5-10 minutes
- **Medium dataset (1000 samples)**: 30-60 minutes
- **Full dataset (5000+ samples)**: 2-4 hours

**Memory Optimization:**
```python
# If running out of memory, try these settings:
func_args = {
    "batch_size": "8",          # Reduce batch size
    "max_samples": "1000",      # Limit dataset size
    "learning_rate": "1e-6",    # Can use even lower LR with smaller batches
}
```

**Monitoring Best Practices:**
1. **Always run TensorBoard**: `uv run tensorboard --logdir=tensorboard_logs --host=0.0.0.0 --port=6006`
2. **Watch positive similarity**: Should increase from ~0.6 to 0.85+ over training
3. **Monitor hard negative updates**: Should see periodic spikes in "Hard_Negatives/New_Pairs_Added"
4. **Check dataset growth**: Total dataset size should increase with hard negative updates

**Expected Results:**
- **Positive similarity**: 0.85-0.89 (excellent), 0.80-0.84 (good), <0.80 (needs more training)
- **Training progression**: Should see steady improvement over first 20-50 epochs
- **Model saturation**: Performance typically plateaus around 150-200 epochs

## 📚 Additional Resources

### Milvus Vector Database (Optional)

If you want to use Milvus for vector storage:

```bash
# Install Milvus in Docker (Linux/Mac)
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Follow instructions at: https://milvus.io/docs/install_standalone-docker.md

### References

- [Feast Documentation](https://docs.feast.dev/)
- [Kubeflow Trainer](https://www.kubeflow.org/docs/components/trainer/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG with Feast Tutorial](https://github.com/feast-dev/feast/blob/master/docs/tutorials/rag-with-docling.md)

### Original Examples

This project builds upon these excellent examples:
- [Distributed Workloads - Feast RAG](https://github.com/efazal/distributed-workloads/tree/main/examples/kfto-sft-feast-rag)
- [Fine-tuned Hybrid RAG](https://github.com/Nehanth/fine-tuned-hybrid-rag/blob/master/finetune/train_weights.py)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with your data
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0.


