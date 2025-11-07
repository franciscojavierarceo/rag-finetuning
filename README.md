# RAG Embedding Fine-tuning with Feast and Kubeflow

A complete pipeline for fine-tuning embedding models for Retrieval-Augmented Generation (RAG) systems using Feast for data management and Kubeflow for distributed training.

## 🎯 Overview

This project implements a hybrid approach to embedding fine-tuning that:
- Uses **Feast offline store** for efficient training data storage
- Implements **dynamic hard negative sampling** during training
- Leverages **Kubeflow trainer** for distributed PyTorch training
- Provides **TensorBoard monitoring** for training metrics
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
- Kubeflow SDK 2.1.0+
- 8GB+ RAM (for model training)

## 🚀 Quick Start

### Step 1: Prepare Training Data

Generate labeled training pairs (positive/negative/hard negative) from your source data:

```bash
# Generate training data with positive, negative, and hard negative pairs
uv run prepare_training_data.py \
    --source-data feature_repo/data/train-00000-of-00157_sample_with_timestamp_chunked.parquet \
    --output-dir feature_repo/data \
    --hard-negatives-per-query 2 \
    --random-negative-ratio 0.3
```

**Output:**
- `feature_repo/data/embedding_training_data.parquet` - Main training dataset
- `feature_repo/data/query_embeddings.parquet` - Query embeddings for dynamic sampling

### Step 2: Run Kubeflow Fine-tuning

Execute the complete fine-tuning pipeline with distributed training:

```bash
# Run embedding fine-tuning with Kubeflow trainer
uv run kubeflow_embedding_training.py
```

**This will:**
- Load training data from Feast offline store
- Fine-tune `all-MiniLM-L6-v2` using hybrid negative sampling
- Update hard negatives every 3 epochs
- Log metrics to TensorBoard
- Save the fine-tuned model

### Step 3: Monitor Training

Open TensorBoard to monitor training progress:

```bash
tensorboard --logdir=./tensorboard_logs
```

**Available Metrics:**
- `Evaluation/Positive_Pair_Similarity` - Model performance on positive pairs
- `Hard_Negatives/New_Pairs_Added` - Dynamic negative sampling activity
- `Dataset/*_count` - Training dataset composition over time

## 📁 Project Structure

```
rag-finetuning/
├── README.md                              # This file
├── prepare_training_data.py               # Generate training data
├── kubeflow_embedding_training.py         # Main training script
├── feature_repo/
│   ├── wiki_features.py                   # Feast feature definitions (inference)
│   ├── training_data_features.py          # Training data feature definitions
│   └── data/
│       ├── train-00000-of-00157_*.parquet # Source data
│       ├── embedding_training_data.parquet # Training pairs
│       └── query_embeddings.parquet       # Query embeddings
├── tensorboard_logs/                      # TensorBoard logs
└── fine_tuned_kubeflow_embeddings/        # Output models
```

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

### Common Issues

1. **ImportError: cannot import name 'TrainJobTemplate'**
   ```
   Solution: Using correct Kubeflow SDK API (v2.1.0+)
   ```

2. **FileNotFoundError: Training data not found**
   ```bash
   # Run data preparation first
   uv run prepare_training_data.py --source-data your_data.parquet
   ```

3. **CUDA out of memory**
   ```python
   # Reduce batch size in training config
   "batch_size": "8",  # Instead of 16
   ```

4. **TensorBoard not showing metrics**
   ```bash
   # Check log directory exists
   ls tensorboard_logs/
   # Start TensorBoard with correct path
   tensorboard --logdir=./tensorboard_logs --port 6006
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


