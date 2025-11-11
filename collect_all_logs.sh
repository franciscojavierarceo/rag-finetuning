#!/bin/bash
# Script to collect all training logs from different sources

echo "🔍 COLLECTING ALL RAG TRAINING LOGS"
echo "=================================="

# Create output directory
mkdir -p all_training_logs
cd all_training_logs

echo "📂 1. TensorBoard Logs"
echo "----------------------"
if [ -d ../tensorboard_logs ]; then
    cp -r ../tensorboard_logs ./tensorboard_logs_backup
    echo "✅ Copied TensorBoard logs"
    find ../tensorboard_logs -name "*.tfevents*" -exec ls -la {} \;
else
    echo "❌ No TensorBoard logs found"
fi

echo ""
echo "📂 2. Model Output"
echo "------------------"
if [ -d ../fine_tuned_kubeflow_embeddings ]; then
    cp -r ../fine_tuned_kubeflow_embeddings ./model_output
    echo "✅ Copied trained model"
    ls -la ../fine_tuned_kubeflow_embeddings/
else
    echo "❌ No trained model found"
fi

echo ""
echo "📂 3. Training Data"
echo "-------------------"
if [ -f ../feature_repo/data/embedding_training_data.parquet ]; then
    cp ../feature_repo/data/embedding_training_data.parquet ./training_data.parquet
    echo "✅ Copied training data"
    echo "Size: $(du -h ../feature_repo/data/embedding_training_data.parquet | cut -f1)"
else
    echo "❌ No training data found"
fi

echo ""
echo "📂 4. Kubernetes Distributed Training Logs"
echo "--------------------------------------------"
# Check if we're collecting from Kubernetes training
if kubectl get trainjobs >/dev/null 2>&1; then
    echo "🔍 Found Kubernetes training - collecting distributed logs..."

    # Get all training job names
    JOBS=$(kubectl get trainjobs -o name 2>/dev/null | head -5)
    if [ ! -z "$JOBS" ]; then
        echo "📋 Found training jobs:"
        kubectl get trainjobs

        mkdir -p kubernetes_logs

        # Collect logs from each training job
        for job in $JOBS; do
            JOB_NAME=$(echo $job | cut -d'/' -f2)
            echo "📜 Collecting logs for job: $JOB_NAME"

            # Get all pods for this job
            PODS=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$JOB_NAME -o name 2>/dev/null)

            if [ ! -z "$PODS" ]; then
                echo "  📦 Found pods for $JOB_NAME:"
                for pod in $PODS; do
                    POD_NAME=$(echo $pod | cut -d'/' -f2)
                    echo "    - $POD_NAME"

                    # Collect logs from each pod
                    kubectl logs $POD_NAME > kubernetes_logs/${JOB_NAME}_${POD_NAME}.log 2>/dev/null || \
                        echo "    ❌ Failed to get logs for $POD_NAME"
                done
            else
                echo "  ⚠️  No pods found for job $JOB_NAME"
            fi
        done

        # Collect training job descriptions
        echo "📋 Collecting job descriptions..."
        kubectl get trainjobs -o yaml > kubernetes_logs/trainjobs.yaml 2>/dev/null
        kubectl describe trainjobs > kubernetes_logs/trainjobs_describe.txt 2>/dev/null

        # Extract files from PVC (TensorBoard logs and models)
        echo "📦 Extracting training outputs from persistent storage..."
        kubectl run temp-extractor --image=busybox --restart=Never --rm --attach --stdin=false \
          --overrides='{
            "spec": {
              "containers": [
                {
                  "name": "temp-extractor",
                  "image": "busybox",
                  "command": ["sleep", "60"],
                  "volumeMounts": [
                    {
                      "name": "training-storage",
                      "mountPath": "/workspace/outputs"
                    }
                  ]
                }
              ],
              "volumes": [
                {
                  "name": "training-storage",
                  "persistentVolumeClaim": {
                    "claimName": "rag-training-storage"
                  }
                }
              ]
            }
          }' -- sleep 60 &

        echo "⏳ Waiting for extraction pod..."
        kubectl wait --for=condition=Ready pod/temp-extractor --timeout=30s 2>/dev/null

        # Copy TensorBoard logs from PVC
        kubectl cp temp-extractor:/workspace/outputs/tensorboard_logs ./persistent_tensorboard_logs 2>/dev/null && \
          echo "✅ TensorBoard logs extracted from PVC" || \
          echo "ℹ️  No TensorBoard logs in PVC yet"

        # Copy models from PVC
        kubectl cp temp-extractor:/workspace/outputs/fine_tuned_kubeflow_embeddings ./persistent_models 2>/dev/null && \
          echo "✅ Models extracted from PVC" || \
          echo "ℹ️  No models in PVC yet"

        # Clean up
        kubectl delete pod temp-extractor --force --grace-period=0 2>/dev/null

        echo "✅ Kubernetes logs collected in kubernetes_logs/"
    else
        echo "ℹ️  No training jobs found in Kubernetes"
    fi
else
    echo "ℹ️  No Kubernetes cluster found - skipping distributed logs"
fi

echo ""
echo "📂 5. Background Process Logs (if available)"
echo "---------------------------------------------"
# This would require you to manually copy outputs from your terminal
echo "ℹ️  Manual step: Copy any terminal output from background training processes"

echo ""
echo "📂 6. Create Combined Summary"
echo "-----------------------------"
cat > training_summary.md << 'EOF'
# RAG Embedding Training Summary

## Training Configuration
- Model: all-MiniLM-L6-v2
- Loss Function: ContrastiveLoss
- Learning Rate: 2e-6
- Batch Size: 16
- Epochs: 25

## Files Collected
- `tensorboard_logs_backup/`: All TensorBoard event files (old runs)
- `persistent_tensorboard_logs/`: TensorBoard logs from PVC (current run)
- `persistent_models/`: Fine-tuned models from PVC (current run)
- `model_output/`: Fine-tuned model and training info (local runs)
- `training_data.parquet`: Training dataset used
- `kubernetes_logs/`: Distributed training logs from all nodes/pods
  - `{job_name}_{pod_name}.log`: Individual pod logs
  - `trainjobs.yaml`: Training job configurations
  - `trainjobs_describe.txt`: Detailed job status
- `training_summary.md`: This summary

## TensorBoard Access
To view training metrics:
```bash
# For current Kubernetes training (with persistent storage):
uv run tensorboard --logdir=all_training_logs/persistent_tensorboard_logs --host=0.0.0.0 --port=6006

# For old local training runs:
uv run tensorboard --logdir=all_training_logs/tensorboard_logs_backup --host=0.0.0.0 --port=6006

# Open http://localhost:6006
```

## Key Metrics to Check
1. Evaluation/positive_similarity (should increase over epochs)
2. Hard_Negatives/New_Pairs_Added (shows dynamic sampling)
3. Dataset/*_count (dataset composition changes)
4. Final_Evaluation/* (final model quality metrics)

## Model Usage
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./model_output/')
embeddings = model.encode(["your text here"])
```
EOF

echo "✅ Created training summary"

echo ""
echo "🎯 COLLECTION COMPLETE!"
echo "========================"
echo "📁 All logs collected in: $(pwd)"
echo "📊 View summary: cat training_summary.md"
echo "🔗 TensorBoard: http://localhost:6006"
echo ""
echo "📋 Archive command:"
echo "tar -czf rag_training_logs_$(date +%Y%m%d_%H%M%S).tar.gz ."