#!/bin/bash

# Deploy RAG Embedding Training to Kubeflow Cluster
# This script builds, pushes, and runs distributed training in Kubernetes

set -e

# Configuration
CLUSTER_NAME="rag-kubeflow"
REGISTRY_PORT="5001"
IMAGE_NAME="localhost:${REGISTRY_PORT}/rag-embedding-training"
IMAGE_TAG="latest"
NAMESPACE="default"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║          🚀 KUBEFLOW DISTRIBUTED TRAINING                ║"
    echo "║                 RAG Embedding Fine-tuning                ║"
    echo "║                                                           ║"
    echo "║         Production-scale Kubernetes Training             ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if KIND cluster exists
    if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
        print_error "KIND cluster '${CLUSTER_NAME}' not found"
        print_info "Run: ./setup-kind-kubeflow.sh all"
        exit 1
    fi

    # Check kubectl context
    if ! kubectl config current-context | grep -q "kind-${CLUSTER_NAME}"; then
        print_warning "Setting kubectl context to kind-${CLUSTER_NAME}"
        kubectl config use-context "kind-${CLUSTER_NAME}"
    fi

    # Check if cluster is ready
    if ! kubectl get nodes >/dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if Kubeflow Trainer is installed
    if ! kubectl get crd trainjobs.trainer.kubeflow.org >/dev/null 2>&1; then
        print_error "Kubeflow Trainer is not installed"
        print_info "Run: ./setup-kind-kubeflow.sh kubeflow"
        exit 1
    fi

    # Check if registry is accessible
    if ! curl -sf http://localhost:${REGISTRY_PORT}/v2/ >/dev/null; then
        print_error "Local registry not accessible at localhost:${REGISTRY_PORT}"
        print_info "Run: ./setup-kind-kubeflow.sh cluster"
        exit 1
    fi

    # Check if training data exists
    if [[ ! -f "feature_repo/data/embedding_training_data.parquet" ]]; then
        print_warning "Training data not found"
        print_info "Run: uv run prepare_training_data.py first"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    print_success "Prerequisites check passed!"
}

# Build and push training image
build_and_push() {
    print_status "Building and pushing training image..."

    # Build the Docker image
    print_status "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

    # Push to local registry
    print_status "Pushing image to local registry..."
    docker push "${IMAGE_NAME}:${IMAGE_TAG}"

    # Verify image was pushed
    if curl -sf "http://localhost:${REGISTRY_PORT}/v2/rag-embedding-training/tags/list" | grep -q "latest"; then
        print_success "Image successfully pushed to registry"
    else
        print_error "Failed to push image to registry"
        exit 1
    fi
}

# Deploy training job
deploy_training() {
    print_status "Deploying RAG embedding training to Kubeflow..."

    # Create training script for Kubernetes
    cat > k8s_training_script.py <<EOF
#!/usr/bin/env python3

import os
import sys

# Add the workspace to Python path
sys.path.insert(0, '/workspace')

# Import our training function
from kubeflow_embedding_training import hybrid_embedding_training

if __name__ == "__main__":
    print("🚀 Starting Kubernetes-based RAG embedding training...")

    # Set up environment for Kubernetes
    os.environ.setdefault('PROJECT_DIR', '/workspace')
    os.environ.setdefault('PYTHONPATH', '/workspace')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')

    # Training configuration - optimized for distributed training
    func_args = {
        "model_name": "all-MiniLM-L6-v2",
        "epochs": "5",  # Start with fewer epochs for distributed test
        "batch_size": "8",  # Smaller batch size per node
        "learning_rate": "2e-6",
        "max_samples": "100",  # Smaller dataset for testing
        "feast_repo_path": "feature_repo",
        "hard_negative_update_frequency": "2"
    }

    print(f"📊 Training configuration: {func_args}")

    try:
        result = hybrid_embedding_training(**func_args)
        print(f"🎉 Training completed successfully: {result}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

    # Create Python training deployment using our existing training function
    python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/farceo/dev/rag-finetuning')

from kubeflow.trainer import TrainerClient, CustomTrainer
from kubeflow_embedding_training import hybrid_embedding_training

def main():
    print("🐍 Creating Kubeflow TrainJob...")

    # Initialize the Kubeflow Trainer client (without local backend for K8s)
    client = TrainerClient()

    # Create distributed training job using our existing function
    job_id = client.train(
        runtime=client.get_runtime("torch-distributed"),
        trainer=CustomTrainer(
            func=hybrid_embedding_training,
            func_args={
                "model_name": "all-MiniLM-L6-v2",
                "epochs": "3",  # Start with fewer epochs for distributed test
                "batch_size": "4",  # Smaller batch size per node for testing
                "learning_rate": "2e-6",
                "max_samples": "50",  # Small dataset for testing
                "feast_repo_path": "feature_repo",
                "hard_negative_update_frequency": "2"
            },
            num_nodes=2,  # Use 2 nodes for distributed training
            resources_per_node={
                "cpu": "1",
                "memory": "1500Mi",  # Reduced to fit KIND node capacity (~1.9GB per worker)
                # Note: No GPU in KIND cluster, but our training works on CPU
            },
        ),
    )

    print(f"🚀 Training job submitted with ID: {job_id}")
    print(f"📊 Monitor with: kubectl get trainjobs -w")
    print(f"📜 View logs with: kubectl logs -f job/{job_id}")

    return job_id

if __name__ == "__main__":
    job_id = main()

    # Wait for job to start
    import subprocess
    import time

    print("⏳ Waiting for job to start...")
    for i in range(30):  # Wait up to 5 minutes
        result = subprocess.run([
            "kubectl", "get", "trainjobs", "-o", "jsonpath='{.items[*].status.conditions[-1].type}'"
        ], capture_output=True, text=True)

        if "Running" in result.stdout or "Succeeded" in result.stdout:
            print("✅ Training job is running!")
            break

        time.sleep(10)
        print(f"   Still waiting... ({i+1}/30)")
    else:
        print("⚠️ Job may not have started. Check manually.")

    print("\n🔍 Current job status:")
    subprocess.run(["kubectl", "get", "trainjobs", "-o", "wide"])

EOF

    print_success "Training job deployed to Kubeflow!"
}

# Monitor training
monitor_training() {
    print_status "Monitoring training progress..."

    echo ""
    echo "📊 Training Jobs:"
    kubectl get trainjobs -o wide

    echo ""
    echo "🔧 Related Pods:"
    kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer

    echo ""
    echo "📜 Recent Events:"
    kubectl get events --sort-by=.metadata.creationTimestamp --field-selector type!=Normal | tail -10

    echo ""
    echo "💡 Monitoring Commands:"
    echo "   Watch jobs:     kubectl get trainjobs -w"
    echo "   View logs:      kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=trainer -f"
    echo "   Job details:    kubectl describe trainjobs"
    echo "   All pods:       kubectl get pods -A"
    echo ""
}

# Show training logs
show_logs() {
    print_status "Showing training logs..."

    # Find the training pods
    local pods=$(kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer --no-headers -o name)

    if [[ -z "$pods" ]]; then
        print_warning "No training pods found"
        echo "Available pods:"
        kubectl get pods --no-headers
    else
        print_success "Found training pods:"
        echo "$pods"
        echo ""
        echo "📜 Streaming logs (press Ctrl+C to exit):"
        kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=trainer -f --prefix=true
    fi
}

# Cleanup training resources
cleanup() {
    print_status "Cleaning up training resources..."

    # Delete all training jobs
    kubectl delete trainjobs --all --timeout=60s

    # Clean up any remaining pods
    kubectl delete pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer --timeout=60s

    # Clean up temporary files
    rm -f k8s_training_script.py

    print_success "Cleanup completed!"
}

# Show help
show_help() {
    echo "Kubeflow RAG Training Deployment Commands:"
    echo ""
    echo "Deployment:"
    echo "  $0 deploy      - Build image and deploy training"
    echo "  $0 build       - Build and push Docker image only"
    echo "  $0 train       - Deploy training job only"
    echo ""
    echo "Monitoring:"
    echo "  $0 status      - Show training status"
    echo "  $0 logs        - Stream training logs"
    echo "  $0 monitor     - Watch training progress"
    echo ""
    echo "Management:"
    echo "  $0 cleanup     - Remove all training resources"
    echo "  $0 restart     - Cleanup and redeploy"
    echo ""
    echo "Examples:"
    echo "  $0 deploy      # Complete deployment"
    echo "  $0 logs        # Watch training progress"
    echo "  $0 cleanup     # Clean up when done"
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            print_banner
            check_prerequisites
            build_and_push
            deploy_training
            monitor_training
            ;;
        "build")
            check_prerequisites
            build_and_push
            ;;
        "train")
            check_prerequisites
            deploy_training
            monitor_training
            ;;
        "status"|"monitor")
            monitor_training
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            cleanup
            sleep 5
            print_banner
            check_prerequisites
            build_and_push
            deploy_training
            monitor_training
            ;;
        "help"|*)
            show_help
            exit 0
            ;;
    esac
}

# Run main function
main "$@"