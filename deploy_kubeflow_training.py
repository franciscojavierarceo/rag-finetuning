#!/usr/bin/env python3
"""
Deploy RAG Embedding Training to Kubeflow Cluster

This script builds, pushes, and runs distributed training in Kubernetes using
the Kubeflow Trainer operator. It replaces the bash version with cleaner Python code.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

# Import training function
from kubeflow_embedding_training import hybrid_embedding_training
from kubeflow.trainer import TrainerClient, CustomTrainer


class Colors:
    """ANSI color codes for pretty output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'  # No Color


def print_banner():
    """Print deployment banner"""
    print(f"{Colors.PURPLE}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          🚀 KUBEFLOW DISTRIBUTED TRAINING                ║")
    print("║                 RAG Embedding Fine-tuning                ║")
    print("║                                                           ║")
    print("║         Production-scale Kubernetes Training             ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")


def print_status(message):
    """Print info message"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def run_command(cmd, description="", check=True):
    """Run shell command with error handling"""
    if description:
        print_status(description)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print_error(f"Command failed: {cmd}")
            print_error(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    except Exception as e:
        print_error(f"Failed to run command: {e}")
        sys.exit(1)


def check_prerequisites():
    """Check all prerequisites are met"""
    print_status("Checking prerequisites...")

    # Check if KIND cluster exists
    result = run_command("kind get clusters", check=False)
    if "rag-kubeflow" not in result.stdout:
        print_error("KIND cluster 'rag-kubeflow' not found")
        print_error("Run: ./setup-kind-kubeflow.sh all")
        sys.exit(1)

    # Check kubectl context
    result = run_command("kubectl config current-context", check=False)
    if "kind-rag-kubeflow" not in result.stdout:
        print_warning("Setting kubectl context to kind-rag-kubeflow")
        run_command("kubectl config use-context kind-rag-kubeflow")

    # Check if cluster is accessible
    result = run_command("kubectl get nodes", check=False)
    if result.returncode != 0:
        print_error("Cannot connect to Kubernetes cluster")
        sys.exit(1)

    # Check if Kubeflow Trainer is installed
    result = run_command("kubectl get crd trainjobs.trainer.kubeflow.org", check=False)
    if result.returncode != 0:
        print_error("Kubeflow Trainer is not installed")
        print_error("Run: ./setup-kind-kubeflow.sh kubeflow")
        sys.exit(1)

    # Check if registry is accessible
    result = run_command("curl -sf http://localhost:5001/v2/", check=False)
    if result.returncode != 0:
        print_error("Local registry not accessible at localhost:5001")
        print_error("Run: ./setup-kind-kubeflow.sh cluster")
        sys.exit(1)

    # Check if training data exists
    if not Path("feature_repo/data/embedding_training_data.parquet").exists():
        print_warning("Training data not found")
        print_warning("Run: uv run prepare_training_data.py first")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(1)

    print_success("Prerequisites check passed!")


def build_and_push_image():
    """Build and push training image to local registry"""
    print_status("Building and pushing training image...")

    image_name = "localhost:5001/rag-embedding-training:latest"

    # Build the Docker image
    print_status(f"Building Docker image: {image_name}")
    run_command(f"docker build -t {image_name} .", "Building Docker image...")

    # Push to local registry
    print_status("Pushing image to local registry...")
    run_command(f"docker push {image_name}", "Pushing to registry...")

    # Verify image was pushed
    result = run_command("curl -sf http://localhost:5001/v2/rag-embedding-training/tags/list", check=False)
    if result.returncode == 0 and "latest" in result.stdout:
        print_success("Image successfully pushed to registry")
    else:
        print_error("Failed to push image to registry")
        sys.exit(1)


def deploy_training():
    """Deploy training job to Kubeflow"""
    print_status("Deploying RAG embedding training to Kubeflow...")

    try:
        print("🐍 Creating Kubeflow TrainJob...")

        # Initialize the Kubeflow Trainer client
        client = TrainerClient()

        # Create distributed training job
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
                    "cpu": "3",  # Optimized for desktop hardware
                    "memory": "12Gi",  # More memory for better performance
                    # Note: GPU support can be added here if available
                },
            ),
        )

        print(f"🚀 Training job submitted with ID: {job_id}")
        print(f"📊 Monitor with: kubectl get trainjobs -w")
        print(f"📜 View logs with: kubectl logs -f job/{job_id}")

        return job_id

    except Exception as e:
        print_error(f"Failed to deploy training: {e}")
        sys.exit(1)


def wait_for_job_start(job_id, timeout_minutes=5):
    """Wait for training job to start"""
    print("⏳ Waiting for job to start...")

    for i in range(timeout_minutes * 6):  # Check every 10 seconds
        result = run_command(
            "kubectl get trainjobs -o jsonpath='{.items[*].status.conditions[-1].type}'",
            check=False
        )

        if "Running" in result.stdout or "Succeeded" in result.stdout:
            print("✅ Training job is running!")
            return True

        time.sleep(10)
        print(f"   Still waiting... ({i+1}/{timeout_minutes * 6})")

    print("⚠️ Job may not have started. Check manually.")
    return False


def monitor_training():
    """Show training status and monitoring commands"""
    print_status("Monitoring training progress...")

    print("\n📊 Training Jobs:")
    run_command("kubectl get trainjobs -o wide")

    print("\n🔧 Related Pods:")
    result = run_command("kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer", check=False)
    if result.returncode != 0:
        print("No training pods found yet (still starting up)")

    print("\n📜 Recent Events:")
    run_command("kubectl get events --sort-by=.metadata.creationTimestamp --field-selector type!=Normal | tail -10", check=False)

    print("\n💡 Monitoring Commands:")
    print("   Watch jobs:     kubectl get trainjobs -w")
    print("   View logs:      kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=trainer -f")
    print("   Job details:    kubectl describe trainjobs")
    print("   All pods:       kubectl get pods -A")


def main():
    """Main deployment function"""
    print_banner()

    # Check prerequisites
    check_prerequisites()

    # Build and push training image
    build_and_push_image()

    # Deploy training job
    job_id = deploy_training()

    # Wait for job to start
    wait_for_job_start(job_id)

    # Show monitoring information
    monitor_training()

    print_success("Training job deployed to Kubeflow!")
    print(f"\n🎯 Your training job '{job_id}' is now running in the cluster!")
    print("🔍 Use the monitoring commands above to track progress.")


if __name__ == "__main__":
    main()