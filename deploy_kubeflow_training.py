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
    print(f"🏗️  Running: docker build -t {image_name} .")
    result = subprocess.run(f"docker build -t {image_name} .", shell=True, text=True)
    if result.returncode != 0:
        print_error("Docker build failed")
        sys.exit(1)
    print_success("Docker image built successfully!")

    # Push to local registry
    print_status("Pushing image to local registry...")
    print(f"📤 Running: docker push {image_name}")
    result = subprocess.run(f"docker push {image_name}", shell=True, text=True)
    if result.returncode != 0:
        print_error("Docker push failed")
        sys.exit(1)

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
        print("⚙️  Training Configuration:")
        print("   📦 Model: all-MiniLM-L6-v2")
        print("   🔄 Epochs: 3 (distributed test)")
        print("   📊 Batch size: 4 per node")
        print("   🧠 Learning rate: 2e-6")
        print("   🎯 Max samples: 50 (testing)")
        print("   🔧 Nodes: 2 (distributed)")
        print("   💾 Resources: 3 CPU, 12Gi memory per node")

        # Initialize the Kubeflow Trainer client
        print("🔌 Initializing Kubeflow Trainer client...")
        client = TrainerClient()

        print("🚀 Submitting training job to Kubernetes...")
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

        print_success(f"Training job submitted with ID: {job_id}")
        print("📊 Monitor with: kubectl get trainjobs -w")
        print("📜 View logs with: kubectl logs -f job/{job_id}")

        return job_id

    except Exception as e:
        print_error(f"Failed to deploy training: {e}")
        import traceback
        print("🐛 Full error details:")
        traceback.print_exc()
        sys.exit(1)


def wait_for_job_start(job_id, timeout_minutes=5):
    """Wait for training job to start"""
    print("⏳ Waiting for job to start...")
    print("🔍 This may take a few minutes while:")
    print("   📥 Pulling PyTorch container images")
    print("   🏗️  Creating training pods")
    print("   ⚙️  Setting up distributed training")

    for i in range(timeout_minutes * 6):  # Check every 10 seconds
        # Check TrainJob status
        result = run_command(
            "kubectl get trainjobs -o jsonpath='{.items[*].status.conditions[-1].type}'",
            check=False
        )

        if "Running" in result.stdout:
            print("✅ Training job is running!")
            return True
        elif "Succeeded" in result.stdout:
            print("✅ Training job completed!")
            return True

        # Show pod status for more detailed progress
        pod_result = run_command(
            f"kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer --no-headers 2>/dev/null || echo 'No pods yet'",
            check=False
        )

        if "No pods yet" not in pod_result.stdout and pod_result.stdout.strip():
            # Parse pod status
            lines = pod_result.stdout.strip().split('\n')
            statuses = [line.split()[2] for line in lines if line.strip()]
            status_summary = ', '.join(set(statuses))
            print(f"   📦 Pods status: {status_summary} ({i+1}/{timeout_minutes * 6})")
        else:
            print(f"   ⌛ Still waiting for pods to be created... ({i+1}/{timeout_minutes * 6})")

        time.sleep(10)

    print("⚠️ Job may not have started yet. This is normal for large PyTorch images.")
    print("🔍 Check manually with: kubectl get trainjobs -w")
    return False


def monitor_training():
    """Show training status and monitoring commands"""
    print_status("Current training status...")

    print("\n📊 Training Jobs:")
    run_command("kubectl get trainjobs -o wide", check=False)

    print("\n🔧 Training Pods:")
    result = run_command("kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer", check=False)
    if result.returncode != 0 or not result.stdout.strip():
        print("   ⚠️  No training pods found yet (still starting up)")

    print("\n⚙️  All Related Pods (JobSet):")
    run_command("kubectl get pods -l jobset.sigs.k8s.io", check=False)

    print("\n📜 Recent Events:")
    event_result = run_command("kubectl get events --sort-by=.metadata.creationTimestamp | tail -10", check=False)
    if event_result.returncode != 0 or not event_result.stdout.strip():
        print("   ℹ️  No recent events found")

    print("\n💡 Next Steps - Monitor your training:")
    print("   🔍 Watch jobs:         kubectl get trainjobs -w")
    print("   📜 Stream logs:        kubectl logs -f -l trainer.kubeflow.org/trainjob-ancestor-step=trainer")
    print("   🔧 Job details:        kubectl describe trainjobs")
    print("   📦 Check all pods:     kubectl get pods -A")
    print("   🌐 Cluster info:       kubectl get nodes")

    print(f"\n🎯 Training Progress:")
    print("   1️⃣  Pods should move from 'ContainerCreating' → 'Running'")
    print("   2️⃣  Training logs will show epoch progress")
    print("   3️⃣  Model will be saved to './fine_tuned_kubeflow_embeddings/'")
    print("   4️⃣  TensorBoard logs will be in './tensorboard_logs/'")

    print(f"\n💡 If pods are stuck 'ContainerCreating':")
    print("   🐳 Large PyTorch images take 5-15 minutes to download")
    print("   ⏰ This is normal - be patient!")
    print("   🔍 Check with: kubectl describe pods")


def main():
    """Main deployment function"""
    print_banner()

    print("🚀 Starting Kubeflow distributed training deployment...")
    print("📋 This process will:")
    print("   1️⃣  Check prerequisites (cluster, registry, data)")
    print("   2️⃣  Build and push Docker training image")
    print("   3️⃣  Deploy distributed training to Kubernetes")
    print("   4️⃣  Monitor initial startup progress")
    print()

    # Check prerequisites
    print("=" * 60)
    print("🔍 STEP 1: CHECKING PREREQUISITES")
    print("=" * 60)
    check_prerequisites()

    # Build and push training image
    print("\n" + "=" * 60)
    print("🐳 STEP 2: BUILDING DOCKER IMAGE")
    print("=" * 60)
    build_and_push_image()

    # Deploy training job
    print("\n" + "=" * 60)
    print("☸️  STEP 3: DEPLOYING TO KUBERNETES")
    print("=" * 60)
    job_id = deploy_training()

    # Wait for job to start
    print("\n" + "=" * 60)
    print("⏳ STEP 4: MONITORING STARTUP")
    print("=" * 60)
    started = wait_for_job_start(job_id)

    # Show monitoring information
    print("\n" + "=" * 60)
    print("📊 STEP 5: CURRENT STATUS")
    print("=" * 60)
    monitor_training()

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print_success(f"Training job '{job_id}' deployed to Kubeflow cluster!")

    if started:
        print("✅ Job is running - check logs for training progress")
    else:
        print("⏳ Job is starting - large images take time to download")

    print("\n🎯 What happens next:")
    print("   📥 PyTorch containers finish downloading")
    print("   🔄 Distributed training begins automatically")
    print("   📊 Training metrics logged to TensorBoard")
    print("   💾 Fine-tuned model saved locally")

    print("\n🔍 Monitor with:")
    print(f"   kubectl logs -f -l trainer.kubeflow.org/trainjob-ancestor-step=trainer")

    print("\n🎯 When training completes, find your model in:")
    print("   📁 ./fine_tuned_kubeflow_embeddings/")
    print("   📈 ./tensorboard_logs/")


if __name__ == "__main__":
    main()