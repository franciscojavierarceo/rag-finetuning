#!/usr/bin/env python3
"""
Deploy Distributed RAG Embedding Training to Kubeflow Cluster
Enhanced for multi-node distributed training and large-scale data processing
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

# Import training function
from distributed_training import distributed_hybrid_embedding_training
from kubeflow.trainer import TrainerClient, CustomTrainer


class Colors:
    """ANSI color codes for pretty output"""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    NC = "\033[0m"  # No Color


def print_banner():
    """Print deployment banner"""
    print(f"{Colors.PURPLE}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║       🚀 DISTRIBUTED KUBEFLOW TRAINING                   ║")
    print("║            Multi-Node RAG Embedding Fine-tuning          ║")
    print("║                                                           ║")
    print("║       Production-scale Distributed Training              ║")
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
    print_status("Checking prerequisites for distributed training...")

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
    result = run_command(
        "kubectl get crd trainjobs.trainer.kubeflow.org", check=False
    )
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
        print_warning("Preparing distributed training data...")

        # Check for source data
        source_files = list(Path("feature_repo/data").glob("train-*.parquet"))
        if source_files:
            largest_file = max(source_files, key=lambda x: x.stat().st_size)
            print_status(f"Using source file: {largest_file}")

            # Run distributed data preparation
            cmd = f"uv run prepare_distributed_data.py --input-data {largest_file} --chunk-size 50000 --max-workers 6"
            run_command(cmd, "Preparing distributed training data...")
        else:
            print_error("No source training data found")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != "y":
                sys.exit(1)

    print_success("Prerequisites check passed!")


def build_and_push_image():
    """Build and push training image to local registry"""
    print_status("Building distributed training image...")

    image_name = "localhost:5001/rag-distributed-training:latest"

    # Check if image already exists in registry
    result = subprocess.run(
        "curl -sf http://localhost:5001/v2/rag-distributed-training/tags/list",
        shell=True, capture_output=True, text=True
    )
    if result.returncode == 0 and "latest" in result.stdout:
        print_success("Distributed training image already exists in registry - skipping build")
        print(f"✅ Found: {image_name}")
        return

    # Build the Docker image
    print_status(f"Building Docker image: {image_name}")
    print(f"🏗️  Running: docker build -t {image_name} .")
    result = subprocess.run(
        f"docker build -t {image_name} .", shell=True, text=True
    )
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
    result = run_command(
        "curl -sf http://localhost:5001/v2/rag-distributed-training/tags/list",
        check=False,
    )
    if result.returncode == 0 and "latest" in result.stdout:
        print_success("Image successfully pushed to registry")
    else:
        print_error("Failed to push image to registry")
        sys.exit(1)


def deploy_distributed_training(num_nodes=8, cpu_per_node=4, memory_per_node=8):
    """Deploy distributed training job to Kubeflow"""
    print_status("Deploying distributed RAG embedding training to Kubeflow...")

    try:
        print("🐍 Creating Distributed Kubeflow TrainJob...")
        print("⚙️  Distributed Training Configuration:")
        print("   📦 Model: all-mpnet-base-v2 (larger model for production)")
        print(f"   🔄 Nodes: {num_nodes} (distributed across cluster)")
        print("   🔄 Epochs: 50 (extended training)")
        print("   📊 Batch size per node: 32")
        print(f"   📊 Global batch size: {32 * num_nodes}")
        print("   🧠 Learning rate: 1e-6 (optimized for distributed)")
        print("   🎯 Full dataset (no sample limits)")
        print(
            f"   💾 Resources per node: {cpu_per_node} CPU, {memory_per_node}Gi memory"
        )
        print("   🔄 Data sharding: Automatic across nodes")
        print("   📊 Synchronized hard negative mining")

        # Initialize the Kubeflow Trainer client
        print("🔌 Initializing Kubeflow Trainer client...")
        client = TrainerClient()

        print("🚀 Submitting distributed training job to Kubernetes...")

        # Create distributed training job
        job_id = client.train(
            runtime=client.get_runtime("torch-distributed"),
            trainer=CustomTrainer(
                func=distributed_hybrid_embedding_training,
                func_args={
                    "model_name": "all-mpnet-base-v2",  # Larger model for production
                    "epochs": "50",  # Extended training
                    "batch_size": "32",  # Per-node batch size
                    "learning_rate": "1e-6",  # Lower LR for stability
                    "max_samples": None,  # Use full dataset
                    "feast_repo_path": "feature_repo",
                    "hard_negative_update_frequency": "5",  # Sync every 5 epochs
                    "output_dir": "fine_tuned_distributed_embeddings",
                    "tensorboard_dir": "tensorboard_logs_distributed",
                },
                num_nodes=num_nodes,  # Multi-node distributed training
                resources_per_node={
                    "cpu": str(cpu_per_node),
                    "memory": f"{memory_per_node}Gi",
                    # "nvidia.com/gpu": "1",  # Uncomment if GPUs available
                },
            ),
        )

        print_success(f"Distributed training job submitted with ID: {job_id}")
        print("📊 Monitor with: kubectl get trainjobs -w")
        print(
            "📜 View logs with: kubectl logs -f -l trainer.kubeflow.org/trainjob-ancestor-step=trainer"
        )
        print(
            f"🎯 Training {num_nodes} nodes with global batch size {32 * num_nodes}"
        )

        return job_id

    except Exception as e:
        print_error(f"Failed to deploy distributed training: {e}")
        import traceback

        print("🐛 Full error details:")
        traceback.print_exc()
        sys.exit(1)


def wait_for_distributed_job_start(job_id, timeout_minutes=10):
    """Wait for distributed training job to start"""
    print("⏳ Waiting for distributed job to start...")
    print("🔍 This may take longer for distributed training:")
    print("   📥 Pulling large PyTorch containers on multiple nodes")
    print("   🏗️  Creating training pods across cluster")
    print("   ⚙️  Setting up distributed communication")
    print("   🔄 Initializing data sharding")

    for i in range(timeout_minutes * 6):  # Check every 10 seconds
        # Check TrainJob status
        result = run_command(
            "kubectl get trainjobs -o jsonpath='{.items[*].status.conditions[-1].type}'",
            check=False,
        )

        if "Running" in result.stdout:
            print("✅ Distributed training job is running!")
            return True
        elif "Succeeded" in result.stdout:
            print("✅ Distributed training job completed!")
            return True

        # Show pod status for more detailed progress
        pod_result = run_command(
            "kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer --no-headers 2>/dev/null || echo 'No pods yet'",
            check=False,
        )

        if "No pods yet" not in pod_result.stdout and pod_result.stdout.strip():
            # Parse pod status
            lines = pod_result.stdout.strip().split("\n")
            statuses = [line.split()[2] for line in lines if line.strip()]
            status_summary = ", ".join(set(statuses))
            ready_pods = len([s for s in statuses if s == "Running"])
            total_pods = len(statuses)

            print(
                f"   📦 Distributed pods: {ready_pods}/{total_pods} running ({status_summary}) ({i + 1}/{timeout_minutes * 6})"
            )
        else:
            print(
                f"   ⌛ Still waiting for distributed pods to be created... ({i + 1}/{timeout_minutes * 6})"
            )

        time.sleep(10)

    print(
        "⚠️ Job may not have started yet. This is normal for distributed setups."
    )
    print("🔍 Check manually with: kubectl get trainjobs -w")
    return False


def monitor_distributed_training():
    """Show distributed training status and monitoring commands"""
    print_status("Monitoring distributed training status...")

    print("\n📊 Training Jobs:")
    run_command("kubectl get trainjobs -o wide", check=False)

    print("\n🔧 Training Pods (Distributed):")
    result = run_command(
        "kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print("   ⚠️  No training pods found yet (still starting up)")

    print("\n⚙️  All Related Pods (JobSet):")
    run_command("kubectl get pods -l jobset.sigs.k8s.io", check=False)

    print("\n📊 Node Status:")
    run_command(
        "kubectl top nodes 2>/dev/null || kubectl get nodes", check=False
    )

    print("\n📜 Recent Events:")
    event_result = run_command(
        "kubectl get events --sort-by=.metadata.creationTimestamp | tail -15",
        check=False,
    )
    if event_result.returncode != 0 or not event_result.stdout.strip():
        print("   ℹ️  No recent events found")

    print("\n💡 Distributed Training Monitoring:")
    print("   🔍 Watch jobs:         kubectl get trainjobs -w")
    print(
        "   📜 Stream all logs:    kubectl logs -f -l trainer.kubeflow.org/trainjob-ancestor-step=trainer --prefix"
    )
    print("   🔧 Job details:        kubectl describe trainjobs")
    print("   📦 Pod distribution:   kubectl get pods -o wide")
    print("   🌐 Node utilization:   kubectl top nodes")

    print("\n🎯 Distributed Training Features:")
    print("   1️⃣  Automatic data sharding across nodes")
    print("   2️⃣  Synchronized gradient updates")
    print("   3️⃣  Global hard negative mining")
    print("   4️⃣  Distributed TensorBoard logging")
    print("   5️⃣  Fault-tolerant training")

    print("\n💡 Expected Training Flow:")
    print("   📥 Each node loads its data shard")
    print("   🔄 Distributed training begins with gradient synchronization")
    print("   📊 TensorBoard shows aggregate metrics")
    print("   💾 Model saved from rank 0 node")


def main():
    """Main deployment function for distributed training"""
    print_banner()

    print("🚀 Starting distributed Kubeflow training deployment...")
    print("📋 This process will:")
    print("   1️⃣  Check prerequisites (cluster, registry, data)")
    print("   2️⃣  Prepare large-scale training data with sharding")
    print("   3️⃣  Build and push distributed training image")
    print("   4️⃣  Deploy multi-node distributed training")
    print("   5️⃣  Monitor distributed startup progress")
    print()

    # Configuration for distributed training
    NUM_NODES = 8  # Scale across 8 nodes
    CPU_PER_NODE = 3  # 3 CPUs per node
    MEMORY_PER_NODE = 6  # 6GB per node

    print("⚙️  Distributed Configuration:")
    print(f"   - Nodes: {NUM_NODES}")
    print(f"   - CPUs per node: {CPU_PER_NODE}")
    print(f"   - Memory per node: {MEMORY_PER_NODE}Gi")
    print(
        f"   - Total resources: {NUM_NODES * CPU_PER_NODE} CPUs, {NUM_NODES * MEMORY_PER_NODE}Gi memory"
    )
    print()

    # Check prerequisites
    print("=" * 60)
    print("🔍 STEP 1: CHECKING PREREQUISITES")
    print("=" * 60)
    check_prerequisites()

    # Build and push training image
    print("\n" + "=" * 60)
    print("🐳 STEP 2: BUILDING DISTRIBUTED TRAINING IMAGE")
    print("=" * 60)
    build_and_push_image()

    # Deploy distributed training job
    print("\n" + "=" * 60)
    print("☸️  STEP 3: DEPLOYING DISTRIBUTED TRAINING")
    print("=" * 60)
    job_id = deploy_distributed_training(
        NUM_NODES, CPU_PER_NODE, MEMORY_PER_NODE
    )

    # Wait for job to start
    print("\n" + "=" * 60)
    print("⏳ STEP 4: MONITORING DISTRIBUTED STARTUP")
    print("=" * 60)
    started = wait_for_distributed_job_start(job_id, timeout_minutes=15)

    # Show monitoring information
    print("\n" + "=" * 60)
    print("📊 STEP 5: DISTRIBUTED TRAINING STATUS")
    print("=" * 60)
    monitor_distributed_training()

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 DISTRIBUTED DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print_success(
        f"Distributed training job '{job_id}' deployed across {NUM_NODES} nodes!"
    )

    if started:
        print(
            "✅ Job is running - check logs for distributed training progress"
        )
    else:
        print(
            "⏳ Job is starting - distributed setups take longer to initialize"
        )

    print("\n🎯 What happens next:")
    print("   📥 Each node downloads PyTorch containers")
    print("   🔄 Distributed training begins with data sharding")
    print("   📊 Training metrics aggregated across all nodes")
    print("   💾 Fine-tuned model saved from master node")

    print("\n🔍 Monitor distributed training with:")
    print(
        "   kubectl logs -f -l trainer.kubeflow.org/trainjob-ancestor-step=trainer --prefix"
    )

    print("\n🎯 When training completes, find your model in:")
    print("   📁 ./fine_tuned_distributed_embeddings/")
    print("   📈 ./tensorboard_logs_distributed/")
    print(
        f"   🏆 Trained on {NUM_NODES} nodes with {NUM_NODES * 32} global batch size"
    )


if __name__ == "__main__":
    main()
