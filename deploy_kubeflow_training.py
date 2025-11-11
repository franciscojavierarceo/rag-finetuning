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
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

# Import training function
from kubeflow_embedding_training import hybrid_embedding_training
from kubeflow.trainer import TrainerClient, CustomTrainer
from kubernetes import client, config
from kubernetes.client.rest import ApiException


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


def create_custom_runtime():
    """Create custom ClusterTrainingRuntime with our Docker image"""
    print_status("Creating custom training runtime...")

    try:
        # Check if runtime already exists
        result = run_command_with_output(["kubectl", "get", "clustertrainingruntime", "rag-torch-distributed"])
        if result:
            print_success("Custom runtime already exists")
            return True
    except:
        pass

    # Define the custom runtime as JSON
    custom_runtime = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "ClusterTrainingRuntime",
        "metadata": {
            "name": "rag-torch-distributed",
            "labels": {
                "trainer.kubeflow.org/framework": "torch"
            }
        },
        "spec": {
            "mlPolicy": {
                "numNodes": 1,  # Will be overridden by CustomTrainer num_nodes
                "torch": {
                    "numProcPerNode": "auto"
                }
            },
            "template": {
                "metadata": {},
                "spec": {
                    "replicatedJobs": [
                        {
                            "name": "node",
                            "template": {
                                "metadata": {
                                    "labels": {
                                        "trainer.kubeflow.org/trainjob-ancestor-step": "trainer"
                                    }
                                },
                                "spec": {
                                    "template": {
                                        "metadata": {},
                                        "spec": {
                                            "containers": [
                                                {
                                                    "name": "node",
                                                    "image": "localhost:5001/rag-embedding-training:latest",
                                                    "env": [
                                                        {
                                                            "name": "PYTHONPATH",
                                                            "value": "/workspace:$PYTHONPATH"
                                                        },
                                                        {
                                                            "name": "PROJECT_DIR",
                                                            "value": "/workspace"
                                                        },
                                                        {
                                                            "name": "TRANSFORMERS_CACHE",
                                                            "value": "/home/trainer/.cache/huggingface"
                                                        },
                                                        {
                                                            "name": "HF_HOME",
                                                            "value": "/home/trainer/.cache/huggingface"
                                                        },
                                                        {
                                                            "name": "TORCH_HOME",
                                                            "value": "/home/trainer/.cache/torch"
                                                        },
                                                        {
                                                            "name": "XDG_CACHE_HOME",
                                                            "value": "/home/trainer/.cache"
                                                        }
                                                    ],
                                                    "volumeMounts": [
                                                        {
                                                            "name": "training-storage",
                                                            "mountPath": "/workspace/outputs"
                                                        }
                                                    ],
                                                    "resources": {
                                                        "requests": {
                                                            "cpu": "2",
                                                            "memory": "4Gi"
                                                        },
                                                        "limits": {
                                                            "cpu": "2",
                                                            "memory": "4Gi"
                                                        }
                                                    }
                                                }
                                            ],
                                            "volumes": [
                                                {
                                                    "name": "training-storage",
                                                    "persistentVolumeClaim": {
                                                        "claimName": "rag-training-storage"
                                                    }
                                                }
                                            ],
                                            "restartPolicy": "Never"
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }
    }

    # Apply the runtime using kubectl
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(custom_runtime, f, indent=2)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", temp_file],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print_success("Custom runtime created successfully")
            print(f"Output: {result.stdout}")
            return True
        else:
            print_error(f"Failed to create runtime. Exit code: {result.returncode}")
            print_error(f"Error: {result.stderr}")
            print_error(f"Output: {result.stdout}")
            return False
    except Exception as e:
        print_error(f"Failed to create runtime: {e}")
        return False
    finally:
        os.unlink(temp_file)


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
        print_warning("Run: uv run prepare_training_data.py first")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != "y":
            sys.exit(1)

    print_success("Prerequisites check passed!")


def build_and_push_image():
    """Build and push training image to local registry"""
    print_status("Building and pushing training image...")

    image_name = "localhost:5001/rag-embedding-training:latest"

    # Check if image already exists in registry
    result = subprocess.run(
        "curl -sf http://localhost:5001/v2/rag-embedding-training/tags/list",
        shell=True, capture_output=True, text=True
    )
    if result.returncode == 0 and "latest" in result.stdout:
        print_success("Image already exists in registry - skipping build")
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
        "curl -sf http://localhost:5001/v2/rag-embedding-training/tags/list",
        check=False,
    )
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
        print("   🔄 Epochs: 10 (fast testing)")
        print("   📊 Batch size: 8 per node (16 global)")
        print("   🧠 Learning rate: 2e-6")
        print("   🎯 Max samples: 500 (substantial dataset)")
        print("   🔧 Nodes: 2 (distributed)")
        print("   💾 Resources: 2 CPU, 4Gi memory per node")
        print("   🔄 Hard negatives: Update at epochs 5 & 10")
        print("   ⚡ Optimized: 1hr timeout, 25 hard negatives per update")
        print("   🐳 Runtime: rag-torch-distributed (custom image with training data)")

        # Create custom runtime first
        print("🏗️  Creating custom runtime...")
        if not create_custom_runtime():
            print_error("Failed to create custom runtime")
            return None

        # Initialize the Kubeflow Trainer client
        print("🔌 Initializing Kubeflow Trainer client...")
        client = TrainerClient()

        print("🚀 Submitting training job to Kubernetes...")
        # Create distributed training job
        job_id = client.train(
            runtime=client.get_runtime("rag-torch-distributed"),
            trainer=CustomTrainer(
                func=hybrid_embedding_training,
                func_args={
                    "model_name": "all-MiniLM-L6-v2",
                    "epochs": "10",  # Faster testing with distributed training
                    "batch_size": "8",  # Larger batch size with good memory
                    "learning_rate": "2e-6",
                    "max_samples": "500",  # Use more of the available dataset
                    "feast_repo_path": "feature_repo",
                    "hard_negative_update_frequency": "5",  # Update hard negatives at epoch 5 and 10
                },
                num_nodes=2,  # Use 2 nodes for distributed training
                resources_per_node={
                    "cpu": "2",  # Reduced CPU per node for better stability
                    "memory": "4Gi",  # Much smaller memory footprint for KIND nodes
                    # Note: GPU support can be added here if available
                },
            ),
        )

        print_success(f"Training job submitted with ID: {job_id}")
        print("📊 Monitor with: kubectl get trainjobs -w")
        print("📜 View logs with:")
        print("   1️⃣  kubectl get pods")
        print(f"   2️⃣  kubectl logs <pod-id> | tail -n 50")
        print(f"   📝 Look for pods containing: {job_id}")

        return job_id

    except Exception as e:
        print_error(f"Failed to deploy training: {e}")
        import traceback

        print("🐛 Full error details:")
        traceback.print_exc()
        sys.exit(1)


def wait_for_job_start(job_id, timeout_minutes=2):
    """Wait for training job to start"""
    print("⏳ Waiting for job to start...")
    print("🔍 This should be faster now with pre-cached model:")
    print("   📥 Using cached sentence-transformers model")
    print("   🏗️  Creating training pods")
    print("   ⚙️  Setting up distributed training")

    for i in range(timeout_minutes * 6):  # Check every 10 seconds, only 12 total checks
        # Check TrainJob status
        result = run_command(
            "kubectl get trainjobs -o jsonpath='{.items[*].status.conditions[-1].type}'",
            check=False,
        )

        if "Running" in result.stdout:
            print("✅ Training job is running!")
            return True
        elif "Succeeded" in result.stdout:
            print("✅ Training job completed!")
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
            print(
                f"   📦 Pods status: {status_summary} ({i + 1}/{timeout_minutes * 6})"
            )
        else:
            print(
                f"   ⌛ Still waiting for pods to be created... ({i + 1}/{timeout_minutes * 6})"
            )

        time.sleep(10)  # Check every 10 seconds

    print(
        "⚠️ Job may not have started yet. This is normal for large PyTorch images."
    )
    print("🔍 Check manually with: kubectl get trainjobs -w")
    return False


def monitor_training():
    """Show training status and monitoring commands"""
    print_status("Current training status...")

    print("\n📊 Training Jobs:")
    run_command("kubectl get trainjobs -o wide", check=False)

    print("\n🔧 Training Pods:")
    result = run_command(
        "kubectl get pods -l trainer.kubeflow.org/trainjob-ancestor-step=trainer",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print("   ⚠️  No training pods found yet (still starting up)")

    print("\n⚙️  All Related Pods (JobSet):")
    run_command("kubectl get pods -l jobset.sigs.k8s.io", check=False)

    print("\n📜 Recent Events:")
    event_result = run_command(
        "kubectl get events --sort-by=.metadata.creationTimestamp | tail -10",
        check=False,
    )
    if event_result.returncode != 0 or not event_result.stdout.strip():
        print("   ℹ️  No recent events found")

    print("\n💡 Next Steps - Monitor your training:")
    print("   🔍 Watch jobs:         kubectl get trainjobs -w")
    print("   📦 List pods:          kubectl get pods")
    print("   📜 View logs:          kubectl logs <pod-id> | tail -n 50")
    print("   🔧 Job details:        kubectl describe trainjobs")
    print("   📦 Check all pods:     kubectl get pods -A")
    print("   🌐 Cluster info:       kubectl get nodes")

    print("\n🎯 Training Progress:")
    print("   1️⃣  Pods should move from 'ContainerCreating' → 'Running'")
    print("   2️⃣  Training logs will show epoch progress")
    print("   3️⃣  Model will be saved to './fine_tuned_kubeflow_embeddings/'")
    print("   4️⃣  TensorBoard logs will be in './tensorboard_logs/'")

    print("\n💡 If pods are stuck 'ContainerCreating':")
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
    print("   1️⃣  List all pods: kubectl get pods")
    print("   2️⃣  Find training pods (look for names containing your job ID)")
    print("   3️⃣  View logs: kubectl logs <pod-id> | tail -n 50")
    print(f"   📝 Your job ID: {job_id}")
    print("   💡 Example: kubectl logs your-job-id-node-0-0-xxxxx | tail -n 50")

    print("\n🎯 When training completes, find your model in:")
    print("   📁 ./fine_tuned_kubeflow_embeddings/")
    print("   📈 ./tensorboard_logs/")


if __name__ == "__main__":
    main()
