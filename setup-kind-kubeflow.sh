#!/bin/bash

# KIND Cluster Setup with Kubeflow Trainer Operator
# This script creates a KIND cluster and installs Kubeflow Trainer for distributed training

set -e

# Configuration
KIND_CLUSTER_NAME="rag-kubeflow"
LOCAL_REGISTRY_NAME="kind-registry"
LOCAL_REGISTRY_PORT="5001"
KUBEFLOW_VERSION="v2.1.0"

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
    echo "║             🚀 KIND + KUBEFLOW TRAINER                   ║"
    echo "║                 Distributed RAG Training                 ║"
    echo "║                                                           ║"
    echo "║          Production-ready Kubernetes Training            ║"
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

    # Check Docker
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    # Check KIND
    if ! command -v kind >/dev/null 2>&1; then
        print_error "KIND is not installed. Installing KIND..."
        # Install KIND on macOS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew >/dev/null 2>&1; then
                brew install kind
            else
                # Install via direct download
                [ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-darwin-amd64
                [ $(uname -m) = arm64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-darwin-arm64
                chmod +x ./kind
                sudo mv ./kind /usr/local/bin/kind
            fi
        else
            print_error "Please install KIND manually: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
            exit 1
        fi
    fi

    # Check kubectl
    if ! command -v kubectl >/dev/null 2>&1; then
        print_error "kubectl is not installed. Installing kubectl..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew >/dev/null 2>&1; then
                brew install kubectl
            else
                curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
                chmod +x kubectl
                sudo mv kubectl /usr/local/bin/
            fi
        else
            print_error "Please install kubectl manually"
            exit 1
        fi
    fi

    print_success "Prerequisites check passed!"
}

# Clean up existing setup
cleanup_existing() {
    print_status "Cleaning up existing setup..."

    # Stop and remove existing KIND cluster
    if kind get clusters | grep -q "^${KIND_CLUSTER_NAME}$"; then
        print_warning "Removing existing cluster: ${KIND_CLUSTER_NAME}"
        kind delete cluster --name "${KIND_CLUSTER_NAME}" || true
    fi

    # Stop and remove local registry
    if docker ps -a | grep -q "${LOCAL_REGISTRY_NAME}"; then
        print_warning "Removing existing registry: ${LOCAL_REGISTRY_NAME}"
        docker rm -f "${LOCAL_REGISTRY_NAME}" || true
    fi

    print_success "Cleanup completed!"
}

# Create local registry
create_local_registry() {
    print_status "Setting up local Docker registry..."

    # Start local registry
    docker run -d \
        --restart=always \
        --name "${LOCAL_REGISTRY_NAME}" \
        -p "${LOCAL_REGISTRY_PORT}:5000" \
        registry:2

    # Wait for registry to be ready
    sleep 5

    # Test registry
    if curl -sf http://localhost:${LOCAL_REGISTRY_PORT}/v2/ >/dev/null; then
        print_success "Local registry started on port ${LOCAL_REGISTRY_PORT}"
    else
        print_error "Failed to start local registry"
        exit 1
    fi
}

# Create optimized KIND cluster configuration
create_kind_config() {
    print_status "Creating optimized KIND cluster configuration..."

    cat <<EOF > kind-config-kubeflow.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ${KIND_CLUSTER_NAME}
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors."localhost:${LOCAL_REGISTRY_PORT}"]
    endpoint = ["http://${LOCAL_REGISTRY_NAME}:5000"]
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
  - containerPort: 6006
    hostPort: 6006
    protocol: TCP
- role: worker
- role: worker
EOF

    print_success "KIND configuration created for distributed training!"
}

# Create KIND cluster with enhanced error handling
create_kind_cluster() {
    print_status "Creating KIND cluster: ${KIND_CLUSTER_NAME}..."

    # Create cluster with longer timeout and better error handling
    local max_attempts=3
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        print_status "Attempt ${attempt}/${max_attempts} to create cluster..."

        if timeout 600 kind create cluster --config=kind-config-kubeflow.yaml --wait=10m; then
            print_success "KIND cluster created successfully!"
            break
        else
            print_warning "Cluster creation failed on attempt ${attempt}"
            if [[ $attempt -eq $max_attempts ]]; then
                print_error "Failed to create cluster after ${max_attempts} attempts"
                print_error "Try increasing Docker resources: Docker Desktop -> Settings -> Resources"
                exit 1
            fi

            # Clean up failed attempt
            kind delete cluster --name "${KIND_CLUSTER_NAME}" || true
            sleep 10
            ((attempt++))
        fi
    done

    # Connect registry to cluster network
    if docker network ls | grep -q "kind"; then
        docker network connect "kind" "${LOCAL_REGISTRY_NAME}" 2>/dev/null || true
    fi

    # Document the local registry for Kubernetes
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${LOCAL_REGISTRY_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

    print_success "KIND cluster is ready!"
}

# Configure registry accessibility for KIND nodes
configure_registry_access() {
    print_status "Configuring registry access for KIND nodes..."

    # Get the registry IP in the KIND network
    local REGISTRY_IP=$(docker inspect "${LOCAL_REGISTRY_NAME}" | grep '"IPAddress"' | grep -o '[0-9.]*' | head -1)

    if [ -z "$REGISTRY_IP" ]; then
        print_warning "Could not determine registry IP, attempting fallback configuration..."
        REGISTRY_IP="172.18.0.5"  # Common KIND network IP for registry
    fi

    print_status "Configuring nodes to access registry at ${REGISTRY_IP}:5000..."

    # Configure registry access on all KIND nodes
    for node in ${KIND_CLUSTER_NAME}-control-plane ${KIND_CLUSTER_NAME}-worker ${KIND_CLUSTER_NAME}-worker2; do
        print_status "Configuring registry access on $node..."
        docker exec "$node" sh -c "
            mkdir -p /etc/containerd/certs.d/localhost:${LOCAL_REGISTRY_PORT}
            cat > /etc/containerd/certs.d/localhost:${LOCAL_REGISTRY_PORT}/hosts.toml << EOF
server = \"http://${REGISTRY_IP}:5000\"

[host.\"http://${REGISTRY_IP}:5000\"]
  capabilities = [\"pull\", \"resolve\", \"push\"]
  skip_verify = true
EOF
        " || print_warning "Failed to configure registry on $node"
    done

    # Restart containerd on all nodes to pick up the new configuration
    print_status "Restarting containerd on all nodes to apply registry configuration..."
    for node in ${KIND_CLUSTER_NAME}-control-plane ${KIND_CLUSTER_NAME}-worker ${KIND_CLUSTER_NAME}-worker2; do
        docker exec "$node" systemctl restart containerd || print_warning "Failed to restart containerd on $node"
    done

    # Wait for nodes to be ready after containerd restart
    print_status "Waiting for nodes to be ready after containerd restart..."
    sleep 10
    kubectl wait --for=condition=Ready nodes --all --timeout=300s

    # Test registry access from a node
    if docker exec "${KIND_CLUSTER_NAME}-worker" curl -sf "http://${REGISTRY_IP}:5000/v2/" >/dev/null; then
        print_success "Registry access configured successfully! Nodes can access registry at ${REGISTRY_IP}:5000"
    else
        print_warning "Registry access verification failed, but configuration was applied"
    fi
}

# Install Kubeflow Trainer Operator
install_kubeflow_trainer() {
    print_status "Installing Kubeflow Trainer Operator..."

    # Wait for cluster to be fully ready
    print_status "Waiting for cluster nodes to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=300s

    # Install Kubeflow Trainer Controller Manager
    print_status "Installing Kubeflow Trainer Controller Manager..."
    kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${KUBEFLOW_VERSION}"

    # Wait for controller manager to be ready
    print_status "Waiting for controller manager to be ready..."
    kubectl wait --for=condition=Available deployment/kubeflow-trainer-controller-manager -n kubeflow-system --timeout=300s

    # Install Kubeflow Training Runtimes
    print_status "Installing Kubeflow Training Runtimes..."
    kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${KUBEFLOW_VERSION}"

    # Wait for all components to be ready
    print_status "Waiting for all Kubeflow components to be ready..."
    kubectl wait --for=condition=Available deployment/jobset-controller-manager -n kubeflow-system --timeout=300s

    print_success "Kubeflow Trainer Operator installed successfully!"
}

# Verify installation
verify_installation() {
    print_status "Verifying Kubeflow Trainer installation..."

    echo ""
    echo "📋 Cluster Information:"
    kubectl cluster-info

    echo ""
    echo "📦 Nodes:"
    kubectl get nodes -o wide

    echo ""
    echo "🔧 Kubeflow System Pods:"
    kubectl get pods -n kubeflow-system

    echo ""
    echo "📊 Storage Classes:"
    kubectl get storageclass

    echo ""
    echo "🏪 Registry Test:"
    if curl -sf http://localhost:${LOCAL_REGISTRY_PORT}/v2/ >/dev/null; then
        echo "   ✅ Local registry is accessible at localhost:${LOCAL_REGISTRY_PORT}"
    else
        echo "   ❌ Local registry is not accessible"
    fi

    echo ""
    echo "🚀 Kubeflow Trainer Verification:"
    if kubectl get crd | grep -q "trainjobs.trainer.kubeflow.org"; then
        echo "   ✅ TrainJob CRD is installed"
    else
        echo "   ❌ TrainJob CRD not found"
    fi

    echo "   ✅ Available Training Runtimes:"
    kubectl get clustertrainingruntimes --no-headers | awk '{print "      - " $1}'

    # Note: TrainJobs are created via Kubeflow Python SDK, not direct YAML
    echo "   ✅ Use Kubeflow Python SDK to create TrainJob resources"

    print_success "Installation verification completed!"
}

# Show next steps
show_next_steps() {
    echo ""
    echo "🎉 Kubeflow Trainer on KIND is ready!"
    echo ""
    echo "📝 What's installed:"
    echo "   ✅ KIND cluster with 3 nodes (1 control-plane, 2 workers)"
    echo "   ✅ Kubeflow Trainer Operator ${KUBEFLOW_VERSION}"
    echo "   ✅ JobSet Controller for distributed training"
    echo "   ✅ Local Docker registry at localhost:${LOCAL_REGISTRY_PORT}"
    echo "   ✅ TrainingJob CRD for Kubeflow training workloads"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Build and push your training image:"
    echo "      docker build -t localhost:${LOCAL_REGISTRY_PORT}/rag-embedding-training:latest ."
    echo "      docker push localhost:${LOCAL_REGISTRY_PORT}/rag-embedding-training:latest"
    echo ""
    echo "   2. Deploy your RAG training job:"
    echo "      ./deploy-kubeflow-training.sh"
    echo ""
    echo "   3. Monitor training:"
    echo "      kubectl logs -f job/rag-embedding-training"
    echo "      kubectl get trainingjobs -w"
    echo ""
    echo "📊 Useful commands:"
    echo "   kubectl config current-context    # Verify context"
    echo "   kubectl get nodes                 # Check cluster health"
    echo "   kubectl get pods -A               # See all pods"
    echo "   kubectl get trainingjobs -A       # Monitor training jobs"
    echo ""
    echo "🧹 Cleanup when done:"
    echo "   kind delete cluster --name ${KIND_CLUSTER_NAME}"
    echo ""
}

# Main execution function
main() {
    print_banner

    case "${1:-all}" in
        "prereq")
            check_prerequisites
            ;;
        "cluster")
            cleanup_existing
            create_local_registry
            create_kind_config
            create_kind_cluster
            configure_registry_access
            ;;
        "kubeflow")
            install_kubeflow_trainer
            verify_installation
            ;;
        "test")
            verify_installation
            ;;
        "all")
            check_prerequisites
            cleanup_existing
            create_local_registry
            create_kind_config
            create_kind_cluster
            configure_registry_access
            install_kubeflow_trainer
            verify_installation
            show_next_steps
            ;;
        "help"|*)
            echo "Usage: $0 {prereq|cluster|kubeflow|test|all}"
            echo ""
            echo "Commands:"
            echo "  prereq    - Check and install prerequisites"
            echo "  cluster   - Create KIND cluster with registry"
            echo "  kubeflow  - Install Kubeflow Trainer operator"
            echo "  test      - Verify installation"
            echo "  all       - Complete setup (default)"
            echo ""
            exit 0
            ;;
    esac

    # Cleanup temp files
    rm -f kind-config-kubeflow.yaml
}

# Run main function
main "$@"