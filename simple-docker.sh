#!/bin/bash

# Simple Docker Training - No Kubernetes Required!
set -e

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
    echo "║                🐳 SIMPLE DOCKER TRAINING                 ║"
    echo "║                  No Kubernetes Needed!                   ║"
    echo "║                                                           ║"
    echo "║              Just Docker + Your Training                 ║"
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

# Build training image
build_image() {
    print_status "Building training image..."

    docker build -t rag-embedding-training:latest .

    print_success "Training image built successfully!"
}

# Start local registry
start_registry() {
    print_status "Starting local registry..."

    # Stop existing registry if running
    docker rm -f rag-registry 2>/dev/null || true

    # Start new registry
    docker run -d \
        --name rag-registry \
        --restart=always \
        -p 5001:5000 \
        registry:2

    # Wait for registry to be ready
    sleep 5

    if curl -sf http://localhost:5001/v2/ >/dev/null 2>&1; then
        print_success "Registry running at http://localhost:5001"
    else
        print_error "Failed to start registry"
        exit 1
    fi
}

# Start TensorBoard
start_tensorboard() {
    print_status "Starting TensorBoard..."

    # Stop existing TensorBoard if running
    docker rm -f rag-tensorboard 2>/dev/null || true

    # Create tensorboard logs directory
    mkdir -p "$(pwd)/tensorboard_logs"

    # Start TensorBoard
    docker run -d \
        --name rag-tensorboard \
        -p 6006:6006 \
        -v "$(pwd)/tensorboard_logs:/logs" \
        tensorflow/tensorflow:latest \
        tensorboard --logdir=/logs --host=0.0.0.0 --port=6006

    sleep 5

    if curl -sf http://localhost:6006/ >/dev/null 2>&1; then
        print_success "TensorBoard running at http://localhost:6006"
    else
        print_warning "TensorBoard may still be starting up..."
    fi
}

# Run training
run_training() {
    print_status "Running training in Docker..."

    # Create output directories
    mkdir -p "$(pwd)/fine_tuned_kubeflow_embeddings"
    mkdir -p "$(pwd)/tensorboard_logs"

    # Stop existing training container if running
    docker rm -f rag-training 2>/dev/null || true

    # Run training
    docker run -it \
        --name rag-training \
        -v "$(pwd):/workspace" \
        -w /workspace \
        -e PROJECT_DIR=/workspace \
        -e PYTHONPATH=/workspace \
        -e PYTHONUNBUFFERED=1 \
        -e DEVELOPMENT_MODE=true \
        --network host \
        rag-embedding-training:latest \
        python kubeflow_embedding_training.py

    print_success "Training completed!"
}

# Show training logs
show_logs() {
    print_status "Showing training logs..."

    if docker ps -a --format "table {{.Names}}" | grep -q "rag-training"; then
        docker logs -f rag-training
    else
        print_warning "Training container not found"
        echo "Available containers:"
        docker ps -a --filter name=rag- --format "table {{.Names}}\t{{.Status}}"
    fi
}

# Show status
show_status() {
    print_status "Checking Docker services status..."

    echo ""
    echo "🐳 Running Containers:"
    docker ps --filter name=rag- --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    echo ""
    echo "📊 Service Health:"
    if curl -sf http://localhost:5001/v2/ >/dev/null 2>&1; then
        echo "   ✅ Registry: http://localhost:5001"
    else
        echo "   ❌ Registry: Not running"
    fi

    if curl -sf http://localhost:6006/ >/dev/null 2>&1; then
        echo "   ✅ TensorBoard: http://localhost:6006"
    else
        echo "   ❌ TensorBoard: Not running"
    fi

    echo ""
    echo "📁 Local Directories:"
    echo "   Training Data: $(pwd)/feature_repo/data/"
    echo "   Model Output:  $(pwd)/fine_tuned_kubeflow_embeddings/"
    echo "   TensorBoard:   $(pwd)/tensorboard_logs/"
}

# Open TensorBoard
open_tensorboard() {
    print_status "Opening TensorBoard..."

    if curl -sf http://localhost:6006/ >/dev/null 2>&1; then
        if command -v open >/dev/null 2>&1; then
            open http://localhost:6006
        else
            print_success "TensorBoard available at: http://localhost:6006"
        fi
    else
        print_error "TensorBoard is not running. Start it with: $0 tensorboard"
    fi
}

# Stop all services
stop_services() {
    print_status "Stopping all services..."

    docker rm -f rag-training rag-tensorboard rag-registry 2>/dev/null || true

    print_success "All services stopped!"
}

# Quick development cycle
dev_cycle() {
    print_status "Development cycle: rebuild and run..."

    build_image
    run_training

    print_success "Development cycle completed!"
}

# Show help
show_help() {
    echo "Simple Docker Training Commands:"
    echo ""
    echo "Setup:"
    echo "  $0 build       - Build training image"
    echo "  $0 registry    - Start local registry"
    echo "  $0 tensorboard - Start TensorBoard"
    echo ""
    echo "Training:"
    echo "  $0 train       - Run training"
    echo "  $0 logs        - Show training logs"
    echo "  $0 status      - Show service status"
    echo ""
    echo "Development:"
    echo "  $0 dev         - Quick rebuild and train"
    echo "  $0 stop        - Stop all services"
    echo "  $0 tb          - Open TensorBoard"
    echo ""
    echo "Complete workflow:"
    echo "  $0 setup       - Build + start registry + TensorBoard"
    echo "  $0 all         - Complete setup + training"
}

# Main function
main() {
    case "${1:-help}" in
        "build")
            build_image
            ;;
        "registry")
            start_registry
            ;;
        "tensorboard")
            start_tensorboard
            ;;
        "train")
            run_training
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "tb")
            open_tensorboard
            ;;
        "dev")
            dev_cycle
            ;;
        "stop")
            stop_services
            ;;
        "setup")
            print_banner
            build_image
            start_registry
            start_tensorboard
            show_status
            ;;
        "all")
            print_banner
            build_image
            start_registry
            start_tensorboard
            run_training
            ;;
        "help"|*)
            show_help
            exit 0
            ;;
    esac
}

# Run main function
main "$@"