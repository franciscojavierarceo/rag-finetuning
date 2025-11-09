#!/bin/bash

# Docker Compose RAG Training - Simple and Reliable
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
    echo "║              🐳 DOCKER COMPOSE TRAINING                  ║"
    echo "║                 Reliable Local Training                  ║"
    echo "║                                                           ║"
    echo "║               No Kubernetes Required!                    ║"
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

    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    print_success "Prerequisites check passed!"
}

# Build services
build_services() {
    print_status "Building training image..."

    docker compose build embedding-training

    print_success "Training image built successfully!"
}

# Start all services
start_services() {
    print_status "Starting all services..."

    # Start registry first
    docker compose up -d registry

    # Wait a moment for registry
    sleep 5

    # Start all other services
    docker compose up -d

    print_success "All services started!"
}

# Start training
start_training() {
    print_status "Starting training job..."

    docker compose up embedding-training

    print_success "Training completed!"
}

# Show logs
show_logs() {
    case "${1:-training}" in
        "training")
            print_status "Showing training logs..."
            docker compose logs -f embedding-training
            ;;
        "tensorboard")
            print_status "Showing TensorBoard logs..."
            docker compose logs -f tensorboard
            ;;
        "all")
            print_status "Showing all logs..."
            docker compose logs -f
            ;;
        *)
            print_error "Unknown log type: $1"
            echo "Usage: $0 logs [training|tensorboard|all]"
            exit 1
            ;;
    esac
}

# Show status
show_status() {
    print_status "Checking service status..."

    echo ""
    echo "🐳 Docker Compose Services:"
    docker compose ps

    echo ""
    echo "📊 Service Health:"
    if curl -sf http://localhost:5001/v2/ >/dev/null 2>&1; then
        echo "   ✅ Registry: http://localhost:5001"
    else
        echo "   ❌ Registry: Not accessible"
    fi

    if curl -sf http://localhost:6006/ >/dev/null 2>&1; then
        echo "   ✅ TensorBoard: http://localhost:6006"
    else
        echo "   ❌ TensorBoard: Not accessible"
    fi

    echo ""
    echo "📁 Data Volumes:"
    docker volume ls | grep rag-finetuning
}

# Open TensorBoard
open_tensorboard() {
    print_status "Opening TensorBoard..."

    if command -v open >/dev/null 2>&1; then
        open http://localhost:6006
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open http://localhost:6006
    else
        print_success "TensorBoard available at: http://localhost:6006"
    fi
}

# Access debug container
debug_shell() {
    print_status "Starting debug shell..."

    if docker compose ps debug | grep -q "Up"; then
        docker compose exec debug sh
    else
        print_warning "Debug container not running, starting it..."
        docker compose up -d debug
        sleep 2
        docker compose exec debug sh
    fi
}

# Development workflow
dev_rebuild() {
    print_status "Development rebuild and restart..."

    # Stop training container
    docker compose stop embedding-training

    # Rebuild image
    docker compose build embedding-training

    # Start training
    docker compose up embedding-training

    print_success "Development rebuild completed!"
}

# Cleanup
cleanup() {
    print_status "Cleaning up all containers and volumes..."

    docker compose down -v --remove-orphans

    # Remove any dangling images
    docker image prune -f

    print_success "Cleanup completed!"
}

# Show help
show_help() {
    echo "Docker Compose RAG Training Commands:"
    echo ""
    echo "Setup and Training:"
    echo "  $0 setup       - Build and start all services"
    echo "  $0 train       - Run training job"
    echo "  $0 build       - Build training image only"
    echo "  $0 start       - Start all services (background)"
    echo ""
    echo "Monitoring:"
    echo "  $0 logs        - Show training logs"
    echo "  $0 status      - Show service status"
    echo "  $0 tensorboard - Open TensorBoard in browser"
    echo ""
    echo "Development:"
    echo "  $0 rebuild     - Rebuild and restart training"
    echo "  $0 debug       - Access debug shell"
    echo "  $0 cleanup     - Stop all services and cleanup"
    echo ""
    echo "Examples:"
    echo "  $0 setup && $0 train    # Complete training run"
    echo "  $0 logs training        # Stream training logs"
    echo "  $0 rebuild              # Quick development cycle"
}

# Main function
main() {
    case "${1:-help}" in
        "setup")
            print_banner
            check_prerequisites
            build_services
            start_services
            ;;
        "train")
            start_training
            ;;
        "build")
            check_prerequisites
            build_services
            ;;
        "start")
            start_services
            ;;
        "logs")
            show_logs "${2:-training}"
            ;;
        "status")
            show_status
            ;;
        "tensorboard")
            open_tensorboard
            ;;
        "debug")
            debug_shell
            ;;
        "rebuild"|"dev")
            dev_rebuild
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            exit 0
            ;;
    esac
}

# Run main function
main "$@"