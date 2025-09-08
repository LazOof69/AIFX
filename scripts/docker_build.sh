#!/bin/bash
# AIFX Docker Build and Deployment Script | AIFX Docker構建和部署腳本
# Comprehensive Docker containerization for production deployment
# 生產部署的綜合Docker容器化

set -e

# Color codes for output | 輸出顏色代碼
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration | 配置
PROJECT_NAME="aifx"
VERSION="${AIFX_VERSION:-4.0.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "dev")
REGISTRY="${DOCKER_REGISTRY:-aifx}"

# Logging functions | 日誌函數
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX Build:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX Build SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX Build WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX Build ERROR:${NC} $1" >&2
}

# Function to show usage | 顯示使用方法的函數
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker images"
    echo "  test      Run tests in containers"
    echo "  deploy    Deploy to production"
    echo "  clean     Clean up Docker resources"
    echo "  health    Check system health"
    echo "  logs      Show container logs"
    echo ""
    echo "Options:"
    echo "  --dev     Use development configuration"
    echo "  --prod    Use production configuration"
    echo "  --push    Push images to registry after build"
    echo "  --no-cache Build without using cache"
    echo "  --verbose Enable verbose output"
    echo ""
    echo "Environment Variables:"
    echo "  AIFX_VERSION        Version tag (default: 4.0.0)"
    echo "  DOCKER_REGISTRY     Docker registry (default: aifx)"
    echo "  AIFX_ENV           Environment (development/production)"
}

# Function to check prerequisites | 檢查前置條件的函數
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check Docker | 檢查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose | 檢查Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running | 檢查Docker守護程序是否運行
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if we're in the right directory | 檢查是否在正確目錄
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found. Please run this script from the project root."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to build Docker images | 構建Docker映像的函數
build_images() {
    local environment="${1:-development}"
    local use_cache="${2:-true}"
    local push_images="${3:-false}"
    
    log "🏗️ Building AIFX Docker images for $environment environment..."
    
    # Set build arguments | 設置構建參數
    local cache_flag=""
    if [[ "$use_cache" == "false" ]]; then
        cache_flag="--no-cache"
    fi
    
    local build_args=(
        --build-arg "AIFX_ENV=$environment"
        --build-arg "BUILD_DATE=$BUILD_DATE"
        --build-arg "VCS_REF=$VCS_REF"
        --build-arg "VERSION=$VERSION"
    )
    
    # Build main application image | 構建主應用程式映像
    log "📦 Building main application image..."
    docker build \
        ${cache_flag} \
        "${build_args[@]}" \
        --target production \
        -t "${REGISTRY}/${PROJECT_NAME}:${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}:latest" \
        .
    
    if [[ $? -eq 0 ]]; then
        log_success "Main application image built successfully"
    else
        log_error "Failed to build main application image"
        exit 1
    fi
    
    # Build development image if needed | 如需要則構建開發映像
    if [[ "$environment" == "development" ]]; then
        log "📦 Building development image..."
        docker build \
            ${cache_flag} \
            "${build_args[@]}" \
            --target development \
            -t "${REGISTRY}/${PROJECT_NAME}:${VERSION}-dev" \
            -t "${REGISTRY}/${PROJECT_NAME}:dev" \
            .
        
        if [[ $? -eq 0 ]]; then
            log_success "Development image built successfully"
        else
            log_error "Failed to build development image"
            exit 1
        fi
    fi
    
    # Build testing image | 構建測試映像
    log "📦 Building testing image..."
    docker build \
        ${cache_flag} \
        "${build_args[@]}" \
        --target testing \
        -t "${REGISTRY}/${PROJECT_NAME}:${VERSION}-test" \
        -t "${REGISTRY}/${PROJECT_NAME}:test" \
        .
    
    if [[ $? -eq 0 ]]; then
        log_success "Testing image built successfully"
    else
        log_error "Failed to build testing image"
        exit 1
    fi
    
    # Show built images | 顯示已構建的映像
    log "📋 Built images:"
    docker images --filter "reference=${REGISTRY}/${PROJECT_NAME}*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # Push images if requested | 如有要求則推送映像
    if [[ "$push_images" == "true" ]]; then
        push_to_registry "$environment"
    fi
}

# Function to push images to registry | 推送映像到註冊表的函數
push_to_registry() {
    local environment="${1:-production}"
    
    log "📤 Pushing images to registry..."
    
    # Push main images | 推送主映像
    docker push "${REGISTRY}/${PROJECT_NAME}:${VERSION}"
    docker push "${REGISTRY}/${PROJECT_NAME}:latest"
    
    # Push environment specific images | 推送環境特定映像
    if [[ "$environment" == "development" ]]; then
        docker push "${REGISTRY}/${PROJECT_NAME}:${VERSION}-dev"
        docker push "${REGISTRY}/${PROJECT_NAME}:dev"
    fi
    
    docker push "${REGISTRY}/${PROJECT_NAME}:${VERSION}-test"
    docker push "${REGISTRY}/${PROJECT_NAME}:test"
    
    log_success "Images pushed to registry successfully"
}

# Function to run tests | 運行測試的函數
run_tests() {
    log "🧪 Running AIFX tests in containers..."
    
    # Run unit tests | 運行單元測試
    log "▶️ Running unit tests..."
    docker run --rm \
        -v "$(pwd)/src/test:/app/src/test" \
        -v "$(pwd)/test_reports:/app/test_reports" \
        "${REGISTRY}/${PROJECT_NAME}:test" \
        python -m pytest src/test/unit/ -v --tb=short --junitxml=/app/test_reports/unit_tests.xml
    
    # Run integration tests | 運行整合測試
    log "▶️ Running integration tests..."
    docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build --abort-on-container-exit aifx-integration-test
    
    # Run Phase 3 core tests | 運行第三階段核心測試
    log "▶️ Running Phase 3 core tests..."
    docker run --rm \
        -v "$(pwd):/app" \
        "${REGISTRY}/${PROJECT_NAME}:test" \
        python test_phase3_core.py
    
    log_success "All tests completed"
}

# Function to deploy to environment | 部署到環境的函數
deploy_to_environment() {
    local environment="${1:-production}"
    
    log "🚀 Deploying AIFX to $environment environment..."
    
    # Choose appropriate compose file | 選擇適當的compose文件
    local compose_file="docker-compose.yml"
    if [[ "$environment" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    # Check if environment file exists | 檢查環境文件是否存在
    local env_file=".env.${environment}"
    if [[ ! -f "$env_file" ]]; then
        log_warning "Environment file $env_file not found, using defaults"
    fi
    
    # Create necessary directories | 創建必要目錄
    log "📁 Creating necessary directories..."
    mkdir -p {data,models,output,logs}
    
    # Pull latest images | 拉取最新映像
    log "📥 Pulling latest images..."
    docker-compose -f "$compose_file" pull
    
    # Deploy with Docker Compose | 使用Docker Compose部署
    log "🔧 Starting services with Docker Compose..."
    if [[ "$environment" == "production" ]]; then
        docker-compose -f "$compose_file" up -d --remove-orphans
    else
        docker-compose -f "$compose_file" up -d --build --remove-orphans
    fi
    
    # Wait for services to be healthy | 等待服務健康
    log "⏳ Waiting for services to become healthy..."
    sleep 30
    
    # Check service health | 檢查服務健康
    check_service_health
    
    log_success "Deployment completed successfully"
}

# Function to check service health | 檢查服務健康的函數
check_service_health() {
    log "💊 Checking service health..."
    
    # Get list of running containers | 獲取運行中的容器列表
    local containers=$(docker-compose ps -q)
    
    for container in $containers; do
        local container_name=$(docker inspect --format='{{.Name}}' "$container" | sed 's/\///')
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no-healthcheck")
        
        if [[ "$health_status" == "healthy" ]]; then
            log_success "✅ $container_name is healthy"
        elif [[ "$health_status" == "no-healthcheck" ]]; then
            log_warning "⚪ $container_name has no health check"
        else
            log_warning "⚠️ $container_name health status: $health_status"
        fi
    done
}

# Function to show container logs | 顯示容器日誌的函數
show_logs() {
    local service="${1:-aifx-app}"
    local lines="${2:-100}"
    
    log "📋 Showing logs for service: $service"
    docker-compose logs --tail="$lines" -f "$service"
}

# Function to clean up Docker resources | 清理Docker資源的函數
cleanup_docker() {
    log "🧹 Cleaning up Docker resources..."
    
    # Stop and remove containers | 停止並移除容器
    log "⏹️ Stopping containers..."
    docker-compose down --remove-orphans
    
    # Remove unused images | 移除未使用的映像
    log "🗑️ Removing unused images..."
    docker image prune -f
    
    # Remove unused volumes (with confirmation) | 移除未使用的卷（需確認）
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
        log_success "Unused volumes removed"
    fi
    
    # Remove unused networks | 移除未使用的網絡
    docker network prune -f
    
    log_success "Docker cleanup completed"
}

# Function to run health monitoring | 運行健康監控的函數
run_health_monitor() {
    log "🏥 Starting AIFX health monitoring..."
    
    # Check if health monitor is available | 檢查健康監控器是否可用
    if [[ -f "docker/monitoring/health_monitor.py" ]]; then
        python3 docker/monitoring/health_monitor.py
    else
        log_error "Health monitor script not found"
        exit 1
    fi
}

# Main execution | 主執行
main() {
    local command="${1:-help}"
    local environment="development"
    local use_cache="true"
    local push_images="false"
    local verbose="false"
    
    # Parse command line arguments | 解析命令行參數
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                environment="development"
                shift
                ;;
            --prod)
                environment="production"
                shift
                ;;
            --push)
                push_images="true"
                shift
                ;;
            --no-cache)
                use_cache="false"
                shift
                ;;
            --verbose)
                verbose="true"
                set -x
                shift
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Set environment variables | 設置環境變數
    export AIFX_ENV="$environment"
    
    log "🚀 AIFX Docker Build and Deployment Tool"
    log "Environment: $environment | Version: $VERSION | Build Date: $BUILD_DATE"
    
    # Execute command | 執行命令
    case $command in
        "build")
            check_prerequisites
            build_images "$environment" "$use_cache" "$push_images"
            ;;
        "test")
            check_prerequisites
            run_tests
            ;;
        "deploy")
            check_prerequisites
            deploy_to_environment "$environment"
            ;;
        "logs")
            show_logs "$2" "$3"
            ;;
        "health")
            run_health_monitor
            ;;
        "clean")
            cleanup_docker
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments | 執行主函數並傳遞所有參數
main "$@"