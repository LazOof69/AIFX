#!/bin/bash
# AIFX Cloud Deployment Script | AIFX 雲端部署腳本
# Comprehensive deployment automation for AWS cloud infrastructure
# 針對 AWS 雲端基礎設施的全面部署自動化

set -e

# Color codes for output | 輸出顏色代碼
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration | 配置
PROJECT_NAME="aifx"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_REGION="us-west-2"
DEFAULT_ENVIRONMENT="staging"

# Logging functions | 日誌函數
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX Cloud Deploy:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

log_header() {
    echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] ==>${NC} $1"
}

# Function to show usage | 顯示使用方法的函數
show_usage() {
    echo "AIFX Cloud Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy           Deploy complete infrastructure and application"
    echo "  infrastructure   Deploy only infrastructure (Terraform)"
    echo "  application      Deploy only application (Kubernetes)"
    echo "  build            Build and push Docker images"
    echo "  test             Run deployment tests"
    echo "  status           Check deployment status"
    echo "  rollback         Rollback to previous version"
    echo "  destroy          Destroy infrastructure (use with caution)"
    echo "  logs             Show application logs"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV    Target environment (development/staging/production)"
    echo "  -r, --region REGION      AWS region (default: us-west-2)"
    echo "  -t, --image-tag TAG      Docker image tag (default: auto-generated)"
    echo "  -f, --force              Force deployment (skip confirmations)"
    echo "  --dry-run                Show what would be deployed without executing"
    echo "  --skip-tests             Skip pre-deployment tests"
    echo "  --skip-build             Skip Docker image build"
    echo "  --verbose                Enable verbose output"
    echo "  --help                   Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_ACCESS_KEY_ID        AWS access key"
    echo "  AWS_SECRET_ACCESS_KEY    AWS secret access key"
    echo "  AWS_ACCOUNT_ID           AWS account ID"
    echo "  GITHUB_TOKEN             GitHub token for image registry"
    echo ""
    echo "Examples:"
    echo "  $0 deploy -e staging                    # Deploy to staging"
    echo "  $0 infrastructure -e production -f     # Force deploy infrastructure to production"
    echo "  $0 build -t v1.2.3                     # Build image with specific tag"
    echo "  $0 status -e production                 # Check production status"
}

# Function to check prerequisites | 檢查前置條件的函數
check_prerequisites() {
    log_header "Checking prerequisites..."
    
    # Check required tools | 檢查必需工具
    local tools=("aws" "kubectl" "terraform" "docker" "helm")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    log_success "All required tools are available"
    
    # Check AWS credentials | 檢查AWS憑證
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials are not configured or invalid"
        exit 1
    fi
    
    local aws_account_id=$(aws sts get-caller-identity --query Account --output text)
    export AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$aws_account_id}"
    log_info "AWS Account ID: $AWS_ACCOUNT_ID"
    
    # Check Docker daemon | 檢查Docker守護程序
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check project structure | 檢查專案結構
    if [[ ! -f "$PROJECT_ROOT/Dockerfile" ]]; then
        log_error "Dockerfile not found in project root"
        exit 1
    fi
    
    if [[ ! -d "$PROJECT_ROOT/infrastructure/terraform" ]]; then
        log_error "Terraform infrastructure directory not found"
        exit 1
    fi
    
    if [[ ! -d "$PROJECT_ROOT/infrastructure/kubernetes" ]]; then
        log_error "Kubernetes manifests directory not found"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Function to generate image tag | 生成映像標籤的函數
generate_image_tag() {
    local environment="$1"
    local custom_tag="$2"
    
    if [[ -n "$custom_tag" ]]; then
        echo "$custom_tag"
        return
    fi
    
    local git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
    local timestamp=$(date +%Y%m%d%H%M%S)
    
    if [[ "$environment" == "production" ]]; then
        echo "v${timestamp}-${git_hash}"
    else
        echo "${environment}-${git_hash}-${timestamp}"
    fi
}

# Function to build and push Docker images | 構建和推送Docker映像的函數
build_and_push_images() {
    local environment="$1"
    local image_tag="$2"
    local region="$3"
    local force_build="$4"
    
    log_header "Building and pushing Docker images..."
    
    local ecr_registry="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com"
    local repository_name="aifx/trading-system"
    local full_image_name="${ecr_registry}/${repository_name}"
    
    # Login to ECR | 登錄到ECR
    log "Logging in to Amazon ECR..."
    aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$ecr_registry"
    
    # Create repository if it doesn't exist | 如果不存在則創建存儲庫
    if ! aws ecr describe-repositories --repository-names "$repository_name" --region "$region" &> /dev/null; then
        log "Creating ECR repository: $repository_name"
        aws ecr create-repository \
            --repository-name "$repository_name" \
            --region "$region" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
    fi
    
    # Check if image already exists | 檢查映像是否已存在
    if [[ "$force_build" != "true" ]] && aws ecr describe-images \
        --repository-name "$repository_name" \
        --image-ids imageTag="$image_tag" \
        --region "$region" &> /dev/null; then
        log_warning "Image $full_image_name:$image_tag already exists. Use --force to rebuild."
        return 0
    fi
    
    # Build Docker image | 構建Docker映像
    log "Building Docker image: $full_image_name:$image_tag"
    
    cd "$PROJECT_ROOT"
    
    docker build \
        --target production \
        --build-arg AIFX_ENV="$environment" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$image_tag" \
        --tag "$full_image_name:$image_tag" \
        --tag "$full_image_name:latest" \
        .
    
    # Push image to ECR | 推送映像到ECR
    log "Pushing Docker image to ECR..."
    docker push "$full_image_name:$image_tag"
    
    if [[ "$environment" == "production" ]]; then
        docker push "$full_image_name:latest"
    fi
    
    log_success "Docker image built and pushed successfully"
    echo "$full_image_name:$image_tag"
}

# Function to deploy infrastructure | 部署基礎設施的函數
deploy_infrastructure() {
    local environment="$1"
    local region="$2"
    local dry_run="$3"
    
    log_header "Deploying infrastructure with Terraform..."
    
    local tf_dir="$PROJECT_ROOT/infrastructure/terraform"
    cd "$tf_dir"
    
    # Initialize Terraform | 初始化Terraform
    log "Initializing Terraform..."
    terraform init \
        -backend-config="bucket=aifx-terraform-state-${AWS_ACCOUNT_ID}" \
        -backend-config="key=${environment}/terraform.tfstate" \
        -backend-config="region=${region}" \
        -backend-config="encrypt=true"
    
    # Create workspace if it doesn't exist | 如果不存在則創建工作空間
    terraform workspace select "$environment" 2>/dev/null || terraform workspace new "$environment"
    
    # Plan deployment | 計劃部署
    log "Planning Terraform deployment..."
    terraform plan \
        -var="environment=$environment" \
        -var="aws_region=$region" \
        -var="aws_account_id=$AWS_ACCOUNT_ID" \
        -out="tfplan-${environment}.out"
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run mode - skipping Terraform apply"
        return 0
    fi
    
    # Apply deployment | 應用部署
    log "Applying Terraform configuration..."
    terraform apply -auto-approve "tfplan-${environment}.out"
    
    # Export outputs | 導出輸出
    log "Exporting Terraform outputs..."
    terraform output -json > "${PROJECT_ROOT}/terraform-outputs-${environment}.json"
    
    log_success "Infrastructure deployment completed"
}

# Function to deploy application | 部署應用程式的函數
deploy_application() {
    local environment="$1"
    local image_tag="$2"
    local region="$3"
    local dry_run="$4"
    
    log_header "Deploying application to Kubernetes..."
    
    local cluster_name="aifx-${environment}"
    local namespace="aifx"
    
    # Update kubeconfig | 更新kubeconfig
    log "Updating kubeconfig for cluster: $cluster_name"
    aws eks update-kubeconfig --region "$region" --name "$cluster_name"
    
    # Verify cluster connection | 驗證集群連接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Read Terraform outputs | 讀取Terraform輸出
    local tf_outputs_file="$PROJECT_ROOT/terraform-outputs-${environment}.json"
    if [[ ! -f "$tf_outputs_file" ]]; then
        log_error "Terraform outputs file not found: $tf_outputs_file"
        exit 1
    fi
    
    local ecr_repository=$(jq -r '.ecr_repository_url.value' "$tf_outputs_file")
    local rds_endpoint=$(jq -r '.rds_endpoint.value' "$tf_outputs_file")
    local redis_endpoint=$(jq -r '.redis_endpoint.value' "$tf_outputs_file")
    
    # Prepare Kubernetes manifests | 準備Kubernetes清單
    log "Preparing Kubernetes manifests..."
    local temp_dir=$(mktemp -d)
    cp -r "$PROJECT_ROOT/infrastructure/kubernetes"/* "$temp_dir/"
    
    # Replace placeholders | 替換佔位符
    find "$temp_dir" -name "*.yaml" -exec sed -i \
        -e "s|\${ECR_REPOSITORY_URL}|${ecr_repository}|g" \
        -e "s|\${IMAGE_TAG}|${image_tag}|g" \
        -e "s|\${ENVIRONMENT}|${environment}|g" \
        -e "s|\${POSTGRES_HOST}|${rds_endpoint}|g" \
        -e "s|\${REDIS_HOST}|${redis_endpoint}|g" \
        -e "s|\${AWS_REGION}|${region}|g" \
        -e "s|\${AWS_ACCOUNT_ID}|${AWS_ACCOUNT_ID}|g" \
        {} \;
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run mode - showing what would be applied:"
        find "$temp_dir" -name "*.yaml" -exec echo "==> {}" \; -exec cat {} \; -exec echo "" \;
        rm -rf "$temp_dir"
        return 0
    fi
    
    # Create namespace | 創建命名空間
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests | 應用Kubernetes清單
    log "Applying Kubernetes manifests..."
    
    # Apply in specific order | 按特定順序應用
    local apply_order=(
        "namespace.yaml"
        "pvc.yaml"
        "configmaps"
        "secrets"
        "service.yaml"
        "deployment.yaml"
        "hpa.yaml"
        "ingress.yaml"
        "monitoring"
    )
    
    for resource in "${apply_order[@]}"; do
        local resource_path="$temp_dir/$resource"
        if [[ -f "$resource_path" ]] || [[ -d "$resource_path" ]]; then
            log "Applying $resource..."
            kubectl apply -f "$resource_path" -n "$namespace"
        fi
    done
    
    # Wait for deployment to be ready | 等待部署準備就緒
    log "Waiting for deployment to be ready..."
    kubectl rollout status deployment/aifx-app -n "$namespace" --timeout=600s
    kubectl rollout status deployment/aifx-ai-worker -n "$namespace" --timeout=600s
    
    # Clean up temp directory | 清理臨時目錄
    rm -rf "$temp_dir"
    
    log_success "Application deployment completed"
}

# Function to run deployment tests | 運行部署測試的函數
run_deployment_tests() {
    local environment="$1"
    local namespace="aifx"
    
    log_header "Running deployment tests..."
    
    # Test pod status | 測試Pod狀態
    log "Checking pod status..."
    kubectl get pods -n "$namespace"
    
    local app_pods=$(kubectl get pods -n "$namespace" -l app=aifx-app --field-selector=status.phase=Running -o name | wc -l)
    local worker_pods=$(kubectl get pods -n "$namespace" -l app=aifx-ai-worker --field-selector=status.phase=Running -o name | wc -l)
    
    if [[ "$app_pods" -lt 1 ]]; then
        log_error "No running application pods found"
        return 1
    fi
    
    if [[ "$worker_pods" -lt 1 ]]; then
        log_error "No running worker pods found"
        return 1
    fi
    
    log_success "$app_pods application pods and $worker_pods worker pods are running"
    
    # Test service endpoints | 測試服務端點
    log "Testing service endpoints..."
    local service_ip=$(kubectl get svc aifx-app -n "$namespace" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [[ -n "$service_ip" ]]; then
        # Wait for load balancer to be ready | 等待負載均衡器準備就緒
        sleep 60
        
        # Health check | 健康檢查
        if curl -f -s --max-time 30 "http://$service_ip:8000/health" > /dev/null; then
            log_success "Health check endpoint is responding"
        else
            log_warning "Health check endpoint not responding (this might be expected during initial deployment)"
        fi
    else
        log_info "Load balancer not available, checking pod health directly"
        
        # Direct pod health check | 直接Pod健康檢查
        local pod_name=$(kubectl get pods -n "$namespace" -l app=aifx-app --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
        if [[ -n "$pod_name" ]]; then
            kubectl port-forward -n "$namespace" "pod/$pod_name" 8080:8000 &
            local pf_pid=$!
            sleep 5
            
            if curl -f -s --max-time 10 "http://localhost:8080/health" > /dev/null; then
                log_success "Pod health check passed"
            else
                log_warning "Pod health check failed"
            fi
            
            kill $pf_pid 2>/dev/null || true
        fi
    fi
    
    # Test database connectivity | 測試資料庫連接
    log "Testing database connectivity..."
    local pod_name=$(kubectl get pods -n "$namespace" -l app=aifx-app --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
    if [[ -n "$pod_name" ]]; then
        kubectl exec -n "$namespace" "$pod_name" -- python -c "
import os, psycopg2
try:
    conn = psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        database=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD']
    )
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
        " && log_success "Database connectivity test passed" || log_warning "Database connectivity test failed"
    fi
    
    log_success "Deployment tests completed"
}

# Function to check deployment status | 檢查部署狀態的函數
check_deployment_status() {
    local environment="$1"
    local region="$2"
    local namespace="aifx"
    
    log_header "Checking deployment status for $environment environment..."
    
    # Check cluster status | 檢查集群狀態
    local cluster_name="aifx-${environment}"
    aws eks update-kubeconfig --region "$region" --name "$cluster_name" &> /dev/null || {
        log_error "Cannot access cluster: $cluster_name"
        return 1
    }
    
    # Cluster info | 集群信息
    echo ""
    log_info "=== Cluster Information ==="
    kubectl cluster-info | head -5
    
    # Node status | 節點狀態
    echo ""
    log_info "=== Node Status ==="
    kubectl get nodes -o wide
    
    # Pod status | Pod狀態
    echo ""
    log_info "=== Pod Status ==="
    kubectl get pods -n "$namespace" -o wide
    
    # Service status | 服務狀態
    echo ""
    log_info "=== Service Status ==="
    kubectl get services -n "$namespace"
    
    # Ingress status | Ingress狀態
    echo ""
    log_info "=== Ingress Status ==="
    kubectl get ingress -n "$namespace" 2>/dev/null || echo "No ingress found"
    
    # HPA status | HPA狀態
    echo ""
    log_info "=== Horizontal Pod Autoscaler Status ==="
    kubectl get hpa -n "$namespace" 2>/dev/null || echo "No HPA found"
    
    # Recent events | 最近事件
    echo ""
    log_info "=== Recent Events ==="
    kubectl get events -n "$namespace" --sort-by='.lastTimestamp' | tail -10
    
    echo ""
    log_success "Status check completed"
}

# Function to show application logs | 顯示應用程式日誌的函數
show_logs() {
    local environment="$1"
    local region="$2"
    local namespace="aifx"
    local tail_lines="${3:-100}"
    
    log_header "Showing application logs..."
    
    # Update kubeconfig | 更新kubeconfig
    local cluster_name="aifx-${environment}"
    aws eks update-kubeconfig --region "$region" --name "$cluster_name" &> /dev/null
    
    # Show logs from all application pods | 顯示所有應用程式Pod的日誌
    local pod_names=$(kubectl get pods -n "$namespace" -l app=aifx-app --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -z "$pod_names" ]]; then
        log_warning "No running application pods found"
        return 1
    fi
    
    for pod_name in $pod_names; do
        echo ""
        log_info "=== Logs from $pod_name ==="
        kubectl logs -n "$namespace" "$pod_name" --tail="$tail_lines" --timestamps=true
    done
}

# Function to rollback deployment | 回滾部署的函數
rollback_deployment() {
    local environment="$1"
    local region="$2"
    local namespace="aifx"
    
    log_header "Rolling back deployment..."
    
    # Update kubeconfig | 更新kubeconfig
    local cluster_name="aifx-${environment}"
    aws eks update-kubeconfig --region "$region" --name "$cluster_name"
    
    # Rollback application deployment | 回滾應用程式部署
    log "Rolling back application deployment..."
    kubectl rollout undo deployment/aifx-app -n "$namespace"
    kubectl rollout undo deployment/aifx-ai-worker -n "$namespace"
    
    # Wait for rollback to complete | 等待回滾完成
    log "Waiting for rollback to complete..."
    kubectl rollout status deployment/aifx-app -n "$namespace" --timeout=300s
    kubectl rollout status deployment/aifx-ai-worker -n "$namespace" --timeout=300s
    
    log_success "Rollback completed"
}

# Main function | 主函數
main() {
    local command="${1:-help}"
    local environment="$DEFAULT_ENVIRONMENT"
    local region="$DEFAULT_REGION"
    local image_tag=""
    local force="false"
    local dry_run="false"
    local skip_tests="false"
    local skip_build="false"
    local verbose="false"
    
    # Parse command line arguments | 解析命令行參數
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                environment="$2"
                shift 2
                ;;
            -r|--region)
                region="$2"
                shift 2
                ;;
            -t|--image-tag)
                image_tag="$2"
                shift 2
                ;;
            -f|--force)
                force="true"
                shift
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            --skip-tests)
                skip_tests="true"
                shift
                ;;
            --skip-build)
                skip_build="true"
                shift
                ;;
            --verbose)
                verbose="true"
                set -x
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate environment | 驗證環境
    if [[ ! "$environment" =~ ^(development|staging|production)$ ]]; then
        log_error "Invalid environment: $environment. Must be development, staging, or production."
        exit 1
    fi
    
    # Set environment variables | 設置環境變數
    export AWS_REGION="$region"
    export AIFX_ENV="$environment"
    
    # Generate image tag if not provided | 如果未提供則生成映像標籤
    if [[ -z "$image_tag" ]]; then
        image_tag=$(generate_image_tag "$environment")
    fi
    
    log "🚀 AIFX Cloud Deployment Tool"
    log "Command: $command | Environment: $environment | Region: $region | Image Tag: $image_tag"
    
    # Check prerequisites | 檢查前置條件
    if [[ "$command" != "help" ]]; then
        check_prerequisites
    fi
    
    # Execute command | 執行命令
    case $command in
        "deploy")
            if [[ "$skip_build" != "true" ]]; then
                build_and_push_images "$environment" "$image_tag" "$region" "$force"
            fi
            deploy_infrastructure "$environment" "$region" "$dry_run"
            deploy_application "$environment" "$image_tag" "$region" "$dry_run"
            if [[ "$skip_tests" != "true" && "$dry_run" != "true" ]]; then
                run_deployment_tests "$environment"
            fi
            ;;
        "infrastructure")
            deploy_infrastructure "$environment" "$region" "$dry_run"
            ;;
        "application")
            deploy_application "$environment" "$image_tag" "$region" "$dry_run"
            if [[ "$skip_tests" != "true" && "$dry_run" != "true" ]]; then
                run_deployment_tests "$environment"
            fi
            ;;
        "build")
            build_and_push_images "$environment" "$image_tag" "$region" "$force"
            ;;
        "test")
            run_deployment_tests "$environment"
            ;;
        "status")
            check_deployment_status "$environment" "$region"
            ;;
        "rollback")
            if [[ "$force" == "true" ]] || read -p "Are you sure you want to rollback $environment? [y/N] " -n 1 -r; then
                echo
                rollback_deployment "$environment" "$region"
            else
                echo
                log_info "Rollback cancelled"
            fi
            ;;
        "logs")
            show_logs "$environment" "$region" "${2:-100}"
            ;;
        "destroy")
            if [[ "$force" == "true" ]] || read -p "⚠️  Are you SURE you want to DESTROY the $environment infrastructure? This is IRREVERSIBLE! [y/N] " -n 1 -r; then
                echo
                log_warning "Destroying infrastructure for $environment environment..."
                cd "$PROJECT_ROOT/infrastructure/terraform"
                terraform destroy -var="environment=$environment" -var="aws_region=$region" -auto-approve
                log_success "Infrastructure destroyed"
            else
                echo
                log_info "Destroy cancelled"
            fi
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