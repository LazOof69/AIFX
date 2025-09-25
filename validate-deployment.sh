#!/bin/bash
# AIFX Ubuntu Deployment Validation Script
# AIFX Ubuntu 部署驗證腳本

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}[✅ PASS]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌ FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️  WARN]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ️  INFO]${NC} $1"
}

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"

    print_info "Testing: $test_name"

    if eval "$test_command" &>/dev/null; then
        print_success "$test_name"
        ((TESTS_PASSED++))
        return 0
    else
        print_error "$test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Header
echo -e "${BLUE}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🧪 AIFX Ubuntu Deployment Validation Script 🧪          ║
║     🧪 AIFX Ubuntu 部署驗證腳本 🧪                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
${NC}"

print_header "🔍 System Validation | 系統驗證"

# Test 1: Docker Installation
run_test "Docker installed" "command -v docker"
run_test "Docker Compose installed" "command -v docker-compose"

# Test 2: Docker Service Status
run_test "Docker service running" "systemctl is-active docker"

# Test 3: Docker permissions
run_test "Docker permissions (user in docker group)" "groups | grep docker"

# Test 4: Project Structure
print_header "📁 Project Structure Validation | 項目結構驗證"

run_test "AIFX directory exists" "[ -d 'AIFX' ]"
run_test "Cloud deployment directory exists" "[ -d 'AIFX/cloud-deployment' ]"
run_test "Docker compose file exists" "[ -f 'AIFX/cloud-deployment/docker-compose.cloud.yml' ]"
run_test "Environment file exists" "[ -f 'AIFX/cloud-deployment/.env' ]"

# Test 5: Docker Configuration Validation
print_header "🐳 Docker Configuration Validation | Docker 配置驗證"

if [ -d "AIFX/cloud-deployment" ]; then
    cd AIFX/cloud-deployment

    run_test "Docker compose file syntax" "docker-compose -f docker-compose.cloud.yml config --quiet"
    run_test "Docker compose environment variables" "docker-compose -f docker-compose.cloud.yml config | grep -q 'aifx-trading-cloud'"

    cd ../..
fi

# Test 6: Container Status
print_header "📦 Container Status Validation | 容器狀態驗證"

if [ -d "AIFX/cloud-deployment" ]; then
    cd AIFX/cloud-deployment

    # Check if containers are running
    if docker-compose -f docker-compose.cloud.yml ps | grep -q "Up"; then
        print_success "AIFX container is running"
        ((TESTS_PASSED++))

        # Get container name
        CONTAINER_NAME=$(docker-compose -f docker-compose.cloud.yml ps --services | head -n1)

        # Test container health
        run_test "Container health check" "docker exec ${CONTAINER_NAME} curl -f http://localhost:8080/api/health"

    else
        print_error "AIFX container is not running"
        ((TESTS_FAILED++))

        print_info "Container status:"
        docker-compose -f docker-compose.cloud.yml ps
    fi

    cd ../..
fi

# Test 7: Network Connectivity
print_header "🌐 Network Connectivity Validation | 網絡連接驗證"

# Check if port is accessible
PORT=$(grep AIFX_WEB_PORT AIFX/cloud-deployment/.env 2>/dev/null | cut -d'=' -f2 || echo "8080")

run_test "Port $PORT is accessible" "curl -f -s http://localhost:$PORT/api/health"
run_test "Web interface accessible" "curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT | grep -q '200'"

# Test 8: API Endpoints
print_header "🔌 API Endpoints Validation | API 端點驗證"

if curl -f -s http://localhost:$PORT/api/health &>/dev/null; then
    run_test "Health endpoint" "curl -f -s http://localhost:$PORT/api/health | grep -q 'status'"
    run_test "Docs endpoint" "curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/docs | grep -q '200'"

    # Test API response
    HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/api/health)
    if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
        print_success "API returns valid JSON response"
        ((TESTS_PASSED++))
    else
        print_error "API response is not valid JSON"
        ((TESTS_FAILED++))
    fi
else
    print_error "Cannot access API endpoints - service may not be running"
    ((TESTS_FAILED+=3))
fi

# Test 9: Resource Usage
print_header "📊 Resource Usage Validation | 資源使用驗證"

# Memory usage
MEMORY_USAGE=$(free | awk '/^Mem:/ {printf "%.0f", ($3/$2)*100}')
if [ "$MEMORY_USAGE" -lt 90 ]; then
    print_success "Memory usage: ${MEMORY_USAGE}% (< 90%)"
    ((TESTS_PASSED++))
else
    print_warning "Memory usage: ${MEMORY_USAGE}% (high usage detected)"
    ((TESTS_FAILED++))
fi

# Disk usage
DISK_USAGE=$(df / | awk 'NR==2 {printf "%.0f", ($3/$2)*100}')
if [ "$DISK_USAGE" -lt 85 ]; then
    print_success "Disk usage: ${DISK_USAGE}% (< 85%)"
    ((TESTS_PASSED++))
else
    print_warning "Disk usage: ${DISK_USAGE}% (high usage detected)"
    ((TESTS_FAILED++))
fi

# Docker container resource usage
if docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep -q aifx; then
    print_success "Container resource monitoring available"
    ((TESTS_PASSED++))

    print_info "Current container resource usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep aifx || true
else
    print_warning "Container resource monitoring not available"
fi

# Test 10: Log Files
print_header "📝 Log Files Validation | 日誌文件驗證"

if [ -d "AIFX/cloud-deployment" ]; then
    cd AIFX/cloud-deployment

    # Check if logs are being generated
    if docker-compose -f docker-compose.cloud.yml logs --tail=1 | grep -q .; then
        print_success "Application logs are being generated"
        ((TESTS_PASSED++))
    else
        print_warning "No application logs found"
    fi

    # Check for error logs
    ERROR_COUNT=$(docker-compose -f docker-compose.cloud.yml logs | grep -i error | wc -l)
    if [ "$ERROR_COUNT" -eq 0 ]; then
        print_success "No errors found in logs"
        ((TESTS_PASSED++))
    else
        print_warning "Found $ERROR_COUNT error messages in logs"
        print_info "Recent errors:"
        docker-compose -f docker-compose.cloud.yml logs | grep -i error | tail -3
    fi

    cd ../..
fi

# Test 11: Security Checks
print_header "🔒 Security Validation | 安全驗證"

# Check firewall status
if command -v ufw &>/dev/null; then
    if sudo ufw status | grep -q "Status: active"; then
        print_success "UFW firewall is active"
        ((TESTS_PASSED++))
    else
        print_warning "UFW firewall is not active"
    fi
else
    print_info "UFW not installed"
fi

# Check for default passwords
if [ -f "AIFX/cloud-deployment/.env" ]; then
    if grep -q "your-secure-api-key-here" AIFX/cloud-deployment/.env; then
        print_warning "Default API key detected - please change for production"
    else
        print_success "No default API keys found"
        ((TESTS_PASSED++))
    fi
fi

# Final Results
print_header "📋 Validation Summary | 驗證摘要"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
PASS_PERCENTAGE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    📊 TEST RESULTS 📊                        ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"

if [ $PASS_PERCENTAGE -ge 90 ]; then
    echo -e "${BLUE}║${GREEN}  ✅ VALIDATION PASSED: $TESTS_PASSED/$TOTAL_TESTS tests passed ($PASS_PERCENTAGE%)${BLUE}  ║${NC}"
    echo -e "${BLUE}║${GREEN}  🎉 AIFX deployment is HEALTHY and ready for use! 🎉${BLUE}       ║${NC}"
elif [ $PASS_PERCENTAGE -ge 75 ]; then
    echo -e "${BLUE}║${YELLOW}  ⚠️  VALIDATION WARNING: $TESTS_PASSED/$TOTAL_TESTS tests passed ($PASS_PERCENTAGE%)${BLUE} ║${NC}"
    echo -e "${BLUE}║${YELLOW}  🔧 AIFX deployment has minor issues - review above${BLUE}      ║${NC}"
else
    echo -e "${BLUE}║${RED}  ❌ VALIDATION FAILED: $TESTS_PASSED/$TOTAL_TESTS tests passed ($PASS_PERCENTAGE%)${BLUE}   ║${NC}"
    echo -e "${BLUE}║${RED}  🚨 AIFX deployment has significant issues${BLUE}               ║${NC}"
fi

echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

echo ""
if [ $TESTS_FAILED -gt 0 ]; then
    print_info "To troubleshoot failed tests, check:"
    echo "  1. Container logs: cd AIFX/cloud-deployment && docker-compose -f docker-compose.cloud.yml logs"
    echo "  2. System resources: htop, df -h, free -h"
    echo "  3. Network connectivity: curl http://localhost:$PORT/api/health"
    echo "  4. Docker status: docker ps, docker-compose ps"
fi

echo ""
print_info "For more help, see:"
echo "  - UBUNTU_SERVER_DEPLOYMENT.md"
echo "  - UBUNTU_QUICK_START.md"
echo "  - cloud-deployment/README.md"

# Exit with appropriate code
if [ $PASS_PERCENTAGE -ge 90 ]; then
    exit 0
elif [ $PASS_PERCENTAGE -ge 75 ]; then
    exit 1
else
    exit 2
fi