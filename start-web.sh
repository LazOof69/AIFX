#!/bin/bash

# AIFX Simplified Web Trading Signals - Quick Start Script
# AIFX 簡化網頁交易信號 - 快速啟動腳本

set -e  # Exit on any error

echo "🚀 AIFX Trading Signals - Quick Start"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

print_status "Docker and Docker Compose are available"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs data/cache nginx/ssl

# Set permissions
chmod 755 logs data/cache

# Default configuration
COMPOSE_FILE="docker-compose.web.yml"
PROFILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-cache)
            PROFILE="${PROFILE} --profile with-cache"
            print_status "Redis caching will be enabled"
            shift
            ;;
        --with-nginx)
            PROFILE="${PROFILE} --profile with-nginx"
            print_status "Nginx reverse proxy will be enabled"
            shift
            ;;
        --full)
            PROFILE="--profile with-cache --profile with-nginx"
            print_status "Full production setup will be used"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-cache    Enable Redis caching"
            echo "  --with-nginx    Enable Nginx reverse proxy"
            echo "  --full          Enable all optional services"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Basic web interface only"
            echo "  $0 --with-cache      # With Redis caching"
            echo "  $0 --with-nginx      # With Nginx proxy"
            echo "  $0 --full            # Full production setup"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create Nginx configuration if needed
if [[ $PROFILE == *"with-nginx"* ]]; then
    print_status "Creating Nginx configuration..."
    mkdir -p nginx
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream aifx-web {
        server aifx-web-signals:8080;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://aifx-web;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws/ {
            proxy_pass http://aifx-web;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://aifx-web;
            access_log off;
        }
    }
}
EOF
    print_status "Nginx configuration created"
fi

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true

# Build and start services
print_status "Building and starting AIFX Trading Signals..."
docker-compose -f $COMPOSE_FILE $PROFILE up -d --build

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Check if the main service is healthy
for i in {1..30}; do
    if docker-compose -f $COMPOSE_FILE ps | grep -q "healthy"; then
        print_status "Services are healthy!"
        break
    elif [ $i -eq 30 ]; then
        print_warning "Services might not be ready yet. Check logs for details."
    else
        echo -n "."
        sleep 2
    fi
done

echo ""
print_status "🎉 AIFX Trading Signals is now running!"
echo ""
echo "📊 Access your trading signals dashboard at:"

if [[ $PROFILE == *"with-nginx"* ]]; then
    echo "   🌐 Main Interface: http://localhost/"
    echo "   ❤️  Health Check:  http://localhost/health"
else
    echo "   🌐 Main Interface: http://localhost:8080/"
    echo "   ❤️  Health Check:  http://localhost:8080/api/health"
fi

echo ""
echo "📋 Useful commands:"
echo "   View logs:    docker-compose -f $COMPOSE_FILE logs -f"
echo "   Stop service: docker-compose -f $COMPOSE_FILE down"
echo "   Restart:      docker-compose -f $COMPOSE_FILE restart"
echo ""

# Show running containers
print_status "Currently running containers:"
docker-compose -f $COMPOSE_FILE ps

echo ""
print_status "Setup complete! Your 24/7 trading signals are now active. 🚀"