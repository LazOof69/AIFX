# AIFX - AI-Enhanced Forex Trading System
# Multi-stage Docker build for production deployment
# AIFX - AI增強外匯交易系統
# 生產部署的多階段Docker構建

# ============================================================================
# Stage 1: Base Python Environment | 第一階段：基礎Python環境
# ============================================================================
FROM python:3.12-slim as base

# Set environment variables | 設置環境變數
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies | 安裝系統依賴項
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    pkg-config \
    curl \
    wget \
    git \
    gnupg2 \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server | 安裝 Microsoft ODBC SQL Server 驅動程式
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application user | 創建應用程式用戶
RUN useradd --create-home --shell /bin/bash aifx
WORKDIR /home/aifx/app

# ============================================================================
# Stage 2: Dependencies Installation | 第二階段：依賴項安裝
# ============================================================================
FROM base as dependencies

# Copy requirements first for better caching | 先複製需求文件以獲得更好的緩存
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies | 安裝Python依賴項
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# ============================================================================
# Stage 3: Development Environment | 第三階段：開發環境
# ============================================================================
FROM dependencies as development

# Install development dependencies | 安裝開發依賴項
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code | 複製源代碼
COPY --chown=aifx:aifx . .

# Set development environment | 設置開發環境
ENV AIFX_ENV=development
USER aifx

# Development entrypoint | 開發入口點
CMD ["python", "-m", "src.main.python.app"]

# ============================================================================
# Stage 4: Production Environment | 第四階段：生產環境
# ============================================================================
FROM dependencies as production

# Copy application source code | 複製應用程式源代碼
COPY --chown=aifx:aifx src/ ./src/
COPY --chown=aifx:aifx config/ ./config/
COPY --chown=aifx:aifx scripts/ ./scripts/

# Copy configuration files | 複製配置文件
COPY --chown=aifx:aifx docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Create necessary directories | 創建必要目錄
RUN mkdir -p logs data models output && \
    chown -R aifx:aifx logs data models output

# Set production environment | 設置生產環境
ENV AIFX_ENV=production \
    PYTHONPATH=/home/aifx/app

# Health check | 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to application user | 切換到應用程式用戶
USER aifx

# Expose application port | 暴露應用程式端口
EXPOSE 8000

# Production entrypoint | 生產入口點
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "-m", "src.main.python.app"]

# ============================================================================
# Stage 5: Testing Environment | 第五階段：測試環境
# ============================================================================
FROM development as testing

# Install testing dependencies | 安裝測試依賴項
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy test files | 複製測試文件
COPY --chown=aifx:aifx src/test/ ./src/test/
COPY --chown=aifx:aifx pytest.ini .

# Set testing environment | 設置測試環境
ENV AIFX_ENV=testing

# Testing entrypoint | 測試入口點
CMD ["python", "-m", "pytest", "src/test/", "-v", "--cov=src"]

# ============================================================================
# Labels and Metadata | 標籤和元數據
# ============================================================================
LABEL maintainer="AIFX Development Team" \
      version="4.0.0" \
      description="AI-Enhanced Forex Trading System - Production Ready" \
      org.opencontainers.image.source="https://github.com/LazOof69/AIFX" \
      org.opencontainers.image.documentation="https://github.com/LazOof69/AIFX/blob/main/README.md" \
      org.opencontainers.image.licenses="MIT"