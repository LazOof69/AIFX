#!/bin/bash

# AIFX Automated Database Backup Script | AIFX 自動資料庫備份腳本
# Phase 4.1.4 Database Optimization - Backup & Recovery
# 第四階段 4.1.4 資料庫優化 - 備份與恢復

set -euo pipefail

# Configuration | 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../config/backup-config.yaml"
LOG_FILE="${SCRIPT_DIR}/../logs/backup.log"

# Environment variables | 環境變量
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-aifx_production}"
POSTGRES_USER="${POSTGRES_USER:-aifx}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD}"

# S3 Configuration | S3配置
AWS_REGION="${AWS_REGION:-us-west-2}"
S3_BACKUP_BUCKET="${S3_BACKUP_BUCKET:-aifx-production-backups}"
S3_BACKUP_PREFIX="${S3_BACKUP_PREFIX:-database-backups}"

# Backup settings | 備份設置
BACKUP_DIR="/app/backups"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${BACKUP_COMPRESSION_LEVEL:-6}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY}"

# Logging function | 日誌函數
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
}

log_error() {
    log "ERROR" "$@"
}

log_warn() {
    log "WARN" "$@"
}

# Create backup directory | 創建備份目錄
setup_backup_directory() {
    log_info "Setting up backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/postgres"
    mkdir -p "$BACKUP_DIR/redis"
    mkdir -p "$BACKUP_DIR/mongodb"
    mkdir -p "$BACKUP_DIR/logs"
}

# PostgreSQL backup | PostgreSQL備份
backup_postgresql() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="$BACKUP_DIR/postgres/aifx_postgres_${timestamp}.sql"
    local compressed_file="${backup_file}.gz"
    local encrypted_file="${compressed_file}.enc"
    
    log_info "Starting PostgreSQL backup..."
    
    # Create database dump | 創建資料庫轉儲
    if PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --verbose \
        --no-password \
        --format=custom \
        --compress="$COMPRESSION_LEVEL" \
        --file="$backup_file"; then
        log_info "PostgreSQL dump created: $backup_file"
    else
        log_error "PostgreSQL backup failed"
        return 1
    fi
    
    # Compress backup | 壓縮備份
    if gzip -f "$backup_file"; then
        log_info "PostgreSQL backup compressed: $compressed_file"
    else
        log_error "PostgreSQL backup compression failed"
        return 1
    fi
    
    # Encrypt backup if encryption key provided | 如果提供加密密鑰則加密備份
    if [[ -n "${ENCRYPTION_KEY:-}" ]]; then
        if openssl enc -aes-256-cbc -salt -in "$compressed_file" -out "$encrypted_file" -k "$ENCRYPTION_KEY"; then
            log_info "PostgreSQL backup encrypted: $encrypted_file"
            rm -f "$compressed_file"
            backup_file="$encrypted_file"
        else
            log_error "PostgreSQL backup encryption failed"
            return 1
        fi
    else
        backup_file="$compressed_file"
    fi
    
    # Upload to S3 | 上傳到S3
    upload_to_s3 "$backup_file" "postgres/$(basename "$backup_file")"
    
    # Verify backup integrity | 驗證備份完整性
    verify_postgres_backup "$backup_file"
    
    log_info "PostgreSQL backup completed successfully"
}

# Redis backup | Redis備份
backup_redis() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="$BACKUP_DIR/redis/aifx_redis_${timestamp}.rdb"
    local compressed_file="${backup_file}.gz"
    
    log_info "Starting Redis backup..."
    
    # Create Redis snapshot | 創建Redis快照
    if [[ -n "${REDIS_PASSWORD:-}" ]]; then
        redis_cli_cmd="redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD"
    else
        redis_cli_cmd="redis-cli -h $REDIS_HOST -p $REDIS_PORT"
    fi
    
    # Trigger BGSAVE | 觸發BGSAVE
    if $redis_cli_cmd BGSAVE; then
        log_info "Redis BGSAVE initiated"
    else
        log_error "Redis BGSAVE failed"
        return 1
    fi
    
    # Wait for BGSAVE to complete | 等待BGSAVE完成
    while [[ "$($redis_cli_cmd LASTSAVE)" == "$($redis_cli_cmd LASTSAVE)" ]]; do
        sleep 5
        log_info "Waiting for Redis BGSAVE to complete..."
    done
    
    # Copy RDB file | 複製RDB文件
    local redis_data_dir="/var/lib/redis"
    if [[ -f "$redis_data_dir/dump.rdb" ]]; then
        cp "$redis_data_dir/dump.rdb" "$backup_file"
        log_info "Redis RDB file copied: $backup_file"
    else
        log_error "Redis RDB file not found"
        return 1
    fi
    
    # Compress backup | 壓縮備份
    if gzip -f "$backup_file"; then
        log_info "Redis backup compressed: $compressed_file"
        backup_file="$compressed_file"
    else
        log_error "Redis backup compression failed"
        return 1
    fi
    
    # Upload to S3 | 上傳到S3
    upload_to_s3 "$backup_file" "redis/$(basename "$backup_file")"
    
    log_info "Redis backup completed successfully"
}

# MongoDB backup | MongoDB備份
backup_mongodb() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_dir="$BACKUP_DIR/mongodb/aifx_mongodb_${timestamp}"
    local compressed_file="${backup_dir}.tar.gz"
    
    log_info "Starting MongoDB backup..."
    
    # Check if MongoDB is configured | 檢查是否配置了MongoDB
    if [[ -z "${MONGODB_URI:-}" ]]; then
        log_info "MongoDB URI not configured, skipping MongoDB backup"
        return 0
    fi
    
    # Create MongoDB dump | 創建MongoDB轉儲
    if mongodump --uri="$MONGODB_URI" --out="$backup_dir"; then
        log_info "MongoDB dump created: $backup_dir"
    else
        log_error "MongoDB backup failed"
        return 1
    fi
    
    # Compress backup | 壓縮備份
    if tar -czf "$compressed_file" -C "$BACKUP_DIR/mongodb" "$(basename "$backup_dir")"; then
        log_info "MongoDB backup compressed: $compressed_file"
        rm -rf "$backup_dir"
    else
        log_error "MongoDB backup compression failed"
        return 1
    fi
    
    # Upload to S3 | 上傳到S3
    upload_to_s3 "$compressed_file" "mongodb/$(basename "$compressed_file")"
    
    log_info "MongoDB backup completed successfully"
}

# Upload to S3 | 上傳到S3
upload_to_s3() {
    local local_file="$1"
    local s3_key="$2"
    local s3_path="s3://$S3_BACKUP_BUCKET/$S3_BACKUP_PREFIX/$s3_key"
    
    log_info "Uploading to S3: $s3_path"
    
    if aws s3 cp "$local_file" "$s3_path" \
        --region "$AWS_REGION" \
        --storage-class STANDARD_IA \
        --server-side-encryption AES256; then
        log_info "S3 upload successful: $s3_path"
        
        # Add metadata | 添加元數據
        aws s3api put-object-tagging \
            --bucket "$S3_BACKUP_BUCKET" \
            --key "$S3_BACKUP_PREFIX/$s3_key" \
            --tagging "TagSet=[{Key=Environment,Value=production},{Key=BackupType,Value=database},{Key=CreatedBy,Value=aifx-backup}]"
    else
        log_error "S3 upload failed: $s3_path"
        return 1
    fi
}

# Verify PostgreSQL backup | 驗證PostgreSQL備份
verify_postgres_backup() {
    local backup_file="$1"
    
    log_info "Verifying PostgreSQL backup integrity: $backup_file"
    
    # Create temporary database for verification | 創建臨時資料庫進行驗證
    local temp_db="aifx_backup_verify_$(date '+%s')"
    local temp_file="/tmp/backup_verify.sql"
    
    # Decompress and decrypt if needed | 如果需要，解壓縮和解密
    if [[ "$backup_file" == *.enc ]]; then
        openssl enc -aes-256-cbc -d -in "$backup_file" -out "$temp_file.gz" -k "$ENCRYPTION_KEY"
        gunzip "$temp_file.gz"
    elif [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" > "$temp_file"
    else
        cp "$backup_file" "$temp_file"
    fi
    
    # Create temporary database | 創建臨時資料庫
    if PGPASSWORD="$POSTGRES_PASSWORD" createdb \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        "$temp_db"; then
        log_info "Temporary verification database created: $temp_db"
    else
        log_error "Failed to create verification database"
        rm -f "$temp_file"
        return 1
    fi
    
    # Restore backup to temporary database | 將備份恢復到臨時資料庫
    if PGPASSWORD="$POSTGRES_PASSWORD" pg_restore \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$temp_db" \
        --verbose \
        --no-password \
        "$temp_file"; then
        log_info "Backup verification successful"
    else
        log_error "Backup verification failed"
        # Cleanup | 清理
        PGPASSWORD="$POSTGRES_PASSWORD" dropdb \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            "$temp_db" || true
        rm -f "$temp_file"
        return 1
    fi
    
    # Cleanup | 清理
    PGPASSWORD="$POSTGRES_PASSWORD" dropdb \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        "$temp_db"
    rm -f "$temp_file"
    
    log_info "Backup verification completed successfully"
}

# Cleanup old backups | 清理舊備份
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days"
    
    # Local cleanup | 本地清理
    find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete
    
    # S3 cleanup | S3清理
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')
    
    aws s3api list-objects-v2 \
        --bucket "$S3_BACKUP_BUCKET" \
        --prefix "$S3_BACKUP_PREFIX/" \
        --query "Contents[?LastModified<'$cutoff_date'].Key" \
        --output text | \
    while read -r key; do
        if [[ -n "$key" ]]; then
            aws s3 rm "s3://$S3_BACKUP_BUCKET/$key"
            log_info "Deleted old S3 backup: $key"
        fi
    done
}

# Send backup notification | 發送備份通知
send_notification() {
    local status="$1"
    local message="$2"
    
    if [[ -n "${BACKUP_NOTIFICATION_URL:-}" ]]; then
        curl -X POST "$BACKUP_NOTIFICATION_URL" \
            -H "Content-Type: application/json" \
            -d "{\"status\": \"$status\", \"message\": \"$message\", \"timestamp\": \"$(date -Iseconds)\"}" \
            > /dev/null 2>&1 || log_warn "Failed to send notification"
    fi
}

# Main backup function | 主備份函數
main() {
    log_info "Starting AIFX automated backup process"
    
    local start_time=$(date '+%s')
    local success=true
    local error_messages=()
    
    # Setup | 設置
    setup_backup_directory
    
    # PostgreSQL backup | PostgreSQL備份
    if ! backup_postgresql; then
        success=false
        error_messages+=("PostgreSQL backup failed")
    fi
    
    # Redis backup | Redis備份
    if ! backup_redis; then
        success=false
        error_messages+=("Redis backup failed")
    fi
    
    # MongoDB backup | MongoDB備份
    if ! backup_mongodb; then
        success=false
        error_messages+=("MongoDB backup failed")
    fi
    
    # Cleanup old backups | 清理舊備份
    cleanup_old_backups
    
    # Calculate duration | 計算持續時間
    local end_time=$(date '+%s')
    local duration=$((end_time - start_time))
    
    # Send notification | 發送通知
    if [[ "$success" == true ]]; then
        local message="AIFX backup completed successfully in ${duration}s"
        log_info "$message"
        send_notification "success" "$message"
    else
        local message="AIFX backup completed with errors: ${error_messages[*]}"
        log_error "$message"
        send_notification "error" "$message"
        exit 1
    fi
    
    log_info "AIFX backup process completed"
}

# Error handling | 錯誤處理
trap 'log_error "Backup script interrupted"; exit 1' INT TERM

# Execute main function | 執行主函數
main "$@"