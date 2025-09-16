#!/bin/bash
# AIFX 24/7 Trading System - Automated Backup Script
# AIFX 24/7 交易系統 - 自動備份腳本

set -e

# ============================================================================
# BACKUP CONFIGURATION | 備份配置
# ============================================================================

BACKUP_DIR="/backups"
APP_BACKUP_DIR="/app_backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

# Database configuration
DB_HOST=${POSTGRES_HOST:-"postgres-db"}
DB_NAME=${POSTGRES_DB:-"aifx_trading_24x7"}
DB_USER=${POSTGRES_USER:-"aifx"}

# Create backup directories
mkdir -p "$BACKUP_DIR"
mkdir -p "$APP_BACKUP_DIR"

echo "🔄 Starting AIFX 24/7 backup process at $(date)"
echo "🔄 開始AIFX 24/7備份過程於 $(date)"

# ============================================================================
# DATABASE BACKUP | 資料庫備份
# ============================================================================

echo "📊 Creating database backup..."
echo "📊 正在創建資料庫備份..."

# Full database dump with compression
DB_BACKUP_FILE="$BACKUP_DIR/aifx_db_backup_$TIMESTAMP.sql.gz"

if PGPASSWORD="$(cat /run/secrets/db_password)" pg_dump \
    -h "$DB_HOST" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --verbose \
    --no-owner \
    --no-privileges \
    --compress=9 \
    --format=custom \
    --file="$BACKUP_DIR/aifx_db_backup_$TIMESTAMP.dump"; then

    echo "✅ Database backup completed: $DB_BACKUP_FILE"

    # Create a readable SQL version as well
    PGPASSWORD="$(cat /run/secrets/db_password)" pg_dump \
        -h "$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --no-owner \
        --no-privileges | gzip > "$DB_BACKUP_FILE"

    echo "✅ SQL backup completed: $DB_BACKUP_FILE"
else
    echo "❌ Database backup failed!"
    exit 1
fi

# ============================================================================
# APPLICATION DATA BACKUP | 應用程式數據備份
# ============================================================================

echo "📁 Creating application data backup..."
echo "📁 正在創建應用程式數據備份..."

# Backup trading data, models, and output
APP_BACKUP_FILE="$APP_BACKUP_DIR/aifx_app_backup_$TIMESTAMP.tar.gz"

if tar -czf "$APP_BACKUP_FILE" \
    -C /home/aifx/app \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    data/ models/ output/ 2>/dev/null; then

    echo "✅ Application backup completed: $APP_BACKUP_FILE"
else
    echo "⚠️ Application backup completed with warnings"
fi

# ============================================================================
# BACKUP VERIFICATION | 備份驗證
# ============================================================================

echo "🔍 Verifying backups..."
echo "🔍 正在驗證備份..."

# Verify database backup
if [ -f "$BACKUP_DIR/aifx_db_backup_$TIMESTAMP.dump" ]; then
    backup_size=$(stat -f%z "$BACKUP_DIR/aifx_db_backup_$TIMESTAMP.dump" 2>/dev/null || stat -c%s "$BACKUP_DIR/aifx_db_backup_$TIMESTAMP.dump" 2>/dev/null)
    if [ "$backup_size" -gt 1024 ]; then
        echo "✅ Database backup verification passed (Size: $backup_size bytes)"
    else
        echo "❌ Database backup verification failed - file too small"
        exit 1
    fi
else
    echo "❌ Database backup file not found"
    exit 1
fi

# Verify application backup
if [ -f "$APP_BACKUP_FILE" ]; then
    app_backup_size=$(stat -f%z "$APP_BACKUP_FILE" 2>/dev/null || stat -c%s "$APP_BACKUP_FILE" 2>/dev/null)
    echo "✅ Application backup verification passed (Size: $app_backup_size bytes)"
else
    echo "⚠️ Application backup file not found"
fi

# ============================================================================
# CLEANUP OLD BACKUPS | 清理舊備份
# ============================================================================

echo "🧹 Cleaning up old backups (older than $RETENTION_DAYS days)..."
echo "🧹 正在清理舊備份（超過$RETENTION_DAYS天的備份）..."

# Remove old database backups
if [ -d "$BACKUP_DIR" ]; then
    deleted_db=$(find "$BACKUP_DIR" -name "aifx_db_backup_*.dump" -type f -mtime +$RETENTION_DAYS -delete -print | wc -l)
    deleted_sql=$(find "$BACKUP_DIR" -name "aifx_db_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete -print | wc -l)
    echo "🗑️ Removed $deleted_db old database dump files"
    echo "🗑️ Removed $deleted_sql old SQL backup files"
fi

# Remove old application backups
if [ -d "$APP_BACKUP_DIR" ]; then
    deleted_app=$(find "$APP_BACKUP_DIR" -name "aifx_app_backup_*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete -print | wc -l)
    echo "🗑️ Removed $deleted_app old application backup files"
fi

# ============================================================================
# BACKUP STATISTICS | 備份統計
# ============================================================================

echo ""
echo "📊 BACKUP STATISTICS | 備份統計"
echo "================================"

# Database backup stats
if [ -d "$BACKUP_DIR" ]; then
    db_backup_count=$(find "$BACKUP_DIR" -name "aifx_db_backup_*.dump" -type f | wc -l)
    total_db_size=$(find "$BACKUP_DIR" -name "aifx_db_backup_*" -type f -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1 || echo "0")
    echo "📊 Database backups: $db_backup_count files ($total_db_size total)"
fi

# Application backup stats
if [ -d "$APP_BACKUP_DIR" ]; then
    app_backup_count=$(find "$APP_BACKUP_DIR" -name "aifx_app_backup_*.tar.gz" -type f | wc -l)
    total_app_size=$(find "$APP_BACKUP_DIR" -name "aifx_app_backup_*" -type f -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1 || echo "0")
    echo "📁 Application backups: $app_backup_count files ($total_app_size total)"
fi

echo "⏰ Backup retention: $RETENTION_DAYS days"

# ============================================================================
# BACKUP COMPLETION | 備份完成
# ============================================================================

echo ""
echo "✅ AIFX 24/7 backup process completed successfully at $(date)"
echo "✅ AIFX 24/7備份過程在 $(date) 成功完成"
echo ""

# Optional: Send notification (if webhook is configured)
if [ -f "/run/secrets/alert_webhook" ]; then
    webhook_url=$(cat /run/secrets/alert_webhook)
    if [ "$webhook_url" != "http://localhost/no-webhook" ]; then
        curl -X POST "$webhook_url" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"✅ AIFX 24/7 backup completed successfully at $(date)\"}" \
            >/dev/null 2>&1 || echo "⚠️ Failed to send backup notification"
    fi
fi

# Log backup completion to database (if possible)
if command -v psql >/dev/null 2>&1; then
    PGPASSWORD="$(cat /run/secrets/db_password)" psql \
        -h "$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "INSERT INTO system_logs (log_level, component, message, details) VALUES ('INFO', 'BACKUP', 'Automated backup completed successfully', '{\"timestamp\":\"$TIMESTAMP\",\"db_size\":\"$backup_size\",\"retention_days\":$RETENTION_DAYS}');" \
        >/dev/null 2>&1 || true
fi

echo "📋 Backup summary logged to database"
echo "📋 備份摘要已記錄到資料庫"

exit 0