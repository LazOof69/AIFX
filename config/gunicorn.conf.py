# AIFX - Gunicorn Configuration for Production
# AIFX - 生產環境Gunicorn配置

import multiprocessing
import os

# ============================================================================
# Server Socket | 服務器套接字
# ============================================================================
bind = "0.0.0.0:8000"
backlog = 2048

# ============================================================================
# Worker Processes | 工作進程
# ============================================================================
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 30
keepalive = 2

# ============================================================================
# Logging | 日誌記錄
# ============================================================================
accesslog = "/home/aifx/app/logs/gunicorn_access.log"
errorlog = "/home/aifx/app/logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ============================================================================
# Process Naming | 進程命名
# ============================================================================
proc_name = 'aifx-gunicorn'

# ============================================================================
# Server Mechanics | 服務器機制
# ============================================================================
daemon = False
pidfile = '/tmp/aifx-gunicorn.pid'
user = 'aifx'
group = 'aifx'
tmp_upload_dir = '/tmp'

# ============================================================================
# SSL Configuration (if needed) | SSL配置（如需要）
# ============================================================================
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# ============================================================================
# Application Callbacks | 應用程式回調
# ============================================================================
def when_ready(server):
    """Called just after the server is started."""
    server.log.info("AIFX Gunicorn server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called just after a worker has been killed by SIGINT or SIGQUIT."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal")

# ============================================================================
# Environment Variables | 環境變數
# ============================================================================
raw_env = [
    'AIFX_ENV=production',
    'PYTHONPATH=/home/aifx/app',
]