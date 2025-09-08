#!/usr/bin/env python3
"""
AIFX Health Monitor Script
AIFX 健康監控腳本

Simple health monitoring for production deployment.
生產部署的簡單健康監控。
"""

import os
import time
import logging
import psutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/aifx/app/logs/health.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('aifx.health')

def check_system_health():
    """Check basic system health metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Log health metrics
        logger.info(f"System Health - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
        
        # Alert on high usage
        if cpu_percent > 90:
            logger.warning(f"High CPU usage: {cpu_percent}%")
        
        if memory_percent > 90:
            logger.warning(f"High memory usage: {memory_percent}%")
            
        if disk_percent > 90:
            logger.warning(f"High disk usage: {disk_percent}%")
            
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def main():
    """Main health monitoring loop"""
    logger.info("AIFX Health Monitor starting...")
    
    while True:
        try:
            check_system_health()
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Health monitor stopping...")
            break
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()