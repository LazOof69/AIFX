#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIFX 依賴檢查腳本
檢查所有必要的依賴和服務是否正常工作
"""

import sys
import socket

print("🔍 AIFX 依賴檢查")
print("=" * 40)

def check_python_package(package_name, import_name=None):
    """檢查 Python 套件"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}: 已安裝")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安裝")
        return False

def check_service(host, port, service_name):
    """檢查服務是否運行"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ {service_name}: 運行中 ({host}:{port})")
            return True
        else:
            print(f"❌ {service_name}: 未運行 ({host}:{port})")
            return False
    except Exception as e:
        print(f"❌ {service_name}: 檢查失敗 - {e}")
        return False

def main():
    """主檢查函數"""
    print("\n📦 檢查 Python 依賴...")
    
    # 必要的依賴
    required_packages = [
        ("websocket-client", "websocket"),
        ("psycopg2-binary", "psycopg2"),
        ("redis", "redis"),
        ("pyyaml", "yaml"),
    ]
    
    # 可選的依賴 (完整功能)
    optional_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("prometheus_client", "prometheus_client"),
    ]
    
    required_ok = 0
    for package, import_name in required_packages:
        if check_python_package(package, import_name):
            required_ok += 1
    
    print(f"\n必要依賴: {required_ok}/{len(required_packages)} ✅")
    
    print("\n📦 檢查可選依賴...")
    optional_ok = 0
    for package, import_name in optional_packages:
        if check_python_package(package, import_name):
            optional_ok += 1
    
    print(f"可選依賴: {optional_ok}/{len(optional_packages)} ✅")
    
    print("\n🔧 檢查服務狀態...")
    
    # 檢查服務
    services = [
        ("localhost", 5432, "PostgreSQL"),
        ("localhost", 6379, "Redis"),
    ]
    
    services_ok = 0
    for host, port, name in services:
        if check_service(host, port, name):
            services_ok += 1
    
    print(f"數據庫服務: {services_ok}/{len(services)} ✅")
    
    # 總結
    print("\n" + "=" * 40)
    print("📋 檢查總結:")
    
    if required_ok == len(required_packages):
        print("✅ 所有必要依賴已安裝")
    else:
        print("❌ 部分必要依賴缺失")
        print("   安裝命令: pip install websocket-client psycopg2-binary redis pyyaml")
    
    if services_ok == len(services):
        print("✅ 所有數據庫服務運行中")
    else:
        print("❌ 部分數據庫服務未運行")
        print("   啟動命令:")
        print("   docker run --name aifx-postgres -e POSTGRES_PASSWORD=aifx_password -p 5432:5432 -d postgres")
        print("   docker run --name aifx-redis -p 6379:6379 -d redis")
    
    print(f"✅ 可選依賴完整度: {optional_ok}/{len(optional_packages)}")
    
    # 測試 AIFX 組件
    print("\n🧪 測試 AIFX 組件...")
    try:
        sys.path.insert(0, 'src/main/python')
        from data.performance_test import LoadGenerator
        gen = LoadGenerator({'symbols': ['EURUSD']})
        tick = gen.generate_forex_tick('EURUSD')
        print(f"✅ AIFX 組件正常: 生成價格 {tick['bid']:.5f}")
    except Exception as e:
        print(f"⚠️ AIFX 組件測試: {e}")
    
    # 最終狀態
    total_score = required_ok + services_ok
    max_score = len(required_packages) + len(services)
    
    if total_score == max_score:
        print("\n🎉 系統準備完成！可以運行完整的 AIFX 管道")
    elif required_ok == len(required_packages):
        print("\n⚠️ 依賴已安裝，但需要啟動數據庫服務")
    else:
        print("\n❌ 需要安裝依賴和啟動服務")
    
    return total_score == max_score

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)