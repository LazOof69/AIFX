#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IG Markets API Solution Test Script
IG Markets API 解決方案測試腳本

Comprehensive test script that tries all available authentication methods
for IG Markets API integration.
全面的測試腳本，嘗試 IG Markets API 整合的所有可用認證方法。
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add src path for imports | 添加 src 路徑用於導入
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging | 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print formatted header | 打印格式化標題"""
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80)

def print_section(title: str):
    """Print section divider | 打印部分分隔符"""
    print(f"\n📋 {title}")
    print("-" * 50)

def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result | 打印測試結果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"   {details}")

async def test_enhanced_ig_connector():
    """Test enhanced IG Markets connector | 測試增強的 IG Markets 連接器"""
    print_header("Enhanced IG Markets API Test Suite")
    
    results = {
        'imports': False,
        'config_load': False,
        'rest_auth': False,
        'oauth_detection': False,
        'connector_status': False
    }
    
    # Test 1: Import enhanced connector | 測試1：導入增強連接器
    print_section("Testing Enhanced Connector Imports")
    try:
        from src.main.python.brokers.ig_markets import (
            IGMarketsConnector, IGWebAPIConnector, create_ig_connector
        )
        results['imports'] = True
        print_result("Enhanced connector imports", True, "IGMarketsConnector, IGWebAPIConnector imported")
    except Exception as e:
        print_result("Enhanced connector imports", False, f"Import error: {e}")
        return results
    
    # Test 2: Configuration loading | 測試2：配置載入
    print_section("Testing Configuration Loading")
    try:
        connector = create_ig_connector("config/trading-config.yaml")
        results['config_load'] = True
        print_result("Configuration loading", True, "Config loaded successfully")
    except Exception as e:
        print_result("Configuration loading", False, f"Config error: {e}")
        return results
    
    # Test 3: Auto-detect authentication | 測試3：自動檢測認證
    print_section("Testing Auto-Detection Authentication")
    try:
        print("🔄 Attempting connection with auto-detection...")
        success = await connector.connect(demo=True)
        
        if success:
            results['rest_auth'] = True
            print_result("REST API authentication", True, f"Method: {connector.auth_method}")
            
            # Get status | 獲取狀態
            status = connector.get_status()
            if status['connected']:
                results['connector_status'] = True
                print_result("Connector status", True, f"Status: {status['status']}")
                
                # Show account info if available | 如果可用則顯示帳戶信息
                if status['account_info']:
                    account = status['account_info']
                    print(f"   Account ID: {account['account_id']}")
                    print(f"   Balance: {account['balance']} {account['currency']}")
        else:
            print_result("REST API authentication", False, f"Method attempted: {connector.auth_method}")
            
            # If OAuth is pending, show instructions | 如果 OAuth 待處理，顯示說明
            if connector.auth_method == 'oauth_pending':
                results['oauth_detection'] = True
                print_result("OAuth detection", True, "Web API OAuth flow detected")
                await show_oauth_instructions(connector)
                
    except Exception as e:
        print_result("Auto-detection authentication", False, f"Error: {e}")
    
    finally:
        # Cleanup | 清理
        try:
            await connector.disconnect()
        except:
            pass
    
    return results

async def show_oauth_instructions(connector):
    """Show OAuth setup instructions | 顯示 OAuth 設置說明"""
    print_section("OAuth Setup Instructions")
    
    try:
        if connector.web_api_connector:
            oauth_url = await connector.web_api_connector.get_oauth_url(demo=True)
            
            print("🔐 Your API key requires OAuth authentication!")
            print("🔐 您的 API 密鑰需要 OAuth 認證！")
            print("")
            print("📋 Steps to complete authentication | 完成認證的步驟:")
            print("1. Open this URL in browser | 在瀏覽器中打開此 URL:")
            print(f"   {oauth_url}")
            print("")
            print("2. Login with IG credentials | 使用 IG 憑證登錄")
            print("3. Authorize the application | 授權應用程式")
            print("4. Get authorization code from callback | 從回調獲取授權碼")
            print("")
            print("💡 Alternative Solutions | 替代解決方案:")
            print("   A) Contact IG support for REST API key | 聯繫 IG 支持獲取 REST API 密鑰")
            print("   B) Implement full OAuth flow | 實現完整 OAuth 流程")
            print("   C) Use alternative broker (MetaTrader, OANDA) | 使用替代券商")
            
    except Exception as e:
        logger.error(f"Error showing OAuth instructions: {e}")

def test_direct_rest_api():
    """Test direct REST API call | 測試直接 REST API 調用"""
    print_section("Testing Direct REST API")
    
    try:
        import requests
        import yaml
        
        # Load config | 載入配置
        with open('config/trading-config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        demo_config = config['ig_markets']['demo']
        
        # Direct API call | 直接 API 調用
        url = "https://demo-api.ig.com/gateway/deal/session"
        headers = {
            'Content-Type': 'application/json; charset=UTF-8',
            'Accept': 'application/json; charset=UTF-8',
            'X-IG-API-KEY': demo_config['api_key'],
            'Version': '2'
        }
        
        payload = {
            'identifier': demo_config['username'],
            'password': demo_config['password']
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print_result("Direct REST API", True, "Authentication successful")
            return True
        else:
            error_msg = "Authentication failed"
            try:
                error_data = response.json()
                if error_data.get('errorCode') == 'error.security.invalid-details':
                    error_msg = "Invalid API key type (Web API detected)"
            except:
                pass
            
            print_result("Direct REST API", False, error_msg)
            return False
            
    except Exception as e:
        print_result("Direct REST API", False, f"Error: {e}")
        return False

async def run_comprehensive_test():
    """Run comprehensive IG API test suite | 運行全面的 IG API 測試套件"""
    print_header("AIFX IG Markets API Comprehensive Test Suite")
    print("🧪 Testing all available authentication methods...")
    print("🧪 測試所有可用的認證方法...")
    
    # Test results | 測試結果
    all_results = {}
    
    # Test 1: Direct REST API | 測試1：直接 REST API
    print_section("Phase 1: Direct REST API Test")
    direct_rest_success = test_direct_rest_api()
    all_results['direct_rest'] = direct_rest_success
    
    # Test 2: Enhanced connector | 測試2：增強連接器
    print_section("Phase 2: Enhanced Connector Test")
    connector_results = await test_enhanced_ig_connector()
    all_results.update(connector_results)
    
    # Final summary | 最終摘要
    print_header("Test Results Summary")
    
    total_tests = len(all_results)
    passed_tests = sum(all_results.values())
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    print("")
    
    for test_name, result in all_results.items():
        print_result(test_name.replace('_', ' ').title(), result)
    
    print_section("Diagnosis & Recommendations")
    
    if all_results.get('direct_rest', False):
        print("✅ SUCCESS: Your API key works with REST API!")
        print("✅ AIFX integration is ready to use")
        print("✅ 成功：您的 API 密鑰適用於 REST API！")
        print("✅ AIFX 整合已準備就緒")
    elif all_results.get('oauth_detection', False):
        print("🔐 DETECTED: Your API key is Web API type")
        print("🔧 SOLUTION: OAuth authentication required")
        print("🔐 檢測到：您的 API 密鑰是 Web API 類型")
        print("🔧 解決方案：需要 OAuth 認證")
        print("")
        print("📋 Recommended Actions:")
        print("1. Contact IG support for REST API key (easiest)")
        print("2. Implement OAuth flow (technical)")
        print("3. Use alternative broker integration")
    else:
        print("❌ ISSUE: Neither authentication method worked")
        print("🔍 DIAGNOSIS: Check credentials and account status")
        print("❌ 問題：兩種認證方法都不起作用")
        print("🔍 診斷：檢查憑證和帳戶狀態")
    
    print_section("Next Steps")
    
    if passed_tests >= 3:
        print("🎉 READY: System is working - proceed with trading strategy")
        print("📈 Continue with AIFX Phase 3: Strategy Integration")
    elif all_results.get('oauth_detection'):
        print("🔧 SETUP: Complete OAuth authentication setup")
        print("📞 CONTACT: IG support for REST API key upgrade")
    else:
        print("🆘 SUPPORT: Contact IG support for API access issues")
        print("🔄 ALTERNATIVE: Consider other broker integrations")

def main():
    """Main function | 主函數"""
    try:
        asyncio.run(run_comprehensive_test())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user | 測試被用戶中斷")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        print(f"❌ 測試過程中發生意外錯誤：{e}")

if __name__ == "__main__":
    main()