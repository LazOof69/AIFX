#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IG Markets OAuth Complete Solution
IG Markets OAuth 完整解決方案

Complete OAuth implementation for Web API keys with automatic callback handling.
為 Web API 密鑰提供完整的 OAuth 實現，自動處理回調。
"""

import sys
import os
import asyncio
import logging
import webbrowser
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import requests
import yaml
import json

# Add src path for imports | 添加 src 路徑用於導入
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging | 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """OAuth callback handler | OAuth 回調處理器"""
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging | 抑制預設 HTTP 服務器日誌"""
        pass
    
    def do_GET(self):
        """Handle OAuth callback | 處理 OAuth 回調"""
        try:
            # Parse URL | 解析 URL
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                # Success - got authorization code | 成功 - 獲得授權碼
                auth_code = query_params['code'][0]
                self.server.auth_code = auth_code
                self.server.callback_received = True
                
                # Send success response | 發送成功響應
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                success_html = """
                <html>
                <head><title>IG OAuth - Success</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: green;">✅ Authorization Successful!</h1>
                    <h2>✅ 授權成功！</h2>
                    <p>You can close this window and return to the AIFX application.</p>
                    <p>您可以關閉此視窗並返回 AIFX 應用程式。</p>
                    <script>setTimeout(function(){ window.close(); }, 3000);</script>
                </body>
                </html>
                """
                self.wfile.write(success_html.encode())
                
            elif 'error' in query_params:
                # Error during authorization | 授權過程中發生錯誤
                error = query_params['error'][0]
                self.server.auth_error = error
                self.server.callback_received = True
                
                # Send error response | 發送錯誤響應
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = f"""
                <html>
                <head><title>IG OAuth - Error</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: red;">❌ Authorization Failed</h1>
                    <h2>❌ 授權失敗</h2>
                    <p>Error: {error}</p>
                    <p>Please try again or contact support.</p>
                    <p>請重試或聯繫支援。</p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
            
        except Exception as e:
            logger.error(f"Callback handling error: {e}")

class IGOAuthManager:
    """
    IG OAuth Manager - Complete OAuth Flow Implementation
    IG OAuth 管理器 - 完整 OAuth 流程實現
    """
    
    def __init__(self, config_path: str = "config/trading-config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = 0
        self.callback_server = None
        self.callback_thread = None
        
    def _load_config(self) -> dict:
        """Load configuration | 載入配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    async def authenticate(self, demo: bool = True, port: int = 8080) -> bool:
        """
        Complete OAuth authentication flow
        完整的 OAuth 認證流程
        
        Args:
            demo: Use demo environment | 使用模擬環境
            port: Callback server port | 回調服務器端口
            
        Returns:
            bool: Authentication success | 認證成功
        """
        try:
            print("🚀 Starting IG OAuth Authentication Flow...")
            print("🚀 開始 IG OAuth 認證流程...")
            
            # Step 1: Start callback server | 步驟1：啟動回調服務器
            print(f"📡 Starting callback server on port {port}...")
            if not self._start_callback_server(port):
                return False
            
            # Step 2: Get authorization URL | 步驟2：獲取授權 URL
            auth_url = self._get_auth_url(demo, port)
            print(f"🔗 Opening authorization URL: {auth_url}")
            
            # Step 3: Open browser | 步驟3：打開瀏覽器
            webbrowser.open(auth_url)
            print("🌐 Browser opened for authorization...")
            print("🌐 瀏覽器已打開進行授權...")
            print("💡 Please login and authorize AIFX in your browser")
            print("💡 請在瀏覽器中登錄並授權 AIFX")
            
            # Step 4: Wait for callback | 步驟4：等待回調
            print("⏳ Waiting for authorization callback...")
            print("⏳ 等待授權回調...")
            
            auth_code = await self._wait_for_callback()
            if not auth_code:
                return False
            
            # Step 5: Exchange code for tokens | 步驟5：交換授權碼為令牌
            print("🔑 Exchanging authorization code for tokens...")
            if await self._exchange_code_for_tokens(auth_code, demo, port):
                print("✅ OAuth authentication successful!")
                print("✅ OAuth 認證成功！")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            return False
        finally:
            # Clean up server | 清理服務器
            self._stop_callback_server()
    
    def _start_callback_server(self, port: int) -> bool:
        """Start OAuth callback server | 啟動 OAuth 回調服務器"""
        try:
            # Create server | 創建服務器
            self.callback_server = HTTPServer(('localhost', port), OAuthCallbackHandler)
            self.callback_server.auth_code = None
            self.callback_server.auth_error = None
            self.callback_server.callback_received = False
            
            # Start in separate thread | 在單獨線程中啟動
            self.callback_thread = threading.Thread(
                target=self.callback_server.serve_forever,
                daemon=True
            )
            self.callback_thread.start()
            
            # Give server time to start | 給服務器啟動時間
            time.sleep(0.5)
            
            print(f"✅ Callback server started on http://localhost:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start callback server: {e}")
            return False
    
    def _stop_callback_server(self):
        """Stop callback server | 停止回調服務器"""
        if self.callback_server:
            self.callback_server.shutdown()
            self.callback_server.server_close()
            self.callback_server = None
    
    def _get_auth_url(self, demo: bool, port: int) -> str:
        """Get OAuth authorization URL | 獲取 OAuth 授權 URL"""
        account_type = 'demo' if demo else 'live'
        ig_config = self.config['ig_markets'][account_type]
        
        base_url = "https://demo-api.ig.com" if demo else "https://api.ig.com"
        
        params = {
            'response_type': 'code',
            'client_id': ig_config['api_key'],
            'redirect_uri': f'http://localhost:{port}/callback',
            'scope': 'read write'
        }
        
        from urllib.parse import urlencode
        return f"{base_url}/gateway/oauth/authorize?{urlencode(params)}"
    
    async def _wait_for_callback(self, timeout: int = 300) -> str:
        """Wait for OAuth callback | 等待 OAuth 回調"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.callback_server and self.callback_server.callback_received:
                if self.callback_server.auth_code:
                    print("✅ Authorization code received!")
                    print("✅ 已收到授權碼！")
                    return self.callback_server.auth_code
                elif self.callback_server.auth_error:
                    print(f"❌ Authorization error: {self.callback_server.auth_error}")
                    return None
            
            await asyncio.sleep(1)
        
        print("⏰ Timeout waiting for authorization")
        print("⏰ 等待授權超時")
        return None
    
    async def _exchange_code_for_tokens(self, auth_code: str, demo: bool, port: int) -> bool:
        """Exchange authorization code for tokens | 交換授權碼為令牌"""
        try:
            account_type = 'demo' if demo else 'live'
            ig_config = self.config['ig_markets'][account_type]
            
            base_url = "https://demo-api.ig.com" if demo else "https://api.ig.com"
            token_url = f"{base_url}/gateway/oauth/token"
            
            data = {
                'grant_type': 'authorization_code',
                'client_id': ig_config['api_key'],
                'code': auth_code,
                'redirect_uri': f'http://localhost:{port}/callback'
            }
            
            response = requests.post(token_url, data=data, timeout=10)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data.get('refresh_token')
                self.token_expires_at = time.time() + token_data.get('expires_in', 3600)
                
                print("🎉 Tokens obtained successfully!")
                print("🎉 令牌獲取成功！")
                print(f"   Access Token: {self.access_token[:20]}...")
                print(f"   Expires in: {token_data.get('expires_in', 3600)} seconds")
                
                return True
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False
    
    def get_authenticated_headers(self) -> dict:
        """Get headers for authenticated requests | 獲取認證請求的標頭"""
        if not self.access_token:
            raise ValueError("No access token available | 沒有可用的訪問令牌")
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    async def test_authenticated_request(self, demo: bool = True) -> bool:
        """Test authenticated API request | 測試認證的 API 請求"""
        try:
            if not self.access_token:
                print("❌ No access token available")
                return False
            
            base_url = "https://demo-api.ig.com" if demo else "https://api.ig.com"
            accounts_url = f"{base_url}/gateway/deal/accounts"
            
            headers = self.get_authenticated_headers()
            response = requests.get(accounts_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                account_data = response.json()
                print("✅ Authenticated API request successful!")
                print("✅ 認證 API 請求成功！")
                
                if 'accounts' in account_data and account_data['accounts']:
                    account = account_data['accounts'][0]
                    print(f"   Account ID: {account['accountId']}")
                    print(f"   Account Name: {account['accountName']}")
                    print(f"   Currency: {account['currency']}")
                    if 'balance' in account:
                        balance = account['balance']
                        print(f"   Balance: {balance.get('balance', 'N/A')} {account['currency']}")
                
                return True
            else:
                logger.error(f"Authenticated request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authenticated request error: {e}")
            return False

async def main():
    """Main OAuth test function | 主要 OAuth 測試函數"""
    print("=" * 80)
    print("🔐 IG Markets Complete OAuth Solution")
    print("🔐 IG Markets 完整 OAuth 解決方案")
    print("=" * 80)
    
    try:
        # Create OAuth manager | 創建 OAuth 管理器
        oauth_manager = IGOAuthManager()
        
        # Perform authentication | 執行認證
        print("🚀 Starting OAuth authentication...")
        success = await oauth_manager.authenticate(demo=True)
        
        if success:
            print("\n🎉 Authentication completed successfully!")
            print("🎉 認證成功完成！")
            
            # Test authenticated request | 測試認證請求
            print("\n🧪 Testing authenticated API request...")
            api_success = await oauth_manager.test_authenticated_request(demo=True)
            
            if api_success:
                print("\n✅ COMPLETE SUCCESS!")
                print("✅ 完全成功！")
                print("🎯 Your Web API key is now working with AIFX!")
                print("🎯 您的 Web API 密鑰現在可以與 AIFX 配合使用了！")
            else:
                print("\n⚠️ Authentication successful but API test failed")
                print("⚠️ 認證成功但 API 測試失敗")
        else:
            print("\n❌ OAuth authentication failed")
            print("❌ OAuth 認證失敗")
            print("💡 Please try again or contact IG support")
            print("💡 請重試或聯繫 IG 支援")
    
    except KeyboardInterrupt:
        print("\n⚠️ Authentication interrupted by user")
        print("⚠️ 認證被用戶中斷")
    except Exception as e:
        print(f"\n❌ Error during OAuth flow: {e}")
        print(f"❌ OAuth 流程中發生錯誤：{e}")

if __name__ == "__main__":
    asyncio.run(main())