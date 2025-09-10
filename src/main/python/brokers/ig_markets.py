#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IG Markets API Integration Module
IG Markets API 整合模組

This module provides secure integration with IG Markets trading platform.
此模組提供與 IG Markets 交易平台的安全整合。

Features | 功能:
- Demo and Live account support | 模擬與實盤帳戶支援
- Real-time price streaming | 實時價格串流
- Order management | 訂單管理
- Position tracking | 倉位追蹤
- Risk management | 風險管理
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import yaml
import requests
import json
import time
from urllib.parse import urlencode, parse_qs, urlparse

try:
    from trading_ig import IGService
    from trading_ig.lightstreamer import Subscription
except ImportError as e:
    raise ImportError(f"Required trading-ig package not installed: {e}")

# Configure logging | 配置日誌
logger = logging.getLogger(__name__)

class IGConnectionStatus(Enum):
    """IG connection status enumeration | IG 連接狀態枚舉"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"

class OrderType(Enum):
    """Order type enumeration | 訂單類型枚舉"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderDirection(Enum):
    """Order direction enumeration | 訂單方向枚舉"""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class IGAccount:
    """IG account information | IG 帳戶信息"""
    account_id: str
    account_name: str
    balance: float
    available: float
    margin: float
    pnl: float
    currency: str = "GBP"

@dataclass
class IGPosition:
    """IG position information | IG 倉位信息"""
    deal_id: str
    epic: str
    instrument_name: str
    direction: str
    size: float
    open_level: float
    current_level: float
    pnl: float
    currency: str

@dataclass
class IGOrder:
    """IG order information | IG 訂單信息"""
    order_type: OrderType
    direction: OrderDirection
    epic: str
    size: float
    level: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    currency_code: str = "GBP"
    force_open: bool = True

class IGWebAPIConnector:
    """
    IG Web API OAuth Connector
    IG Web API OAuth 連接器
    
    Provides OAuth authentication for Web API keys.
    為 Web API 密鑰提供 OAuth 認證。
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = 0
        self.session = requests.Session()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration | 載入配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def get_oauth_url(self, demo: bool = True) -> str:
        """Get OAuth authorization URL | 獲取 OAuth 授權 URL"""
        account_type = 'demo' if demo else 'live'
        ig_config = self.config['ig_markets'][account_type]
        
        base_url = "https://demo-api.ig.com" if demo else "https://api.ig.com"
        oauth_url = f"{base_url}/gateway/oauth/authorize"
        
        params = {
            'response_type': 'code',
            'client_id': ig_config['api_key'],
            'redirect_uri': 'http://localhost:8080/callback',
            'scope': 'read write'
        }
        
        return f"{oauth_url}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str, demo: bool = True) -> bool:
        """Exchange authorization code for access token | 交換授權碼為訪問令牌"""
        try:
            account_type = 'demo' if demo else 'live'
            ig_config = self.config['ig_markets'][account_type]
            
            base_url = "https://demo-api.ig.com" if demo else "https://api.ig.com"
            token_url = f"{base_url}/gateway/oauth/token"
            
            data = {
                'grant_type': 'authorization_code',
                'client_id': ig_config['api_key'],
                'code': code,
                'redirect_uri': 'http://localhost:8080/callback'
            }
            
            response = self.session.post(token_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data.get('refresh_token')
                self.token_expires_at = time.time() + token_data.get('expires_in', 3600)
                
                logger.info("OAuth tokens obtained successfully | OAuth 令牌獲取成功")
                return True
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False

class IGMarketsConnector:
    """
    IG Markets API Connector (Enhanced with Web API OAuth Support)
    IG Markets API 連接器（增強 Web API OAuth 支援）
    
    Provides secure connection and trading capabilities with IG Markets.
    Supports both REST API and Web API OAuth authentication.
    提供與 IG Markets 的安全連接和交易功能。
    支持 REST API 和 Web API OAuth 認證。
    """
    
    def __init__(self, config_path: str):
        """
        Initialize IG Markets connector
        初始化 IG Markets 連接器
        
        Args:
            config_path: Path to trading configuration file | 交易配置文件路徑
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.ig_service = None
        self.web_api_connector = None
        self.status = IGConnectionStatus.DISCONNECTED
        self.account_info: Optional[IGAccount] = None
        self.positions: Dict[str, IGPosition] = {}
        self.auth_method = None  # 'rest' or 'oauth'
        
        # Rate limiting | 速率限制
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit = self.config.get('rate_limiting', {}).get('requests_per_minute', 60)
        
        logger.info("IG Markets connector initialized | IG Markets 連接器已初始化")

    def _load_config(self) -> Dict[str, Any]:
        """Load trading configuration | 載入交易配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate configuration | 驗證配置
            if 'ig_markets' not in config:
                raise ValueError("IG Markets configuration not found | 未找到 IG Markets 配置")
                
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    async def connect(self, demo: bool = True, force_oauth: bool = False) -> bool:
        """
        Connect to IG Markets API (Auto-detect authentication method)
        連接到 IG Markets API（自動檢測認證方法）
        
        Args:
            demo: Use demo account if True | 如果為 True 則使用模擬帳戶
            force_oauth: Force OAuth flow for Web API keys | 強制為 Web API 密鑰使用 OAuth 流程
            
        Returns:
            bool: Connection success status | 連接成功狀態
        """
        try:
            self.status = IGConnectionStatus.CONNECTING
            
            # Get configuration | 獲取配置
            account_type = 'demo' if demo else 'live'
            ig_config = self.config['ig_markets'][account_type]
            
            if not ig_config['enabled']:
                raise ValueError(f"IG {account_type} account is disabled | IG {account_type} 帳戶已禁用")
            
            # Try REST API first (unless forced OAuth) | 首先嘗試 REST API（除非強制 OAuth）
            if not force_oauth:
                logger.info("Attempting REST API authentication | 嘗試 REST API 認證...")
                if await self._try_rest_authentication(demo):
                    self.auth_method = 'rest'
                    return True
            
            # If REST API fails, try OAuth Web API | 如果 REST API 失敗，嘗試 OAuth Web API
            logger.info("Attempting Web API OAuth authentication | 嘗試 Web API OAuth 認證...")
            return await self._try_oauth_authentication(demo)
                
        except Exception as e:
            self.status = IGConnectionStatus.ERROR
            logger.error(f"Connection failed: {e}")
            return False

    async def _try_rest_authentication(self, demo: bool) -> bool:
        """Try REST API authentication | 嘗試 REST API 認證"""
        try:
            account_type = 'demo' if demo else 'live'
            ig_config = self.config['ig_markets'][account_type]
            
            # Initialize IG service | 初始化 IG 服務
            self.ig_service = IGService(
                username=ig_config['username'],
                password=ig_config['password'], 
                api_key=ig_config['api_key'],
                acc_type='DEMO' if demo else 'LIVE'
            )
            
            # Create session | 創建會話
            response = self.ig_service.create_session()
            
            if response and response.get('accountId'):
                self.status = IGConnectionStatus.AUTHENTICATED
                await self._update_account_info()
                
                logger.info(f"Successfully connected via REST API | 成功通過 REST API 連接")
                return True
            else:
                logger.warning("REST API authentication failed | REST API 認證失敗")
                return False
                
        except Exception as e:
            logger.warning(f"REST API authentication error: {e}")
            return False

    async def _try_oauth_authentication(self, demo: bool) -> bool:
        """Try Web API OAuth authentication | 嘗試 Web API OAuth 認證"""
        try:
            # Initialize Web API connector | 初始化 Web API 連接器
            self.web_api_connector = IGWebAPIConnector(self.config_path)
            
            # Get OAuth URL | 獲取 OAuth URL
            oauth_url = await self.web_api_connector.get_oauth_url(demo)
            
            logger.info("=" * 80)
            logger.info("🔐 WEB API OAuth Authentication Required | 需要 Web API OAuth 認證")
            logger.info("=" * 80)
            logger.info("Your API key requires OAuth authentication | 您的 API 密鑰需要 OAuth 認證")
            logger.info("Please complete the following steps: | 請完成以下步驟：")
            logger.info("")
            logger.info("1. Open this URL in your browser: | 在瀏覽器中打開此 URL：")
            logger.info(f"   {oauth_url}")
            logger.info("")
            logger.info("2. Login with your IG credentials | 使用您的 IG 憑證登錄")
            logger.info("3. Authorize the application | 授權應用程式")
            logger.info("4. Copy the authorization code from the callback URL | 從回調 URL 複製授權碼")
            logger.info("5. Enter the code when prompted | 在提示時輸入授權碼")
            logger.info("")
            logger.info("=" * 80)
            
            # For now, return False to indicate manual intervention needed
            # In a production system, you'd implement a web server to handle the callback
            # 目前返回 False 表示需要手動干預
            # 在生產系統中，您需要實現一個 Web 服務器來處理回調
            
            self.auth_method = 'oauth_pending'
            logger.warning("OAuth authentication requires manual setup | OAuth 認證需要手動設置")
            return False
                
        except Exception as e:
            logger.error(f"OAuth authentication error: {e}")
            return False

    async def _update_account_info(self):
        """Update account information | 更新帳戶信息"""
        try:
            if not self.ig_service:
                return
                
            response = self.ig_service.fetch_accounts()
            if response and len(response['accounts']) > 0:
                account = response['accounts'][0]
                
                self.account_info = IGAccount(
                    account_id=account['accountId'],
                    account_name=account['accountName'],
                    balance=float(account['balance']['balance']),
                    available=float(account['balance']['available']),
                    margin=float(account['balance']['deposit']),
                    pnl=float(account['balance']['profitLoss']),
                    currency=account['currency']
                )
                
                logger.info(f"Account updated: Balance={self.account_info.balance} {self.account_info.currency}")
                
        except Exception as e:
            logger.error(f"Failed to update account info: {e}")

    async def get_market_data(self, epic: str) -> Dict[str, Any]:
        """
        Get market data for an instrument
        獲取交易工具的市場數據
        
        Args:
            epic: Instrument epic code | 交易工具代碼
            
        Returns:
            Dict: Market data | 市場數據
        """
        try:
            if not self.ig_service:
                raise Exception("Not connected to IG | 未連接到 IG")
                
            response = self.ig_service.fetch_market_by_epic(epic)
            
            if response:
                instrument = response['instrument']
                snapshot = response['snapshot']
                
                return {
                    'epic': epic,
                    'instrument_name': instrument['name'],
                    'bid': float(snapshot['bid']),
                    'ask': float(snapshot['offer']),
                    'mid': (float(snapshot['bid']) + float(snapshot['offer'])) / 2,
                    'high': float(snapshot['high']),
                    'low': float(snapshot['low']),
                    'timestamp': datetime.now(),
                    'market_status': snapshot['marketStatus']
                }
            else:
                raise Exception(f"No data for epic {epic}")
                
        except Exception as e:
            logger.error(f"Failed to get market data for {epic}: {e}")
            return {}

    async def place_order(self, order: IGOrder) -> Dict[str, Any]:
        """
        Place a trading order
        下達交易訂單
        
        Args:
            order: Order details | 訂單詳情
            
        Returns:
            Dict: Order result | 訂單結果
        """
        try:
            if not self.ig_service:
                raise Exception("Not connected to IG | 未連接到 IG")
            
            # Check risk limits | 檢查風險限制
            if not self._check_risk_limits(order):
                raise Exception("Order violates risk limits | 訂單違反風險限制")
            
            # Prepare order request | 準備訂單請求
            request = {
                'epic': order.epic,
                'direction': order.direction.value,
                'size': order.size,
                'orderType': order.order_type.value,
                'currencyCode': order.currency_code,
                'forceOpen': order.force_open
            }
            
            # Add optional parameters | 添加可選參數
            if order.level:
                request['level'] = order.level
            if order.stop_loss:
                request['stopLevel'] = order.stop_loss
            if order.take_profit:
                request['limitLevel'] = order.take_profit
            
            # Execute order | 執行訂單
            response = self.ig_service.create_open_position(**request)
            
            if response and response.get('dealReference'):
                logger.info(f"Order placed successfully: {response['dealReference']}")
                
                # Update positions | 更新倉位
                await self._update_positions()
                
                return {
                    'success': True,
                    'deal_reference': response['dealReference'],
                    'reason': response.get('reason', 'Order placed')
                }
            else:
                return {
                    'success': False,
                    'reason': response.get('reason', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {'success': False, 'reason': str(e)}

    def _check_risk_limits(self, order: IGOrder) -> bool:
        """
        Check if order complies with risk limits
        檢查訂單是否符合風險限制
        
        Args:
            order: Order to check | 要檢查的訂單
            
        Returns:
            bool: True if order is within limits | 如果訂單在限制內則返回 True
        """
        trading_config = self.config.get('trading', {})
        
        # Check position size | 檢查倉位大小
        max_size = trading_config.get('max_position_size', 1000)
        if order.size > max_size:
            logger.warning(f"Order size {order.size} exceeds limit {max_size}")
            return False
        
        # Check account balance | 檢查帳戶餘額
        if self.account_info:
            # Simple margin check (1:1 ratio for demo) | 簡單保證金檢查
            required_margin = order.size * (order.level or 1)
            if required_margin > self.account_info.available:
                logger.warning(f"Insufficient margin: required={required_margin}, available={self.account_info.available}")
                return False
        
        return True

    async def _update_positions(self):
        """Update current positions | 更新當前倉位"""
        try:
            if not self.ig_service:
                return
                
            response = self.ig_service.fetch_open_positions()
            
            if response and 'positions' in response:
                self.positions.clear()
                
                for pos in response['positions']:
                    position = IGPosition(
                        deal_id=pos['position']['dealId'],
                        epic=pos['market']['epic'],
                        instrument_name=pos['market']['instrumentName'],
                        direction=pos['position']['direction'],
                        size=float(pos['position']['size']),
                        open_level=float(pos['position']['level']),
                        current_level=float(pos['market']['bid'] if pos['position']['direction'] == 'SELL' 
                                          else pos['market']['offer']),
                        pnl=float(pos['position']['unrealisedPL']),
                        currency=pos['position']['currency']
                    )
                    
                    self.positions[position.deal_id] = position
                
                logger.info(f"Updated {len(self.positions)} positions")
                
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    async def close_position(self, deal_id: str, size: Optional[float] = None) -> Dict[str, Any]:
        """
        Close a position
        平倉
        
        Args:
            deal_id: Deal ID to close | 要平倉的交易ID
            size: Partial close size (optional) | 部分平倉大小（可選）
            
        Returns:
            Dict: Close result | 平倉結果
        """
        try:
            if not self.ig_service:
                raise Exception("Not connected to IG | 未連接到 IG")
            
            if deal_id not in self.positions:
                raise Exception(f"Position {deal_id} not found | 未找到倉位 {deal_id}")
            
            position = self.positions[deal_id]
            close_size = size or position.size
            
            # Determine close direction | 確定平倉方向
            close_direction = "SELL" if position.direction == "BUY" else "BUY"
            
            response = self.ig_service.close_open_position(
                deal_id=deal_id,
                direction=close_direction,
                size=close_size,
                order_type="MARKET"
            )
            
            if response and response.get('dealReference'):
                logger.info(f"Position {deal_id} closed successfully")
                
                # Update positions | 更新倉位
                await self._update_positions()
                
                return {
                    'success': True,
                    'deal_reference': response['dealReference'],
                    'reason': response.get('reason', 'Position closed')
                }
            else:
                return {
                    'success': False,
                    'reason': response.get('reason', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Failed to close position {deal_id}: {e}")
            return {'success': False, 'reason': str(e)}

    async def disconnect(self):
        """Disconnect from IG Markets API | 斷開與 IG Markets API 的連接"""
        try:
            if self.ig_service:
                # Close any streaming subscriptions | 關閉任何流式訂閱
                # The IG service will be garbage collected | IG 服務將被垃圾回收
                pass
                
            self.ig_service = None
            self.status = IGConnectionStatus.DISCONNECTED
            self.account_info = None
            self.positions.clear()
            
            logger.info("Disconnected from IG Markets | 已斷開與 IG Markets 的連接")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get connector status
        獲取連接器狀態
        
        Returns:
            Dict: Status information | 狀態信息
        """
        return {
            'status': self.status.value,
            'connected': self.status == IGConnectionStatus.AUTHENTICATED,
            'account_info': self.account_info.__dict__ if self.account_info else None,
            'positions_count': len(self.positions),
            'positions': [pos.__dict__ for pos in self.positions.values()]
        }

# Factory function | 工廠函數
def create_ig_connector(config_path: str = "config/trading-config.yaml") -> IGMarketsConnector:
    """
    Create an IG Markets connector instance
    創建 IG Markets 連接器實例
    
    Args:
        config_path: Path to configuration file | 配置文件路徑
        
    Returns:
        IGMarketsConnector: Configured connector instance | 配置好的連接器實例
    """
    return IGMarketsConnector(config_path)

# Example usage | 使用示例
async def main():
    """Example usage of IG Markets connector | IG Markets 連接器使用示例"""
    
    # Create connector | 創建連接器
    connector = create_ig_connector("config/trading-config.yaml")
    
    try:
        # Connect to demo account | 連接到模擬帳戶
        success = await connector.connect(demo=True)
        
        if success:
            print("✅ Connected to IG Markets demo account | 已連接到 IG Markets 模擬帳戶")
            
            # Get account status | 獲取帳戶狀態
            status = connector.get_status()
            print(f"Account Balance: {status['account_info']['balance']} {status['account_info']['currency']}")
            
            # Get market data example | 獲取市場數據示例
            market_data = await connector.get_market_data("CS.D.EURUSD.MINI.IP")
            if market_data:
                print(f"EUR/USD: Bid={market_data['bid']}, Ask={market_data['ask']}")
            
        else:
            print("❌ Failed to connect to IG Markets | 連接 IG Markets 失敗")
            
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        await connector.disconnect()

if __name__ == "__main__":
    # Run example | 運行示例
    asyncio.run(main())