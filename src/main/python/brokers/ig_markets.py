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
from jsonschema import validate, ValidationError

try:
    from trading_ig import IGService
    from trading_ig.lightstreamer import Subscription
except ImportError as e:
    raise ImportError(f"Required trading-ig package not installed: {e}")

# Configure logging | 配置日誌
logger = logging.getLogger(__name__)

# JSON Schema validation for API responses | API 響應的 JSON Schema 驗證
MARKET_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "instrument": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        },
        "snapshot": {
            "type": "object",
            "properties": {
                "bid": {"type": ["number", "string"]},
                "offer": {"type": ["number", "string"]},
                "high": {"type": ["number", "string"]},
                "low": {"type": ["number", "string"]},
                "marketStatus": {"type": "string"}
            },
            "required": ["bid", "offer"]
        }
    },
    "required": ["instrument", "snapshot"]
}

ORDER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "dealReference": {"type": "string"},
        "reason": {"type": "string"}
    },
    "required": ["dealReference"]
}

POSITIONS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "positions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "object",
                        "properties": {
                            "dealId": {"type": "string"},
                            "direction": {"type": "string"},
                            "size": {"type": ["number", "string"]},
                            "level": {"type": ["number", "string"]},
                            "unrealisedPL": {"type": ["number", "string"]},
                            "currency": {"type": "string"}
                        },
                        "required": ["dealId", "direction", "size"]
                    },
                    "market": {
                        "type": "object",
                        "properties": {
                            "epic": {"type": "string"},
                            "instrumentName": {"type": "string"},
                            "bid": {"type": ["number", "string"]},
                            "offer": {"type": ["number", "string"]}
                        },
                        "required": ["epic", "instrumentName"]
                    }
                },
                "required": ["position", "market"]
            }
        }
    },
    "required": ["positions"]
}

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

def validate_json_response(response_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate JSON response against schema
    根據模式驗證 JSON 響應
    
    Args:
        response_data: JSON response data | JSON 響應數據
        schema: JSON Schema to validate against | 用於驗證的 JSON Schema
        
    Returns:
        bool: True if valid, False otherwise | 如果有效則返回 True，否則返回 False
    """
    try:
        validate(instance=response_data, schema=schema)
        return True
    except ValidationError as e:
        logger.warning(f"JSON validation failed: {e.message}")
        return False
    except Exception as e:
        logger.error(f"JSON validation error: {e}")
        return False

def safe_float_conversion(value: Union[str, int, float], default: float = 0.0) -> float:
    """
    Safely convert value to float with proper error handling
    安全地將值轉換為浮點數並進行適當的錯誤處理
    
    Args:
        value: Value to convert | 要轉換的值
        default: Default value if conversion fails | 轉換失敗時的默認值
        
    Returns:
        float: Converted value or default | 轉換後的值或默認值
    """
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to float, using default {default}")
        return default

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
        Get market data for an instrument using REST API standards
        使用 REST API 標準獲取交易工具的市場數據
        
        This method follows IG's REST API guide:
        - GET HTTP request to https://api.ig.com/gateway/deal/markets/{epic}
        - JSON response format handling
        - Proper error handling and status codes
        
        Args:
            epic: Instrument epic code | 交易工具代碼
            
        Returns:
            Dict: Market data following REST standards | 遵循 REST 標準的市場數據
        """
        try:
            # Use REST API authentication if available | 如果可用，使用 REST API 認證
            if self.ig_service and self.auth_method == 'rest':
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
                        'market_status': snapshot['marketStatus'],
                        'source': 'REST_API'
                    }
                else:
                    raise Exception(f"No data for epic {epic}")
            
            # Use Web API OAuth if REST not available | 如果 REST 不可用，使用 Web API OAuth
            elif self.web_api_connector and self.auth_method == 'oauth':
                return await self._get_market_data_oauth(epic)
            
            else:
                raise Exception("No valid authentication method available | 沒有可用的有效認證方法")
                
        except Exception as e:
            logger.error(f"Failed to get market data for {epic}: {e}")
            return {}

    async def _get_market_data_oauth(self, epic: str) -> Dict[str, Any]:
        """
        Get market data using OAuth Web API (REST compliant)
        使用 OAuth Web API 獲取市場數據（符合 REST 標準）
        
        Implements REST principles:
        - GET request to resource endpoint
        - JSON response handling
        - Proper HTTP status code validation
        
        Args:
            epic: Instrument epic code | 交易工具代碼
            
        Returns:
            Dict: Market data | 市場數據
        """
        try:
            account_type = 'demo' if 'demo' in str(self.config_path) else 'live'
            base_url = "https://demo-api.ig.com" if 'demo' in account_type else "https://api.ig.com"
            
            # REST API endpoint for markets | 市場的 REST API 端點
            url = f"{base_url}/gateway/deal/markets/{epic}"
            
            headers = {
                'Authorization': f'Bearer {self.web_api_connector.access_token}',
                'Content-Type': 'application/json; charset=UTF-8',
                'Accept': 'application/json; charset=UTF-8',
                'X-IG-API-KEY': self.config['ig_markets']['demo']['api_key'],
                'Version': '3'  # Use version 3 for comprehensive market data
            }
            
            response = self.web_api_connector.session.get(url, headers=headers, timeout=10)
            
            # REST API standard response handling with JSON validation | REST API 標準響應處理並進行 JSON 驗證
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Validate response structure | 驗證響應結構
                    if validate_json_response(data, MARKET_DATA_SCHEMA):
                        instrument = data.get('instrument', {})
                        snapshot = data.get('snapshot', {})
                        
                        # Use safe float conversion | 使用安全浮點數轉換
                        bid = safe_float_conversion(snapshot.get('bid', 0))
                        ask = safe_float_conversion(snapshot.get('offer', 0))
                        
                        return {
                            'epic': epic,
                            'instrument_name': instrument.get('name', 'Unknown'),
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2 if bid > 0 and ask > 0 else 0,
                            'high': safe_float_conversion(snapshot.get('high', 0)),
                            'low': safe_float_conversion(snapshot.get('low', 0)),
                            'timestamp': datetime.now(),
                            'market_status': snapshot.get('marketStatus', 'UNKNOWN'),
                            'source': 'OAUTH_API',
                            'http_status': response.status_code,
                            'validated': True
                        }
                    else:
                        logger.warning(f"Market data response validation failed for {epic}")
                        return {
                            'epic': epic,
                            'error': 'Invalid response structure',
                            'source': 'OAUTH_API',
                            'http_status': response.status_code,
                            'validated': False
                        }
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response for market data: {epic}")
                    raise Exception("Invalid JSON response from IG API")
                
            elif response.status_code == 401:
                # Token might be expired | 令牌可能已過期
                logger.warning("OAuth token expired, attempting refresh | OAuth 令牌已過期，嘗試刷新")
                # TODO: Implement token refresh logic
                raise Exception("OAuth token expired | OAuth 令牌已過期")
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Market data request failed: {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"OAuth market data request failed for {epic}: {e}")
            raise

    async def place_order(self, order: IGOrder) -> Dict[str, Any]:
        """
        Place a trading order using REST API standards
        使用 REST API 標準下達交易訂單
        
        This method follows IG's REST API guide:
        - POST HTTP request to https://api.ig.com/gateway/deal/positions/otc
        - JSON request and response format
        - Proper HTTP status code handling
        
        Args:
            order: Order details | 訂單詳情
            
        Returns:
            Dict: Order result following REST standards | 遵循 REST 標準的訂單結果
        """
        try:
            # Check risk limits first | 首先檢查風險限制
            if not self._check_risk_limits(order):
                raise Exception("Order violates risk limits | 訂單違反風險限制")
            
            # Use REST API authentication if available | 如果可用，使用 REST API 認證
            if self.ig_service and self.auth_method == 'rest':
                return await self._place_order_rest(order)
            
            # Use Web API OAuth if REST not available | 如果 REST 不可用，使用 Web API OAuth
            elif self.web_api_connector and self.auth_method == 'oauth':
                return await self._place_order_oauth(order)
            
            else:
                raise Exception("No valid authentication method available | 沒有可用的有效認證方法")
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {'success': False, 'reason': str(e), 'http_status': None}

    async def _place_order_rest(self, order: IGOrder) -> Dict[str, Any]:
        """
        Place order using REST API (trading-ig library)
        使用 REST API 下單（trading-ig 庫）
        """
        try:
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
                logger.info(f"REST order placed successfully: {response['dealReference']}")
                
                # Update positions | 更新倉位
                await self._update_positions()
                
                return {
                    'success': True,
                    'deal_reference': response['dealReference'],
                    'reason': response.get('reason', 'Order placed'),
                    'source': 'REST_API',
                    'http_status': 200
                }
            else:
                return {
                    'success': False,
                    'reason': response.get('reason', 'Unknown error'),
                    'source': 'REST_API',
                    'http_status': None
                }
                
        except Exception as e:
            logger.error(f"REST order placement failed: {e}")
            return {'success': False, 'reason': str(e), 'source': 'REST_API'}

    async def _place_order_oauth(self, order: IGOrder) -> Dict[str, Any]:
        """
        Place order using OAuth Web API (REST compliant)
        使用 OAuth Web API 下單（符合 REST 標準）
        
        Implements REST principles:
        - POST request to positions endpoint
        - JSON request body
        - Proper HTTP status code validation
        
        Args:
            order: Order details | 訂單詳情
            
        Returns:
            Dict: Order result | 訂單結果
        """
        try:
            account_type = 'demo' if 'demo' in str(self.config_path) else 'live'
            base_url = "https://demo-api.ig.com" if 'demo' in account_type else "https://api.ig.com"
            
            # REST API endpoint for creating positions | 創建倉位的 REST API 端點
            url = f"{base_url}/gateway/deal/positions/otc"
            
            headers = {
                'Authorization': f'Bearer {self.web_api_connector.access_token}',
                'Content-Type': 'application/json; charset=UTF-8',
                'Accept': 'application/json; charset=UTF-8',
                'X-IG-API-KEY': self.config['ig_markets']['demo']['api_key'],
                'Version': '2'  # Use version 2 for position creation
            }
            
            # Prepare JSON request body following IG REST API specification
            # 根據 IG REST API 規範準備 JSON 請求體
            payload = {
                'epic': order.epic,
                'direction': order.direction.value,
                'size': str(order.size),  # Size as string per IG API spec
                'orderType': order.order_type.value,
                'currencyCode': order.currency_code,
                'forceOpen': order.force_open
            }
            
            # Add optional parameters | 添加可選參數
            if order.level:
                payload['level'] = str(order.level)
            if order.stop_loss:
                payload['stopLevel'] = str(order.stop_loss)
            if order.take_profit:
                payload['limitLevel'] = str(order.take_profit)
            
            # Execute POST request | 執行 POST 請求
            response = self.web_api_connector.session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=15
            )
            
            # REST API standard response handling | REST API 標準響應處理
            if response.status_code == 200:
                data = response.json()
                deal_reference = data.get('dealReference')
                
                if deal_reference:
                    logger.info(f"OAuth order placed successfully: {deal_reference}")
                    
                    # Update positions | 更新倉位
                    await self._update_positions()
                    
                    return {
                        'success': True,
                        'deal_reference': deal_reference,
                        'reason': data.get('reason', 'Order placed via OAuth'),
                        'source': 'OAUTH_API',
                        'http_status': response.status_code
                    }
                else:
                    return {
                        'success': False,
                        'reason': data.get('errorCode', 'No deal reference returned'),
                        'source': 'OAUTH_API',
                        'http_status': response.status_code
                    }
                    
            elif response.status_code == 400:
                # Bad request - validation error | 錯誤請求 - 驗證錯誤
                error_data = response.json() if response.text else {}
                error_msg = error_data.get('errorCode', 'Invalid order parameters')
                logger.error(f"Order validation failed: {error_msg}")
                
                return {
                    'success': False,
                    'reason': f"Validation error: {error_msg}",
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
            elif response.status_code == 401:
                # Unauthorized - token expired | 未授權 - 令牌過期
                logger.warning("OAuth token expired during order placement | 下單時 OAuth 令牌過期")
                return {
                    'success': False,
                    'reason': 'OAuth token expired',
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
            elif response.status_code == 403:
                # Forbidden - insufficient permissions | 禁止 - 權限不足
                return {
                    'success': False,
                    'reason': 'Insufficient permissions for trading',
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Order placement failed: {error_msg}")
                
                return {
                    'success': False,
                    'reason': error_msg,
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
        except Exception as e:
            logger.error(f"OAuth order placement failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'source': 'OAUTH_API',
                'http_status': None
            }

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
        Close a position using REST API standards
        使用 REST API 標準平倉
        
        This method follows IG's REST API guide:
        - DELETE HTTP request to https://api.ig.com/gateway/deal/positions/otc
        - Or POST request to close endpoint with proper JSON payload
        - Proper HTTP status code handling
        
        Args:
            deal_id: Deal ID to close | 要平倉的交易ID
            size: Partial close size (optional) | 部分平倉大小（可選）
            
        Returns:
            Dict: Close result following REST standards | 遵循 REST 標準的平倉結果
        """
        try:
            if deal_id not in self.positions:
                raise Exception(f"Position {deal_id} not found | 未找到倉位 {deal_id}")
            
            # Use REST API authentication if available | 如果可用，使用 REST API 認證
            if self.ig_service and self.auth_method == 'rest':
                return await self._close_position_rest(deal_id, size)
            
            # Use Web API OAuth if REST not available | 如果 REST 不可用，使用 Web API OAuth
            elif self.web_api_connector and self.auth_method == 'oauth':
                return await self._close_position_oauth(deal_id, size)
            
            else:
                raise Exception("No valid authentication method available | 沒有可用的有效認證方法")
                
        except Exception as e:
            logger.error(f"Failed to close position {deal_id}: {e}")
            return {'success': False, 'reason': str(e), 'http_status': None}

    async def _close_position_rest(self, deal_id: str, size: Optional[float] = None) -> Dict[str, Any]:
        """
        Close position using REST API (trading-ig library)
        使用 REST API 平倉（trading-ig 庫）
        """
        try:
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
                logger.info(f"REST position {deal_id} closed successfully")
                
                # Update positions | 更新倉位
                await self._update_positions()
                
                return {
                    'success': True,
                    'deal_reference': response['dealReference'],
                    'reason': response.get('reason', 'Position closed'),
                    'source': 'REST_API',
                    'http_status': 200
                }
            else:
                return {
                    'success': False,
                    'reason': response.get('reason', 'Unknown error'),
                    'source': 'REST_API',
                    'http_status': None
                }
                
        except Exception as e:
            logger.error(f"REST position close failed: {e}")
            return {'success': False, 'reason': str(e), 'source': 'REST_API'}

    async def _close_position_oauth(self, deal_id: str, size: Optional[float] = None) -> Dict[str, Any]:
        """
        Close position using OAuth Web API (REST compliant)
        使用 OAuth Web API 平倉（符合 REST 標準）
        
        Implements REST principles:
        - DELETE or POST request to positions endpoint
        - JSON request body for close parameters
        - Proper HTTP status code validation
        
        Args:
            deal_id: Deal ID to close | 要平倉的交易ID
            size: Partial close size (optional) | 部分平倉大小（可選）
            
        Returns:
            Dict: Close result | 平倉結果
        """
        try:
            position = self.positions[deal_id]
            close_size = size or position.size
            
            account_type = 'demo' if 'demo' in str(self.config_path) else 'live'
            base_url = "https://demo-api.ig.com" if 'demo' in account_type else "https://api.ig.com"
            
            # REST API endpoint for closing positions | 平倉的 REST API 端點
            url = f"{base_url}/gateway/deal/positions/otc"
            
            headers = {
                'Authorization': f'Bearer {self.web_api_connector.access_token}',
                'Content-Type': 'application/json; charset=UTF-8',
                'Accept': 'application/json; charset=UTF-8',
                'X-IG-API-KEY': self.config['ig_markets']['demo']['api_key'],
                'Version': '1',  # Use version 1 for position closure
                '_method': 'DELETE'  # Override HTTP method for position closure
            }
            
            # Determine close direction | 確定平倉方向
            close_direction = "SELL" if position.direction == "BUY" else "BUY"
            
            # Prepare JSON request body following IG REST API specification
            # 根據 IG REST API 規範準備 JSON 請求體
            payload = {
                'dealId': deal_id,
                'direction': close_direction,
                'size': str(close_size),  # Size as string per IG API spec
                'orderType': 'MARKET'
            }
            
            # Execute DELETE request (using POST with _method override) | 執行 DELETE 請求（使用 POST 並覆蓋方法）
            response = self.web_api_connector.session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=15
            )
            
            # REST API standard response handling | REST API 標準響應處理
            if response.status_code == 200:
                try:
                    data = response.json()
                    deal_reference = data.get('dealReference')
                    
                    if deal_reference:
                        logger.info(f"OAuth position {deal_id} closed successfully: {deal_reference}")
                        
                        # Update positions | 更新倉位
                        await self._update_positions()
                        
                        return {
                            'success': True,
                            'deal_reference': deal_reference,
                            'reason': data.get('reason', 'Position closed via OAuth'),
                            'source': 'OAUTH_API',
                            'http_status': response.status_code,
                            'validated': True
                        }
                    else:
                        return {
                            'success': False,
                            'reason': data.get('errorCode', 'No deal reference returned'),
                            'source': 'OAUTH_API',
                            'http_status': response.status_code
                        }
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON response for position close")
                    return {
                        'success': False,
                        'reason': 'Invalid JSON response from IG API',
                        'source': 'OAUTH_API',
                        'http_status': response.status_code
                    }
                    
            elif response.status_code == 400:
                # Bad request - validation error | 錯誤請求 - 驗證錯誤
                try:
                    error_data = response.json()
                    error_msg = error_data.get('errorCode', 'Invalid close parameters')
                except:
                    error_msg = 'Invalid close parameters'
                
                logger.error(f"Position close validation failed: {error_msg}")
                return {
                    'success': False,
                    'reason': f"Validation error: {error_msg}",
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
            elif response.status_code == 401:
                # Unauthorized - token expired | 未授權 - 令牌過期
                logger.warning("OAuth token expired during position close | 平倉時 OAuth 令牌過期")
                return {
                    'success': False,
                    'reason': 'OAuth token expired',
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
            elif response.status_code == 404:
                # Not found - position doesn't exist | 未找到 - 倉位不存在
                return {
                    'success': False,
                    'reason': f'Position {deal_id} not found on server',
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Position close failed: {error_msg}")
                
                return {
                    'success': False,
                    'reason': error_msg,
                    'source': 'OAUTH_API',
                    'http_status': response.status_code
                }
                
        except Exception as e:
            logger.error(f"OAuth position close failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'source': 'OAUTH_API',
                'http_status': None
            }

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