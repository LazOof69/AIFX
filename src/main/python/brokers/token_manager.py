#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IG Markets Token Manager | IG Markets 令牌管理器
==============================================

This module handles OAuth token lifecycle management for IG Markets API.
Automatically refreshes access tokens and manages token persistence.
此模組處理 IG Markets API 的 OAuth 令牌生命週期管理。
自動刷新訪問令牌並管理令牌持久性。

Features | 功能:
- Automatic token refresh | 自動令牌刷新
- Token expiry monitoring | 令牌到期監控
- Secure token storage | 安全令牌存儲
- Thread-safe operations | 線程安全操作
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import requests

# Configure logging | 配置日誌
logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    """
    OAuth Token Information | OAuth 令牌信息
    """
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 60  # seconds
    scope: str = "profile"
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    @property
    def expires_at(self) -> float:
        """Token expiry timestamp | 令牌到期時間戳"""
        return self.created_at + self.expires_in

    @property
    def is_expired(self) -> bool:
        """Check if token is expired | 檢查令牌是否已過期"""
        return time.time() >= (self.expires_at - 5)  # 5 second buffer

    @property
    def time_until_expiry(self) -> int:
        """Seconds until token expires | 令牌到期前的秒數"""
        return max(0, int(self.expires_at - time.time()))

class IGTokenManager:
    """
    IG Markets OAuth Token Manager | IG Markets OAuth 令牌管理器

    Handles automatic token refresh and persistence for IG Markets API authentication.
    處理 IG Markets API 認證的自動令牌刷新和持久性。

    Key Features | 主要功能:
    - Automatic token refresh before expiry | 到期前自動刷新令牌
    - Thread-safe token operations | 線程安全的令牌操作
    - Token persistence to file | 令牌持久化到文件
    - Retry logic for refresh failures | 刷新失敗的重試邏輯
    """

    def __init__(self, config_path: str, demo: bool = True):
        """
        Initialize Token Manager | 初始化令牌管理器

        Args:
            config_path: Path to credentials configuration | 憑據配置路徑
            demo: Use demo environment | 使用演示環境
        """
        self.config_path = config_path
        self.demo = demo
        self.token_info: Optional[TokenInfo] = None
        self.config = self._load_config()

        # Thread safety | 線程安全
        self._lock = threading.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # API endpoints | API 端點
        self.base_url = "https://demo-api.ig.com" if demo else "https://api.ig.com"
        self.refresh_url = f"{self.base_url}/gateway/deal/session/refresh-token"

        # Session for HTTP requests | HTTP 請求會話
        self.session = requests.Session()

        logger.info(f"Token Manager initialized for {'demo' if demo else 'live'} environment")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file | 載入配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _save_config(self):
        """Save updated configuration | 保存更新的配置"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            logger.debug("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def initialize_tokens(self, oauth_token_data: Dict[str, Any]) -> bool:
        """
        Initialize tokens from OAuth response | 從 OAuth 響應初始化令牌

        Args:
            oauth_token_data: OAuth token response data

        Returns:
            bool: Success status
        """
        try:
            with self._lock:
                self.token_info = TokenInfo(
                    access_token=oauth_token_data['access_token'],
                    refresh_token=oauth_token_data['refresh_token'],
                    token_type=oauth_token_data.get('token_type', 'Bearer'),
                    expires_in=int(oauth_token_data.get('expires_in', 60)),
                    scope=oauth_token_data.get('scope', 'profile')
                )

                # Update config with new tokens | 使用新令牌更新配置
                account_type = 'demo' if self.demo else 'live'
                self.config['ig_markets'][account_type]['oauthToken'] = {
                    'access_token': self.token_info.access_token,
                    'refresh_token': self.token_info.refresh_token,
                    'token_type': self.token_info.token_type,
                    'expires_in': str(self.token_info.expires_in),
                    'scope': self.token_info.scope
                }

                self._save_config()

                logger.info(f"Tokens initialized. Expires in {self.token_info.expires_in} seconds")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize tokens: {e}")
            return False

    def get_valid_access_token(self) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary | 獲取有效的訪問令牌，必要時刷新

        Returns:
            Optional[str]: Valid access token or None if failed
        """
        with self._lock:
            if not self.token_info:
                logger.warning("No token info available")
                return None

            if self.token_info.is_expired:
                logger.info("Access token expired, attempting refresh")
                if not self._refresh_token_sync():
                    logger.error("Token refresh failed")
                    return None

            return self.token_info.access_token

    def _refresh_token_sync(self) -> bool:
        """
        Synchronously refresh the access token | 同步刷新訪問令牌

        Returns:
            bool: Success status
        """
        try:
            if not self.token_info or not self.token_info.refresh_token:
                logger.error("No refresh token available")
                return False

            # Prepare refresh request | 準備刷新請求
            headers = {
                'Content-Type': 'application/json; charset=UTF-8',
                'Accept': 'application/json; charset=UTF-8',
                'Version': '1'
            }

            # Get API key if available | 如果可用，獲取 API 密鑰
            account_type = 'demo' if self.demo else 'live'
            api_key = self.config.get('ig_markets', {}).get(account_type, {}).get('api_key')
            if api_key:
                headers['X-IG-API-KEY'] = api_key

            data = {
                'refresh_token': self.token_info.refresh_token
            }

            logger.info("Sending token refresh request")
            response = self.session.post(
                self.refresh_url,
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                token_data = response.json()

                # Update token info | 更新令牌信息
                self.token_info = TokenInfo(
                    access_token=token_data['oauthToken']['access_token'],
                    refresh_token=token_data['oauthToken']['refresh_token'],
                    token_type=token_data['oauthToken'].get('token_type', 'Bearer'),
                    expires_in=int(token_data['oauthToken'].get('expires_in', 60)),
                    scope=token_data['oauthToken'].get('scope', 'profile')
                )

                # Update config | 更新配置
                self.config['ig_markets'][account_type]['oauthToken'] = {
                    'access_token': self.token_info.access_token,
                    'refresh_token': self.token_info.refresh_token,
                    'token_type': self.token_info.token_type,
                    'expires_in': str(self.token_info.expires_in),
                    'scope': self.token_info.scope
                }

                self._save_config()

                logger.info(f"Token refreshed successfully. New expiry: {self.token_info.expires_in} seconds")
                return True

            else:
                logger.error(f"Token refresh failed: HTTP {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    async def start_auto_refresh(self):
        """
        Start automatic token refresh monitoring | 啟動自動令牌刷新監控
        """
        if self._refresh_task and not self._refresh_task.done():
            logger.warning("Auto-refresh already running")
            return

        self._shutdown = False
        self._refresh_task = asyncio.create_task(self._auto_refresh_loop())
        logger.info("Auto-refresh monitoring started")

    async def stop_auto_refresh(self):
        """
        Stop automatic token refresh monitoring | 停止自動令牌刷新監控
        """
        self._shutdown = True

        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        logger.info("Auto-refresh monitoring stopped")

    async def _auto_refresh_loop(self):
        """
        Automatic token refresh loop | 自動令牌刷新循環
        """
        while not self._shutdown:
            try:
                if self.token_info:
                    time_until_expiry = self.token_info.time_until_expiry

                    # Refresh when 10 seconds or less remaining | 剩餘10秒或更少時刷新
                    if time_until_expiry <= 10:
                        logger.info("Token expiring soon, refreshing...")

                        with self._lock:
                            success = self._refresh_token_sync()

                        if not success:
                            logger.error("Auto-refresh failed")
                            # Wait a bit before retrying | 稍等片刻後重試
                            await asyncio.sleep(5)
                            continue

                # Check every 5 seconds | 每5秒檢查一次
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                logger.info("Auto-refresh loop cancelled")
                break
            except Exception as e:
                logger.error(f"Auto-refresh loop error: {e}")
                await asyncio.sleep(10)  # Wait longer on errors

    def get_auth_headers(self, include_account_id: bool = True) -> Dict[str, str]:
        """
        Get authentication headers for API requests | 獲取 API 請求的認證標頭

        Args:
            include_account_id: Include IG-ACCOUNT-ID header

        Returns:
            Dict[str, str]: Authentication headers
        """
        access_token = self.get_valid_access_token()
        if not access_token:
            raise Exception("No valid access token available")

        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json; charset=UTF-8',
            'Accept': 'application/json; charset=UTF-8'
        }

        # Add account ID if requested | 如果需要，添加帳戶 ID
        if include_account_id:
            account_type = 'demo' if self.demo else 'live'
            account_id = self.config.get('ig_markets', {}).get(account_type, {}).get('accountId')
            if account_id:
                headers['IG-ACCOUNT-ID'] = account_id

        # Add API key if available | 如果可用，添加 API 密鑰
        account_type = 'demo' if self.demo else 'live'
        api_key = self.config.get('ig_markets', {}).get(account_type, {}).get('api_key')
        if api_key:
            headers['X-IG-API-KEY'] = api_key

        return headers

    def get_token_status(self) -> Dict[str, Any]:
        """
        Get current token status information | 獲取當前令牌狀態信息

        Returns:
            Dict[str, Any]: Token status information
        """
        if not self.token_info:
            return {
                'status': 'no_token',
                'message': 'No token information available'
            }

        return {
            'status': 'expired' if self.token_info.is_expired else 'valid',
            'expires_in': self.token_info.time_until_expiry,
            'expires_at': datetime.fromtimestamp(self.token_info.expires_at).isoformat(),
            'token_type': self.token_info.token_type,
            'scope': self.token_info.scope,
            'created_at': datetime.fromtimestamp(self.token_info.created_at).isoformat()
        }

    def __enter__(self):
        """Context manager entry | 上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit | 上下文管理器出口"""
        if self._refresh_task and not self._refresh_task.done():
            asyncio.create_task(self.stop_auto_refresh())