#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test AIFX with working IG API tokens
使用有效IG API令牌測試AIFX
"""

import sys
import os
sys.path.append('src/main/python')

from brokers.ig_markets import IGMarketsConnector
import json

def test_aifx_ig_integration():
    """Test AIFX integration with IG using working tokens"""
    print("🎯 Testing AIFX + IG Markets Integration")
    print("=" * 50)
    
    # Load tokens
    try:
        with open('config/ig_tokens.json', 'r') as f:
            tokens = json.load(f)
        
        demo_tokens = tokens['demo']
        print("✅ Tokens loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load tokens: {e}")
        return False
    
    # Test IG connector
    try:
        # Create IG connector
        connector = IGMarketsConnector('config/trading-config.yaml')
        
        # Authenticate with tokens
        success = connector.authenticate_with_tokens(
            demo_tokens['cst'],
            demo_tokens['x_security_token']
        )
        
        if success:
            print("✅ IG API authentication successful")
            print("🎉 AIFX can now trade with your IG account!")
            
            # Test basic operations
            if hasattr(connector, 'account_info'):
                print(f"📊 Account info loaded: {len(connector.account_info.get('accounts', []))} accounts")
            
            return True
        else:
            print("❌ IG API authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

if __name__ == "__main__":
    test_aifx_ig_integration()
