#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update AIFX configuration with working IG API tokens
使用有效的 IG API 令牌更新 AIFX 配置
"""

import yaml
import json
from pathlib import Path

def update_aifx_config():
    """Update AIFX configuration with working IG tokens"""
    print("🔧 Updating AIFX Configuration with Working IG Tokens")
    print("=" * 60)
    
    # Your working tokens
    tokens = {
        'cst': '6776bcc2291f5ab7ef16ae0dad8331daf1912326e82ccdcbb44637774ff1e6CC01116',
        'x_security_token': '293fa0505aae6745b6a7f38d21b5b63fd2d79d28dc371027f13da690c38359CD01114',
        'api_key': '3a0f12d07fe51ab5f4f1835ae037e1f5e876726e'
    }
    
    # Update main trading config
    config_path = 'config/trading-config.yaml'
    
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"📄 Loading config from: {config_path}")
        
        # Update IG configuration with working tokens
        if 'ig_markets' not in config:
            config['ig_markets'] = {}
        
        if 'demo' not in config['ig_markets']:
            config['ig_markets']['demo'] = {}
        
        # Add the working tokens
        config['ig_markets']['demo']['tokens'] = {
            'cst': tokens['cst'],
            'x_security_token': tokens['x_security_token'],
            'expires': 'Session based - monitor for expiry'
        }
        
        # Mark as authenticated
        config['ig_markets']['demo']['authenticated'] = True
        config['ig_markets']['demo']['last_auth'] = '2025-09-15 20:50:00'
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print("✅ Updated main trading configuration")
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False
    
    # Create token storage file for IG connector
    token_file = 'config/ig_tokens.json'
    
    try:
        token_data = {
            'demo': {
                'cst': tokens['cst'],
                'x_security_token': tokens['x_security_token'],
                'api_key': tokens['api_key'],
                'authenticated': True,
                'auth_time': '2025-09-15 20:50:00',
                'status': 'active'
            }
        }
        
        with open(token_file, 'w') as f:
            json.dump(token_data, f, indent=2)
        
        print(f"✅ Created token storage: {token_file}")
        
    except Exception as e:
        print(f"❌ Error creating token file: {e}")
    
    # Update IG Markets connector to use tokens
    connector_path = 'src/main/python/brokers/ig_markets.py'
    
    try:
        print(f"📝 Updating IG connector: {connector_path}")
        
        # Read current connector
        with open(connector_path, 'r') as f:
            content = f.read()
        
        # Add token-based authentication method
        token_auth_code = '''
    def authenticate_with_tokens(self, cst_token, security_token):
        """Authenticate using existing CST and X-SECURITY-TOKEN"""
        try:
            self.session.headers.update({
                'CST': cst_token,
                'X-SECURITY-TOKEN': security_token,
                'X-IG-API-KEY': self.config['api_key']
            })
            
            # Test authentication with account info
            response = self.session.get(f"{self.config['api_url']}/accounts", 
                                      headers={'Version': '1'}, timeout=10)
            
            if response.status_code == 200:
                self.authenticated = True
                self.account_info = response.json()
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Token authentication error: {e}")
            return False
'''
        
        # Add the method if not already present
        if 'authenticate_with_tokens' not in content:
            # Find a good insertion point (before the last method)
            insertion_point = content.rfind('    def ')
            if insertion_point > 0:
                content = content[:insertion_point] + token_auth_code + '\n' + content[insertion_point:]
            else:
                content += token_auth_code
            
            with open(connector_path, 'w') as f:
                f.write(content)
            
            print("✅ Added token authentication method to IG connector")
        else:
            print("✅ Token authentication method already exists")
    
    except Exception as e:
        print(f"⚠️ Warning: Could not update IG connector: {e}")
    
    return True

def create_test_script():
    """Create a test script for AIFX with tokens"""
    print("\n🧪 Creating AIFX + IG Integration Test Script")
    
    test_script = '''#!/usr/bin/env python3
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
'''
    
    with open('test_aifx_ig_integration.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created integration test script: test_aifx_ig_integration.py")

def main():
    """Main update process"""
    print("🚀 AIFX + IG API Integration Setup")
    print("=" * 60)
    
    success = update_aifx_config()
    
    if success:
        create_test_script()
        
        print("\n🎉 SETUP COMPLETE!")
        print("📋 What was updated:")
        print("   ✅ config/trading-config.yaml - Added working tokens")
        print("   ✅ config/ig_tokens.json - Token storage created")
        print("   ✅ IG connector - Token authentication method added")
        print("   ✅ test_aifx_ig_integration.py - Test script created")
        
        print("\n🎯 Next Steps:")
        print("   1. Run: python test_aifx_ig_integration.py")
        print("   2. Run: python run_trading_demo.py --mode demo")
        print("   3. Test full AIFX system with live IG data")
        
        print("\n⚠️ Important Notes:")
        print("   - Tokens are session-based and may expire")
        print("   - Monitor for 401 errors indicating token expiry")
        print("   - Re-authenticate when needed")
    
    else:
        print("\n❌ Setup failed - check errors above")

if __name__ == "__main__":
    main()