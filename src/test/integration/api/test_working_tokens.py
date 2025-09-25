#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test IG API with provided working tokens
使用提供的有效令牌測試 IG API
"""

import requests
import json
from datetime import datetime

def test_ig_api_with_tokens():
    """Test IG API with the provided CST and X-SECURITY-TOKEN"""
    print("🎉 Testing IG API with Working Tokens!")
    print("=" * 60)
    
    # Your provided tokens
    cst_token = "6776bcc2291f5ab7ef16ae0dad8331daf1912326e82ccdcbb44637774ff1e6CC01116"
    security_token = "293fa0505aae6745b6a7f38d21b5b63fd2d79d28dc371027f13da690c38359CD01114"
    api_key = "3a0f12d07fe51ab5f4f1835ae037e1f5e876726e"
    
    # Common headers for authenticated requests
    headers = {
        'Accept': 'application/json; charset=UTF-8',
        'Content-Type': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': api_key,
        'CST': cst_token,
        'X-SECURITY-TOKEN': security_token
    }
    
    tests = [
        {
            'name': 'Get Account Info',
            'url': 'https://demo-api.ig.com/gateway/deal/accounts',
            'method': 'GET',
            'version': '1'
        },
        {
            'name': 'Get Market Info - EUR/USD',
            'url': 'https://demo-api.ig.com/gateway/deal/markets/CS.D.EURUSD.CFD.IP',
            'method': 'GET', 
            'version': '3'
        },
        {
            'name': 'Get Market Info - USD/JPY',
            'url': 'https://demo-api.ig.com/gateway/deal/markets/CS.D.USDJPY.CFD.IP',
            'method': 'GET',
            'version': '3'
        },
        {
            'name': 'Get Positions',
            'url': 'https://demo-api.ig.com/gateway/deal/positions',
            'method': 'GET',
            'version': '2'
        },
        {
            'name': 'Get Historical Prices - EUR/USD',
            'url': 'https://demo-api.ig.com/gateway/deal/prices/CS.D.EURUSD.CFD.IP/HOUR/10',
            'method': 'GET',
            'version': '3'
        }
    ]
    
    results = {}
    
    for test in tests:
        print(f"\n🔍 Testing: {test['name']}")
        
        test_headers = headers.copy()
        test_headers['Version'] = test['version']
        
        try:
            if test['method'] == 'GET':
                response = requests.get(test['url'], headers=test_headers, timeout=10)
            else:
                response = requests.post(test['url'], headers=test_headers, timeout=10)
            
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS: {test['name']}")
                
                # Show relevant data
                if 'accounts' in test['name'].lower():
                    for account in data.get('accounts', []):
                        print(f"   💰 Account: {account.get('accountId')} - {account.get('accountName')}")
                        print(f"   💵 Balance: {account.get('balance', {}).get('balance', 'N/A')} {account.get('currency')}")
                
                elif 'market' in test['name'].lower():
                    instrument = data.get('instrument', {})
                    snapshot = data.get('snapshot', {})
                    print(f"   📈 Market: {instrument.get('name', 'N/A')}")
                    print(f"   💰 Bid: {snapshot.get('bid', 'N/A')}")
                    print(f"   💰 Ask: {snapshot.get('offer', 'N/A')}")
                    print(f"   📊 Status: {instrument.get('marketStatus', 'N/A')}")
                
                elif 'positions' in test['name'].lower():
                    positions = data.get('positions', [])
                    print(f"   📊 Open Positions: {len(positions)}")
                    for pos in positions[:3]:  # Show first 3
                        print(f"   🎯 {pos.get('market', {}).get('instrumentName', 'N/A')}: {pos.get('position', {}).get('direction', 'N/A')}")
                
                elif 'historical' in test['name'].lower():
                    prices = data.get('prices', [])
                    print(f"   📊 Price Records: {len(prices)}")
                    if prices:
                        latest = prices[-1]
                        print(f"   📈 Latest: O:{latest.get('openPrice', {}).get('bid', 'N/A')} H:{latest.get('highPrice', {}).get('bid', 'N/A')} L:{latest.get('lowPrice', {}).get('bid', 'N/A')} C:{latest.get('closePrice', {}).get('bid', 'N/A')}")
                
                results[test['name']] = True
                
            elif response.status_code == 401:
                print(f"❌ FAILED: {test['name']} - Authentication failed (tokens may be expired)")
                results[test['name']] = False
                
            else:
                print(f"❌ FAILED: {test['name']} - HTTP {response.status_code}")
                try:
                    error = response.json()
                    print(f"   Error: {error}")
                except:
                    print(f"   Response: {response.text}")
                results[test['name']] = False
                
        except Exception as e:
            print(f"❌ ERROR: {test['name']} - {e}")
            results[test['name']] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 RESULTS SUMMARY:")
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count > 0:
        print("\n🎉 BREAKTHROUGH! IG API is working!")
        print("🚀 AIFX can now integrate with your IG account")
        print("💡 Next step: Update AIFX configuration with these tokens")
    else:
        print("\n⚠️ All tests failed - tokens may be expired")
        print("💡 Try getting fresh tokens from IG platform")
    
    return success_count > 0

if __name__ == "__main__":
    test_ig_api_with_tokens()