#!/usr/bin/env python3
"""
AIFX Web System Test Suite
AIFX 網頁系統測試套件

Test suite for validating the simplified web trading signals system.
用於驗證簡化網頁交易信號系統的測試套件。
"""

import asyncio
import json
import logging
import requests
import time
import websocket
from datetime import datetime
from typing import Dict, Any, List
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSystemTester:
    """
    Comprehensive test suite for AIFX Web System
    AIFX網頁系統的綜合測試套件
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.websocket_messages = []
        self.websocket_connected = False
        self.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'details': []
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests"""
        print("🧪 AIFX Web System Test Suite")
        print("=" * 50)
        print()

        # Test API endpoints
        self.test_health_endpoint()
        self.test_signals_endpoint()
        self.test_web_interface()

        # Test WebSocket connection
        self.test_websocket_connection()

        # Test signal generation with demo data
        self.test_signal_generation()

        # Test system under load
        self.test_load_handling()

        # Print summary
        self.print_test_summary()

        return self.test_results

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        test_name = "Health Endpoint Test"
        self._start_test(test_name)

        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)

            if response.status_code == 200:
                data = response.json()

                required_fields = ['status', 'timestamp', 'is_monitoring']
                missing_fields = [field for field in required_fields if field not in data]

                if not missing_fields:
                    self._pass_test(test_name, f"Health endpoint responding correctly")
                else:
                    self._fail_test(test_name, f"Missing required fields: {missing_fields}")
            else:
                self._fail_test(test_name, f"Unexpected status code: {response.status_code}")

        except Exception as e:
            self._fail_test(test_name, f"Health endpoint error: {e}")

    def test_signals_endpoint(self):
        """Test the signals API endpoint"""
        test_name = "Signals API Test"
        self._start_test(test_name)

        try:
            response = requests.get(f"{self.base_url}/api/signals", timeout=15)

            if response.status_code == 200:
                data = response.json()

                required_fields = ['status', 'signals', 'stats']
                missing_fields = [field for field in required_fields if field not in data]

                if not missing_fields:
                    signals = data.get('signals', {})
                    if 'EURUSD' in signals or 'USDJPY' in signals:
                        self._pass_test(test_name, f"Signals API working with {len(signals)} currency pairs")
                    else:
                        self._fail_test(test_name, "No expected currency pairs found in signals")
                else:
                    self._fail_test(test_name, f"Missing required fields: {missing_fields}")
            else:
                self._fail_test(test_name, f"Unexpected status code: {response.status_code}")

        except Exception as e:
            self._fail_test(test_name, f"Signals API error: {e}")

    def test_web_interface(self):
        """Test the main web interface"""
        test_name = "Web Interface Test"
        self._start_test(test_name)

        try:
            response = requests.get(f"{self.base_url}/", timeout=10)

            if response.status_code == 200:
                content = response.text

                # Check for key HTML elements
                key_elements = [
                    'AIFX Trading Signals',
                    'connection-status',
                    'signals-grid',
                    'WebSocket'
                ]

                missing_elements = []
                for element in key_elements:
                    if element not in content:
                        missing_elements.append(element)

                if not missing_elements:
                    self._pass_test(test_name, "Web interface HTML loaded successfully")
                else:
                    self._fail_test(test_name, f"Missing key elements: {missing_elements}")
            else:
                self._fail_test(test_name, f"Unexpected status code: {response.status_code}")

        except Exception as e:
            self._fail_test(test_name, f"Web interface error: {e}")

    def test_websocket_connection(self):
        """Test WebSocket real-time connection"""
        test_name = "WebSocket Connection Test"
        self._start_test(test_name)

        try:
            ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/signals'

            def on_message(ws, message):
                self.websocket_messages.append(json.loads(message))
                logger.info(f"WebSocket message received: {message[:100]}...")

            def on_open(ws):
                self.websocket_connected = True
                logger.info("WebSocket connection established")

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")

            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_open=on_open,
                on_error=on_error,
                on_close=on_close
            )

            # Run WebSocket in background thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()

            # Wait for connection
            time.sleep(3)

            if self.websocket_connected:
                # Send test message and wait for response
                ws.send("ping")
                time.sleep(2)

                ws.close()
                self._pass_test(test_name, "WebSocket connection established successfully")
            else:
                self._fail_test(test_name, "WebSocket connection failed to establish")

        except Exception as e:
            self._fail_test(test_name, f"WebSocket test error: {e}")

    def test_signal_generation(self):
        """Test signal generation with demo data"""
        test_name = "Signal Generation Test"
        self._start_test(test_name)

        try:
            # Wait for system to generate signals
            logger.info("Waiting for signal generation...")
            time.sleep(10)

            # Check signals multiple times to see if they update
            signals_snapshots = []
            for i in range(3):
                response = requests.get(f"{self.base_url}/api/signals", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    signals_snapshots.append(data.get('signals', {}))
                    time.sleep(5)

            # Analyze signal generation
            if len(signals_snapshots) >= 3:
                # Check if signals are being generated
                has_signals = any(len(snapshot) > 0 for snapshot in signals_snapshots)

                if has_signals:
                    # Check for signal variety
                    all_actions = set()
                    for snapshot in signals_snapshots:
                        for symbol_data in snapshot.values():
                            if 'action' in symbol_data:
                                all_actions.add(symbol_data['action'])

                    self._pass_test(test_name, f"Signal generation working - {len(all_actions)} unique actions detected")
                else:
                    self._fail_test(test_name, "No signals generated during test period")
            else:
                self._fail_test(test_name, "Insufficient signal snapshots collected")

        except Exception as e:
            self._fail_test(test_name, f"Signal generation test error: {e}")

    def test_load_handling(self):
        """Test system performance under load"""
        test_name = "Load Handling Test"
        self._start_test(test_name)

        try:
            # Simulate multiple concurrent requests
            import concurrent.futures

            def make_request(endpoint):
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    return response.status_code == 200
                except:
                    return False

            endpoints = ['/api/health', '/api/signals', '/']
            concurrent_requests = 10

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit multiple requests for each endpoint
                futures = []
                for endpoint in endpoints:
                    for _ in range(concurrent_requests):
                        futures.append(executor.submit(make_request, endpoint))

                # Collect results
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            success_rate = sum(results) / len(results) * 100

            if success_rate >= 90:
                self._pass_test(test_name, f"Load test passed - {success_rate:.1f}% success rate")
            else:
                self._fail_test(test_name, f"Load test failed - {success_rate:.1f}% success rate")

        except Exception as e:
            self._fail_test(test_name, f"Load handling test error: {e}")

    def _start_test(self, test_name: str):
        """Start a new test"""
        self.test_results['total_tests'] += 1
        logger.info(f"🧪 Running: {test_name}")

    def _pass_test(self, test_name: str, message: str = ""):
        """Mark test as passed"""
        self.test_results['passed'] += 1
        self.test_results['details'].append({
            'test': test_name,
            'status': 'PASS',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"✅ PASS: {test_name} - {message}")

    def _fail_test(self, test_name: str, message: str = ""):
        """Mark test as failed"""
        self.test_results['failed'] += 1
        self.test_results['details'].append({
            'test': test_name,
            'status': 'FAIL',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        logger.error(f"❌ FAIL: {test_name} - {message}")

    def print_test_summary(self):
        """Print comprehensive test summary"""
        print()
        print("📊 TEST SUMMARY")
        print("=" * 50)

        total = self.test_results['total_tests']
        passed = self.test_results['passed']
        failed = self.test_results['failed']

        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed} ({passed/total*100:.1f}%)")
        print(f"Failed:       {failed} ({failed/total*100:.1f}%)")
        print()

        if failed == 0:
            print("🎉 ALL TESTS PASSED! The system is working correctly.")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED. Please check the issues above.")

        print()
        print("📋 DETAILED RESULTS:")
        print("-" * 30)
        for detail in self.test_results['details']:
            status_icon = "✅" if detail['status'] == 'PASS' else "❌"
            print(f"{status_icon} {detail['test']}: {detail['message']}")

        print()

    def generate_test_report(self, filename: str = None):
        """Generate detailed test report"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"aifx_web_test_report_{timestamp}.json"

        report = {
            'test_run_info': {
                'timestamp': datetime.now().isoformat(),
                'system_url': self.base_url,
                'test_duration': '~60 seconds'
            },
            'summary': {
                'total_tests': self.test_results['total_tests'],
                'passed': self.test_results['passed'],
                'failed': self.test_results['failed'],
                'success_rate': f"{self.test_results['passed']/self.test_results['total_tests']*100:.1f}%"
            },
            'details': self.test_results['details'],
            'websocket_messages': self.websocket_messages
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Test report saved to: {filename}")
        return filename


def main():
    """Main test execution"""
    print("🚀 AIFX Simplified Web Trading Signals - System Test")
    print("=" * 60)
    print()

    # Check if system is running
    try:
        response = requests.get("http://localhost:8080/api/health", timeout=5)
        print("✅ System is running and accessible")
        print()
    except:
        print("❌ System is not accessible at http://localhost:8080")
        print("   Please start the system first using:")
        print("   docker-compose -f docker-compose.web.yml up -d")
        print("   or")
        print("   python -m src.main.python.web_interface")
        return False

    # Run tests
    tester = WebSystemTester()
    results = tester.run_all_tests()

    # Generate report
    report_file = tester.generate_test_report()

    # Return success status
    return results['failed'] == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)