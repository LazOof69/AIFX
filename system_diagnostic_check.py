#!/usr/bin/env python3
"""
AIFX System Diagnostic Check | AIFX系統診斷檢查
Comprehensive system health and dependency verification
全面的系統健康狀況和依賴性驗證
"""

import sys
import os
import importlib
import traceback
import json
from pathlib import Path
from typing import Dict, List, Any

def test_core_modules():
    """Test core AIFX modules import | 測試AIFX核心模組導入"""
    results = {}

    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src" / "main" / "python"))

    core_modules = [
        # Utils modules
        ("utils.logger", "get_logger, setup_logger"),
        ("utils.config", "Configuration"),
        ("utils.data_loader", "DataLoader"),
        ("utils.data_preprocessor", "DataPreprocessor"),
        ("utils.feature_generator", "FeatureGenerator"),
        ("utils.technical_indicators", "TechnicalIndicators"),

        # Models
        ("models.base_model", "BaseModel, ModelRegistry"),
        ("models.xgboost_model", "XGBoostModel"),
        ("models.random_forest_model", "RandomForestModel"),
        ("models.lstm_model", "LSTMModel"),  # Optional

        # Core modules
        ("core.trading_strategy", "TradingStrategy"),
        ("core.risk_manager", "RiskManager"),
        ("core.signal_combiner", "SignalCombiner"),

        # Trading modules
        ("trading.execution_engine", "ExecutionEngine"),
        ("trading.live_trader", "LiveTrader"),
        ("trading.position_manager", "PositionManager"),

        # Brokers
        ("brokers.ig_markets", "IGMarketsAPI"),

        # Evaluation
        ("evaluation.backtest_engine", "BacktestEngine"),
        ("evaluation.performance_metrics", "PerformanceMetrics"),
    ]

    for module_path, expected_classes in core_modules:
        try:
            module = importlib.import_module(module_path)

            # Check if expected classes exist
            missing_classes = []
            available_classes = []

            for class_name in expected_classes.split(", "):
                if hasattr(module, class_name):
                    available_classes.append(class_name)
                else:
                    missing_classes.append(class_name)

            results[module_path] = {
                "status": "SUCCESS" if not missing_classes else "PARTIAL",
                "available_classes": available_classes,
                "missing_classes": missing_classes,
                "error": None
            }

        except Exception as e:
            results[module_path] = {
                "status": "FAILED",
                "available_classes": [],
                "missing_classes": expected_classes.split(", "),
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    return results

def test_external_dependencies():
    """Test external package dependencies | 測試外部包依賴"""
    dependencies = [
        "pandas", "numpy", "scikit-learn", "xgboost",
        "yfinance", "matplotlib", "seaborn", "requests",
        "yaml", "jsonschema", "trading_ig"
    ]

    results = {}

    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, "__version__", "Unknown")
            results[dep] = {
                "status": "SUCCESS",
                "version": version,
                "error": None
            }
        except Exception as e:
            results[dep] = {
                "status": "FAILED",
                "version": None,
                "error": str(e)
            }

    return results

def check_file_structure():
    """Check project file structure | 檢查項目文件結構"""
    project_root = Path(__file__).parent

    required_files = [
        "src/main/python/utils/__init__.py",
        "src/main/python/models/__init__.py",
        "src/main/python/core/__init__.py",
        "src/main/python/trading/__init__.py",
        "src/main/python/brokers/__init__.py",
        "src/main/python/evaluation/__init__.py",
        "requirements.txt",
        "CLAUDE.md"
    ]

    results = {}

    for file_path in required_files:
        full_path = project_root / file_path
        results[file_path] = {
            "exists": full_path.exists(),
            "size": full_path.stat().st_size if full_path.exists() else 0
        }

    return results

def run_diagnostic():
    """Run complete diagnostic check | 運行完整診斷檢查"""
    print("🔍 AIFX System Diagnostic Check Started | AIFX系統診斷檢查開始")
    print("=" * 60)

    # Test core modules
    print("\n📦 Testing Core Modules | 測試核心模組")
    print("-" * 40)
    core_results = test_core_modules()

    success_count = sum(1 for r in core_results.values() if r["status"] == "SUCCESS")
    partial_count = sum(1 for r in core_results.values() if r["status"] == "PARTIAL")
    failed_count = sum(1 for r in core_results.values() if r["status"] == "FAILED")

    print(f"✅ SUCCESS: {success_count}")
    print(f"⚠️ PARTIAL: {partial_count}")
    print(f"❌ FAILED: {failed_count}")

    # Test external dependencies
    print("\n🔗 Testing External Dependencies | 測試外部依賴")
    print("-" * 40)
    dep_results = test_external_dependencies()

    dep_success = sum(1 for r in dep_results.values() if r["status"] == "SUCCESS")
    dep_failed = sum(1 for r in dep_results.values() if r["status"] == "FAILED")

    print(f"✅ SUCCESS: {dep_success}")
    print(f"❌ FAILED: {dep_failed}")

    # Check file structure
    print("\n📁 Checking File Structure | 檢查文件結構")
    print("-" * 40)
    file_results = check_file_structure()

    file_success = sum(1 for r in file_results.values() if r["exists"])
    file_missing = sum(1 for r in file_results.values() if not r["exists"])

    print(f"✅ EXISTS: {file_success}")
    print(f"❌ MISSING: {file_missing}")

    # Detailed results
    print("\n📋 Detailed Results | 詳細結果")
    print("=" * 60)

    # Core modules details
    print("\n🔧 Core Modules Issues | 核心模組問題:")
    for module, result in core_results.items():
        if result["status"] != "SUCCESS":
            print(f"❌ {module}: {result['error']}")
            if result["missing_classes"]:
                print(f"   Missing: {', '.join(result['missing_classes'])}")

    # Dependencies details
    print("\n📦 Dependencies Issues | 依賴問題:")
    for dep, result in dep_results.items():
        if result["status"] != "SUCCESS":
            print(f"❌ {dep}: {result['error']}")

    # File structure details
    print("\n📁 Missing Files | 缺失文件:")
    for file_path, result in file_results.items():
        if not result["exists"]:
            print(f"❌ {file_path}")

    # Save detailed results to JSON
    detailed_results = {
        "timestamp": str(datetime.now()),
        "core_modules": core_results,
        "dependencies": dep_results,
        "file_structure": file_results,
        "summary": {
            "core_modules": {"success": success_count, "partial": partial_count, "failed": failed_count},
            "dependencies": {"success": dep_success, "failed": dep_failed},
            "files": {"exists": file_success, "missing": file_missing}
        }
    }

    with open("system_diagnostic_report.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed report saved to: system_diagnostic_report.json")

    return detailed_results

if __name__ == "__main__":
    from datetime import datetime
    run_diagnostic()