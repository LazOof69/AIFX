"""
AIFX Environment Verification Script | AIFX環境驗證腳本

This script quickly checks if your environment is ready for AIFX development.
此腳本快速檢查您的環境是否準備好進行AIFX開發。

Usage | 使用方法:
    python check_environment.py
    
    # With verbose output | 詳細輸出
    python check_environment.py --verbose
    
    # Install missing packages | 安裝缺失的包
    python check_environment.py --install
"""

import sys
import os
import subprocess
import pkg_resources
import argparse
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_python_version():
    """Check Python version | 檢查Python版本"""
    version = sys.version_info
    print(f"Python Version | Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print(f"{Colors.GREEN}✅ Python version is compatible{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ Python 3.8+ required, found {version.major}.{version.minor}{Colors.END}")
        return False

def check_required_packages():
    """Check required packages | 檢查必需包"""
    print(f"\n{Colors.BOLD}Checking Required Packages | 檢查必需包{Colors.END}")
    
    # Core packages | 核心包
    core_packages = [
        ('pandas', '>=1.5.0'),
        ('numpy', '>=1.21.0'),
        ('scipy', '>=1.7.0'),
        ('matplotlib', '>=3.5.0'),
        ('scikit-learn', '>=1.0.0'),
        ('yfinance', '>=0.2.0'),
    ]
    
    # ML packages | ML包
    ml_packages = [
        ('xgboost', '>=1.6.0'),
        ('lightgbm', '>=3.3.0'),
    ]
    
    # Optional packages | 可選包
    optional_packages = [
        ('tensorflow', '>=2.10.0'),
        ('torch', '>=1.12.0'),
        ('plotly', '>=5.10.0'),
        ('jupyter', '>=1.0.0'),
        ('pytest', '>=7.0.0'),
        ('structlog', '>=22.0.0'),
    ]
    
    def check_package_list(packages, category_name, required=True):
        """Check a list of packages | 檢查包列表"""
        print(f"\n{category_name}:")
        missing_packages = []
        
        for package, version in packages:
            try:
                pkg_resources.require(f"{package}{version}")
                installed_version = pkg_resources.get_distribution(package).version
                print(f"  {Colors.GREEN}✅{Colors.END} {package} ({installed_version})")
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
                if required:
                    print(f"  {Colors.RED}❌{Colors.END} {package} - Missing or incompatible")
                    missing_packages.append(package)
                else:
                    print(f"  {Colors.YELLOW}⚠️{Colors.END} {package} - Optional, not installed")
        
        return missing_packages
    
    # Check all package categories | 檢查所有包類別
    missing_core = check_package_list(core_packages, "Core Packages | 核心包", True)
    missing_ml = check_package_list(ml_packages, "ML Packages | ML包", True)
    missing_optional = check_package_list(optional_packages, "Optional Packages | 可選包", False)
    
    total_missing = missing_core + missing_ml
    
    if not total_missing:
        print(f"\n{Colors.GREEN}✅ All required packages are installed{Colors.END}")
        return True, []
    else:
        print(f"\n{Colors.RED}❌ Missing {len(total_missing)} required packages{Colors.END}")
        return False, total_missing

def check_project_structure():
    """Check project structure | 檢查項目結構"""
    print(f"\n{Colors.BOLD}Checking Project Structure | 檢查項目結構{Colors.END}")
    
    required_dirs = [
        'src/main/python',
        'src/main/resources',
        'src/test',
        'data',
        'models',
        'notebooks',
        'logs',
        'output'
    ]
    
    required_files = [
        'requirements.txt',
        'pytest.ini',
        'CLAUDE.md',
        'README.md',
        'src/main/python/utils/config.py',
        'src/main/python/utils/logger.py',
        'src/main/python/utils/data_loader.py',
        'src/main/python/utils/data_preprocessor.py',
        'src/main/python/utils/technical_indicators.py',
    ]
    
    all_good = True
    
    # Check directories | 檢查目錄
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  {Colors.GREEN}✅{Colors.END} Directory: {dir_path}")
        else:
            print(f"  {Colors.RED}❌{Colors.END} Directory: {dir_path}")
            all_good = False
    
    # Check files | 檢查文件
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  {Colors.GREEN}✅{Colors.END} File: {file_path}")
        else:
            print(f"  {Colors.RED}❌{Colors.END} File: {file_path}")
            all_good = False
    
    if all_good:
        print(f"\n{Colors.GREEN}✅ Project structure is complete{Colors.END}")
    else:
        print(f"\n{Colors.RED}❌ Project structure has missing components{Colors.END}")
    
    return all_good

def check_system_capabilities():
    """Check system capabilities | 檢查系統能力"""
    print(f"\n{Colors.BOLD}Checking System Capabilities | 檢查系統能力{Colors.END}")
    
    # Check available memory | 檢查可用內存
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"  Available RAM | 可用內存: {memory_gb:.1f} GB")
        
        if memory_gb >= 4:
            print(f"  {Colors.GREEN}✅{Colors.END} Sufficient memory for ML operations")
        else:
            print(f"  {Colors.YELLOW}⚠️{Colors.END} Limited memory may affect ML performance")
    except ImportError:
        print(f"  {Colors.YELLOW}⚠️{Colors.END} Cannot check memory (psutil not installed)")
    
    # Check CPU cores | 檢查CPU核心
    cpu_count = os.cpu_count()
    print(f"  CPU Cores | CPU核心: {cpu_count}")
    
    if cpu_count >= 4:
        print(f"  {Colors.GREEN}✅{Colors.END} Good CPU count for parallel processing")
    else:
        print(f"  {Colors.YELLOW}⚠️{Colors.END} Limited cores may slow down processing")
    
    return True

def install_packages(missing_packages):
    """Install missing packages | 安裝缺失的包"""
    if not missing_packages:
        print(f"{Colors.GREEN}No packages to install{Colors.END}")
        return True
    
    print(f"\n{Colors.BOLD}Installing Missing Packages | 安裝缺失包{Colors.END}")
    print(f"Packages to install | 要安裝的包: {', '.join(missing_packages)}")
    
    try:
        # First try to install from requirements.txt if it exists
        # 首先嘗試從requirements.txt安裝（如果存在）
        if Path('requirements.txt').exists():
            print(f"\nInstalling from requirements.txt...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ Successfully installed from requirements.txt{Colors.END}")
                return True
            else:
                print(f"{Colors.YELLOW}⚠️ requirements.txt installation failed, trying individual packages{Colors.END}")
        
        # Install individual packages | 安裝單個包
        for package in missing_packages:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  {Colors.GREEN}✅{Colors.END} {package}")
            else:
                print(f"  {Colors.RED}❌{Colors.END} {package} - {result.stderr}")
                return False
        
        print(f"{Colors.GREEN}✅ All packages installed successfully{Colors.END}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Installation failed: {str(e)}{Colors.END}")
        return False

def main():
    """Main environment check function | 主要環境檢查函數"""
    parser = argparse.ArgumentParser(description='AIFX Environment Checker')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--install', '-i', action='store_true', help='Install missing packages')
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 AIFX Environment Verification | AIFX環境驗證{Colors.END}")
    print("="*60)
    
    checks_passed = 0
    total_checks = 0
    
    # Check Python version | 檢查Python版本
    total_checks += 1
    if check_python_version():
        checks_passed += 1
    
    # Check packages | 檢查包
    total_checks += 1
    packages_ok, missing_packages = check_required_packages()
    if packages_ok:
        checks_passed += 1
    elif args.install:
        print(f"\n{Colors.CYAN}Attempting to install missing packages...{Colors.END}")
        if install_packages(missing_packages):
            # Re-check after installation | 安裝後重新檢查
            packages_ok, _ = check_required_packages()
            if packages_ok:
                checks_passed += 1
    
    # Check project structure | 檢查項目結構
    total_checks += 1
    if check_project_structure():
        checks_passed += 1
    
    # Check system capabilities | 檢查系統能力
    total_checks += 1
    if check_system_capabilities():
        checks_passed += 1
    
    # Final report | 最終報告
    print(f"\n{Colors.BOLD}Environment Check Results | 環境檢查結果{Colors.END}")
    print("="*60)
    
    success_rate = (checks_passed / total_checks) * 100
    print(f"Passed: {Colors.GREEN}{checks_passed}/{total_checks}{Colors.END} ({success_rate:.0f}%)")
    
    if checks_passed == total_checks:
        print(f"\n{Colors.GREEN}✅ Environment is ready for AIFX development!{Colors.END}")
        print(f"{Colors.GREEN}✅ 環境已準備好進行AIFX開發！{Colors.END}")
        print(f"\nNext step | 下一步: Run the comprehensive test")
        print(f"python test_phase1_complete.py")
        return True
    elif checks_passed >= total_checks * 0.75:
        print(f"\n{Colors.YELLOW}⚠️ Environment is mostly ready with minor issues{Colors.END}")
        print(f"{Colors.YELLOW}⚠️ 環境基本準備就緒，有輕微問題{Colors.END}")
        if missing_packages and not args.install:
            print(f"\nTo install missing packages | 安裝缺失包:")
            print(f"python check_environment.py --install")
        return True
    else:
        print(f"\n{Colors.RED}❌ Environment needs attention before development{Colors.END}")
        print(f"{Colors.RED}❌ 開發前環境需要處理{Colors.END}")
        
        if missing_packages and not args.install:
            print(f"\nTo install missing packages | 安裝缺失包:")
            print(f"python check_environment.py --install")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Check interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error during environment check: {str(e)}{Colors.END}")
        sys.exit(1)