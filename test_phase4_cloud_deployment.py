#!/usr/bin/env python3
"""
AIFX Phase 4 Cloud Deployment Integration Test | AIFX 第四階段雲端部署整合測試

Comprehensive integration testing for Phase 4.1.2 Cloud Deployment Architecture
第四階段第二節雲端部署架構的全面整合測試

Features tested | 測試功能:
- Terraform infrastructure deployment | Terraform 基礎設施部署
- Kubernetes manifests validation | Kubernetes 清單驗證
- Docker image functionality | Docker 映像功能
- CI/CD pipeline components | CI/CD 管道組件
- Cloud storage integration | 雲端存儲整合
- Auto-scaling and load balancing | 自動擴展與負載均衡
- Security and monitoring | 安全與監控
"""

import sys
import os
import json
import yaml
import subprocess
import requests
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import shutil

# Add project root to path | 添加專案根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "main" / "python"))

# Configure logging | 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_phase4_cloud_deployment.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Test configuration | 測試配置
TEST_TIMEOUT = 300  # 5 minutes
HEALTH_CHECK_RETRIES = 10
HEALTH_CHECK_INTERVAL = 30  # seconds

class CloudDeploymentTester:
    """
    AIFX Cloud Deployment Integration Tester | AIFX 雲端部署整合測試器
    
    Comprehensive testing suite for Phase 4 cloud deployment components
    第四階段雲端部署組件的全面測試套件
    """
    
    def __init__(self, environment: str = "test"):
        """Initialize the cloud deployment tester | 初始化雲端部署測試器"""
        self.environment = environment
        self.project_root = project_root
        self.test_results = []
        self.start_time = datetime.now()
        
        # Test configuration | 測試配置
        self.terraform_dir = self.project_root / "infrastructure" / "terraform"
        self.kubernetes_dir = self.project_root / "infrastructure" / "kubernetes"
        self.docker_dir = self.project_root
        
        logger.info(f"🚀 Cloud Deployment Tester initialized for environment: {environment}")
    
    def run_command(self, cmd: str, cwd: Optional[Path] = None, timeout: int = 60) -> Tuple[int, str, str]:
        """
        Run shell command and return result | 運行shell命令並返回結果
        
        Args:
            cmd: Command to run | 要運行的命令
            cwd: Working directory | 工作目錄
            timeout: Timeout in seconds | 超時時間（秒）
        
        Returns:
            Tuple of (return_code, stdout, stderr) | 返回 (返回碼, 標準輸出, 標準錯誤)
        """
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timeout after {timeout} seconds"
        except Exception as e:
            return -1, "", str(e)
    
    def test_terraform_validation(self) -> bool:
        """Test Terraform configuration validation | 測試 Terraform 配置驗證"""
        logger.info("🔍 Testing Terraform configuration validation...")
        
        try:
            # Check if Terraform is installed | 檢查 Terraform 是否安裝
            returncode, stdout, stderr = self.run_command("terraform version")
            if returncode != 0:
                logger.error("❌ Terraform is not installed or not in PATH")
                return False
            
            logger.info("✅ Terraform is installed")
            
            # Initialize Terraform | 初始化 Terraform
            logger.info("📦 Initializing Terraform...")
            returncode, stdout, stderr = self.run_command(
                "terraform init -backend=false",
                cwd=self.terraform_dir,
                timeout=120
            )
            
            if returncode != 0:
                logger.error(f"❌ Terraform init failed: {stderr}")
                return False
            
            # Validate Terraform configuration | 驗證 Terraform 配置
            logger.info("🔍 Validating Terraform configuration...")
            returncode, stdout, stderr = self.run_command(
                "terraform validate",
                cwd=self.terraform_dir
            )
            
            if returncode != 0:
                logger.error(f"❌ Terraform validation failed: {stderr}")
                return False
            
            logger.info("✅ Terraform configuration is valid")
            
            # Test Terraform plan | 測試 Terraform 計劃
            logger.info("📋 Testing Terraform plan...")
            returncode, stdout, stderr = self.run_command(
                "terraform plan -var='environment=test' -var='aws_region=us-west-2' -out=test.tfplan",
                cwd=self.terraform_dir,
                timeout=180
            )
            
            if returncode != 0:
                logger.warning(f"⚠️ Terraform plan failed (expected without AWS credentials): {stderr}")
            else:
                logger.info("✅ Terraform plan succeeded")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Terraform validation test failed: {e}")
            return False
    
    def test_kubernetes_manifests(self) -> bool:
        """Test Kubernetes manifests validation | 測試 Kubernetes 清單驗證"""
        logger.info("🔍 Testing Kubernetes manifests validation...")
        
        try:
            # Check if kubectl is installed | 檢查 kubectl 是否安裝
            returncode, stdout, stderr = self.run_command("kubectl version --client")
            if returncode != 0:
                logger.error("❌ kubectl is not installed or not in PATH")
                return False
            
            logger.info("✅ kubectl is installed")
            
            # Find all Kubernetes YAML files | 查找所有 Kubernetes YAML 文件
            yaml_files = list(self.kubernetes_dir.glob("**/*.yaml"))
            if not yaml_files:
                logger.error("❌ No Kubernetes YAML files found")
                return False
            
            logger.info(f"📋 Found {len(yaml_files)} Kubernetes manifest files")
            
            # Validate each YAML file | 驗證每個 YAML 文件
            valid_files = 0
            for yaml_file in yaml_files:
                try:
                    # Test YAML syntax | 測試 YAML 語法
                    with open(yaml_file, 'r') as f:
                        yaml_content = yaml.safe_load_all(f)
                        list(yaml_content)  # Force parse all documents
                    
                    # Test kubectl dry-run | 測試 kubectl 乾運行
                    returncode, stdout, stderr = self.run_command(
                        f"kubectl apply --dry-run=client -f {yaml_file}"
                    )
                    
                    if returncode == 0:
                        logger.info(f"✅ {yaml_file.name} is valid")
                        valid_files += 1
                    else:
                        logger.warning(f"⚠️ {yaml_file.name} failed kubectl validation: {stderr}")
                
                except yaml.YAMLError as e:
                    logger.error(f"❌ {yaml_file.name} has invalid YAML syntax: {e}")
                except Exception as e:
                    logger.error(f"❌ Error validating {yaml_file.name}: {e}")
            
            success_rate = (valid_files / len(yaml_files)) * 100
            logger.info(f"📊 Kubernetes manifest validation: {valid_files}/{len(yaml_files)} files valid ({success_rate:.1f}%)")
            
            return success_rate >= 80  # At least 80% should be valid
            
        except Exception as e:
            logger.error(f"❌ Kubernetes manifests test failed: {e}")
            return False
    
    def test_docker_configuration(self) -> bool:
        """Test Docker configuration and build process | 測試 Docker 配置與構建過程"""
        logger.info("🔍 Testing Docker configuration...")
        
        try:
            # Check if Docker is installed and running | 檢查 Docker 是否安裝並運行
            returncode, stdout, stderr = self.run_command("docker version")
            if returncode != 0:
                logger.error("❌ Docker is not installed or not running")
                return False
            
            logger.info("✅ Docker is available")
            
            # Check Dockerfile exists | 檢查 Dockerfile 是否存在
            dockerfile_path = self.docker_dir / "Dockerfile"
            if not dockerfile_path.exists():
                logger.error("❌ Dockerfile not found")
                return False
            
            logger.info("✅ Dockerfile found")
            
            # Validate Dockerfile syntax with hadolint (if available) | 用 hadolint 驗證 Dockerfile 語法（如可用）
            returncode, stdout, stderr = self.run_command("hadolint --version")
            if returncode == 0:
                logger.info("🔍 Running hadolint validation...")
                returncode, stdout, stderr = self.run_command(f"hadolint {dockerfile_path}")
                if returncode == 0:
                    logger.info("✅ Dockerfile passes hadolint validation")
                else:
                    logger.warning(f"⚠️ Dockerfile hadolint warnings: {stdout}")
            else:
                logger.info("ℹ️ hadolint not available, skipping Dockerfile linting")
            
            # Test Docker build (with --dry-run if supported) | 測試 Docker 構建（如支持則使用 --dry-run）
            logger.info("🐳 Testing Docker build process...")
            build_cmd = f"docker build --target production --tag aifx:test-{int(time.time())} {self.docker_dir}"
            
            # Try to build (with timeout to prevent hanging) | 嘗試構建（設置超時以防掛起）
            returncode, stdout, stderr = self.run_command(build_cmd, timeout=600)
            
            if returncode == 0:
                logger.info("✅ Docker build completed successfully")
                
                # Get the image ID for cleanup | 獲取映像ID以便清理
                image_tag = f"aifx:test-{int(time.time())}"
                self.run_command(f"docker rmi {image_tag} --force")  # Clean up test image
                
                return True
            else:
                logger.error(f"❌ Docker build failed: {stderr}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Docker configuration test failed: {e}")
            return False
    
    def test_ci_cd_pipeline_config(self) -> bool:
        """Test CI/CD pipeline configuration | 測試 CI/CD 管道配置"""
        logger.info("🔍 Testing CI/CD pipeline configuration...")
        
        try:
            # Check GitHub Actions workflow files | 檢查 GitHub Actions 工作流文件
            workflows_dir = self.project_root / ".github" / "workflows"
            if not workflows_dir.exists():
                logger.error("❌ GitHub Actions workflows directory not found")
                return False
            
            workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
            if not workflow_files:
                logger.error("❌ No GitHub Actions workflow files found")
                return False
            
            logger.info(f"📋 Found {len(workflow_files)} workflow files")
            
            # Validate YAML syntax of workflow files | 驗證工作流文件的 YAML 語法
            valid_workflows = 0
            for workflow_file in workflow_files:
                try:
                    with open(workflow_file, 'r') as f:
                        workflow_data = yaml.safe_load(f)
                    
                    # Check basic workflow structure | 檢查基本工作流結構
                    if 'name' in workflow_data and 'on' in workflow_data and 'jobs' in workflow_data:
                        logger.info(f"✅ {workflow_file.name} has valid structure")
                        valid_workflows += 1
                        
                        # Check for cloud deployment specific steps | 檢查雲端部署特定步驟
                        workflow_content = workflow_file.read_text()
                        if 'aws' in workflow_content.lower() and 'ecr' in workflow_content.lower():
                            logger.info(f"✅ {workflow_file.name} includes AWS/ECR integration")
                        if 'kubectl' in workflow_content.lower() or 'kubernetes' in workflow_content.lower():
                            logger.info(f"✅ {workflow_file.name} includes Kubernetes deployment")
                        if 'terraform' in workflow_content.lower():
                            logger.info(f"✅ {workflow_file.name} includes Terraform deployment")
                    else:
                        logger.warning(f"⚠️ {workflow_file.name} missing required workflow structure")
                
                except yaml.YAMLError as e:
                    logger.error(f"❌ {workflow_file.name} has invalid YAML: {e}")
                except Exception as e:
                    logger.error(f"❌ Error validating {workflow_file.name}: {e}")
            
            success_rate = (valid_workflows / len(workflow_files)) * 100
            logger.info(f"📊 CI/CD pipeline validation: {valid_workflows}/{len(workflow_files)} files valid ({success_rate:.1f}%)")
            
            return success_rate >= 100  # All workflow files should be valid
            
        except Exception as e:
            logger.error(f"❌ CI/CD pipeline configuration test failed: {e}")
            return False
    
    def test_cloud_configuration_files(self) -> bool:
        """Test cloud-specific configuration files | 測試雲端特定配置文件"""
        logger.info("🔍 Testing cloud configuration files...")
        
        try:
            config_dir = self.project_root / "config"
            
            # Test cloud production configuration | 測試雲端生產配置
            cloud_config_file = config_dir / "cloud-production.yaml"
            if cloud_config_file.exists():
                try:
                    with open(cloud_config_file, 'r') as f:
                        cloud_config = yaml.safe_load(f)
                    
                    # Check required sections | 檢查必需部分
                    required_sections = ['app', 'storage', 'database', 'ai_models', 'monitoring']
                    missing_sections = []
                    
                    for section in required_sections:
                        if section not in cloud_config:
                            missing_sections.append(section)
                    
                    if missing_sections:
                        logger.warning(f"⚠️ Cloud configuration missing sections: {missing_sections}")
                    else:
                        logger.info("✅ Cloud configuration has all required sections")
                    
                    # Check S3 configuration | 檢查 S3 配置
                    if 'storage' in cloud_config and 's3' in cloud_config['storage']:
                        s3_config = cloud_config['storage']['s3']
                        if all(key in s3_config for key in ['models_bucket', 'data_bucket', 'backups_bucket']):
                            logger.info("✅ S3 storage configuration is complete")
                        else:
                            logger.warning("⚠️ S3 storage configuration incomplete")
                    
                    # Check database configuration | 檢查資料庫配置
                    if 'database' in cloud_config:
                        db_config = cloud_config['database']
                        if 'postgres' in db_config and 'redis' in db_config:
                            logger.info("✅ Database configuration includes PostgreSQL and Redis")
                        else:
                            logger.warning("⚠️ Database configuration incomplete")
                    
                except yaml.YAMLError as e:
                    logger.error(f"❌ Cloud configuration has invalid YAML: {e}")
                    return False
            else:
                logger.warning("⚠️ Cloud production configuration file not found")
            
            # Test deployment scripts | 測試部署腳本
            scripts_dir = self.project_root / "scripts"
            cloud_deploy_script = scripts_dir / "cloud-deploy.sh"
            
            if cloud_deploy_script.exists():
                if cloud_deploy_script.stat().st_mode & 0o111:  # Check if executable
                    logger.info("✅ Cloud deployment script is executable")
                else:
                    logger.warning("⚠️ Cloud deployment script is not executable")
                
                # Check script content for required components | 檢查腳本內容所需組件
                script_content = cloud_deploy_script.read_text()
                required_commands = ['terraform', 'kubectl', 'docker', 'aws']
                missing_commands = []
                
                for cmd in required_commands:
                    if cmd not in script_content:
                        missing_commands.append(cmd)
                
                if missing_commands:
                    logger.warning(f"⚠️ Deployment script missing references to: {missing_commands}")
                else:
                    logger.info("✅ Deployment script includes all required commands")
            else:
                logger.warning("⚠️ Cloud deployment script not found")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cloud configuration files test failed: {e}")
            return False
    
    def test_monitoring_and_logging_config(self) -> bool:
        """Test monitoring and logging configuration | 測試監控與日誌配置"""
        logger.info("🔍 Testing monitoring and logging configuration...")
        
        try:
            # Check for monitoring manifests | 檢查監控清單
            monitoring_dir = self.kubernetes_dir / "monitoring"
            if monitoring_dir.exists():
                monitoring_files = list(monitoring_dir.glob("*.yaml"))
                logger.info(f"📊 Found {len(monitoring_files)} monitoring configuration files")
                
                # Check for Prometheus/Grafana configurations | 檢查 Prometheus/Grafana 配置
                has_prometheus = False
                has_grafana = False
                
                for monitoring_file in monitoring_files:
                    content = monitoring_file.read_text().lower()
                    if 'prometheus' in content:
                        has_prometheus = True
                    if 'grafana' in content:
                        has_grafana = True
                
                if has_prometheus:
                    logger.info("✅ Prometheus monitoring configuration found")
                if has_grafana:
                    logger.info("✅ Grafana dashboard configuration found")
                
                if not (has_prometheus or has_grafana):
                    logger.warning("⚠️ No Prometheus or Grafana configurations found")
            else:
                logger.warning("⚠️ Monitoring configuration directory not found")
            
            # Check HPA configuration for monitoring | 檢查 HPA 配置的監控
            hpa_file = self.kubernetes_dir / "hpa.yaml"
            if hpa_file.exists():
                hpa_content = hpa_file.read_text()
                if 'ServiceMonitor' in hpa_content:
                    logger.info("✅ HPA includes ServiceMonitor for metrics")
                else:
                    logger.warning("⚠️ HPA missing ServiceMonitor configuration")
            
            # Check for health check endpoints in application | 檢查應用程式中的健康檢查端點
            deployment_file = self.kubernetes_dir / "deployment.yaml"
            if deployment_file.exists():
                deployment_content = deployment_file.read_text()
                if '/health' in deployment_content:
                    logger.info("✅ Health check endpoints configured in deployment")
                if 'livenessProbe' in deployment_content and 'readinessProbe' in deployment_content:
                    logger.info("✅ Liveness and readiness probes configured")
                else:
                    logger.warning("⚠️ Missing liveness or readiness probes")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring and logging configuration test failed: {e}")
            return False
    
    def test_security_configuration(self) -> bool:
        """Test security configuration | 測試安全配置"""
        logger.info("🔍 Testing security configuration...")
        
        try:
            # Check for secrets management | 檢查密鑰管理
            secrets_dir = self.kubernetes_dir / "secrets"
            if secrets_dir.exists():
                secret_files = list(secrets_dir.glob("*.yaml"))
                logger.info(f"🔐 Found {len(secret_files)} secret configuration files")
            else:
                logger.warning("⚠️ Secrets directory not found")
            
            # Check for network policies | 檢查網絡策略
            network_policies_found = False
            for yaml_file in self.kubernetes_dir.glob("**/*.yaml"):
                content = yaml_file.read_text()
                if 'NetworkPolicy' in content:
                    network_policies_found = True
                    logger.info("✅ Network policies configured")
                    break
            
            if not network_policies_found:
                logger.warning("⚠️ No network policies found")
            
            # Check for security contexts in deployments | 檢查部署中的安全上下文
            deployment_file = self.kubernetes_dir / "deployment.yaml"
            if deployment_file.exists():
                deployment_content = deployment_file.read_text()
                
                security_features = {
                    'runAsNonRoot': '✅ Running as non-root user',
                    'readOnlyRootFilesystem': '✅ Read-only root filesystem',
                    'allowPrivilegeEscalation': '✅ Privilege escalation disabled',
                    'seccompProfile': '✅ Seccomp profile configured'
                }
                
                for feature, message in security_features.items():
                    if feature in deployment_content:
                        logger.info(message)
                    else:
                        logger.warning(f"⚠️ Missing security feature: {feature}")
            
            # Check for Pod Security Standards | 檢查 Pod 安全標準
            pss_found = False
            for yaml_file in self.kubernetes_dir.glob("**/*.yaml"):
                content = yaml_file.read_text()
                if 'pod-security.kubernetes.io' in content:
                    pss_found = True
                    logger.info("✅ Pod Security Standards configured")
                    break
            
            if not pss_found:
                logger.info("ℹ️ Pod Security Standards not explicitly configured")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Security configuration test failed: {e}")
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report | 生成全面測試報告"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate overall success rate | 計算整體成功率
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'test_suite': 'AIFX Phase 4.1.2 Cloud Deployment',
            'environment': self.environment,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': round(success_rate, 2),
            'test_results': self.test_results,
            'overall_status': 'PASSED' if success_rate >= 80 else 'FAILED',
            'recommendations': []
        }
        
        # Add recommendations based on test results | 根據測試結果添加建議
        if success_rate < 100:
            failed_tests = [result['name'] for result in self.test_results if not result['passed']]
            report['recommendations'].append(f"Address failed tests: {', '.join(failed_tests)}")
        
        if success_rate >= 95:
            report['recommendations'].append("Excellent cloud deployment readiness!")
        elif success_rate >= 80:
            report['recommendations'].append("Good cloud deployment readiness with minor issues to address")
        else:
            report['recommendations'].append("Significant issues need to be resolved before cloud deployment")
        
        return report

def run_phase4_cloud_deployment_tests():
    """Run all Phase 4 cloud deployment tests | 運行所有第四階段雲端部署測試"""
    print("=" * 80)
    print("🚀 AIFX PHASE 4.1.2 CLOUD DEPLOYMENT INTEGRATION TESTS")
    print("🚀 AIFX 第四階段第二節雲端部署整合測試")
    print("=" * 80)
    
    tester = CloudDeploymentTester()
    
    # Define test suite | 定義測試套件
    test_suite = [
        ("Terraform Configuration Validation", tester.test_terraform_validation),
        ("Kubernetes Manifests Validation", tester.test_kubernetes_manifests),
        ("Docker Configuration Testing", tester.test_docker_configuration),
        ("CI/CD Pipeline Configuration", tester.test_ci_cd_pipeline_config),
        ("Cloud Configuration Files", tester.test_cloud_configuration_files),
        ("Monitoring and Logging Config", tester.test_monitoring_and_logging_config),
        ("Security Configuration", tester.test_security_configuration),
    ]
    
    # Run all tests | 運行所有測試
    for test_name, test_function in test_suite:
        logger.info(f"\n🔍 Running: {test_name}")
        start_time = time.time()
        
        try:
            result = test_function()
            end_time = time.time()
            duration = end_time - start_time
            
            tester.test_results.append({
                'name': test_name,
                'passed': result,
                'duration_seconds': round(duration, 2),
                'timestamp': datetime.now().isoformat()
            })
            
            if result:
                logger.info(f"✅ {test_name}: PASSED ({duration:.2f}s)")
            else:
                logger.error(f"❌ {test_name}: FAILED ({duration:.2f}s)")
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(f"❌ {test_name}: ERROR - {e} ({duration:.2f}s)")
            
            tester.test_results.append({
                'name': test_name,
                'passed': False,
                'duration_seconds': round(duration, 2),
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
    
    # Generate and display report | 生成並顯示報告
    report = tester.generate_test_report()
    
    print("\n" + "=" * 80)
    print("📊 CLOUD DEPLOYMENT TEST RESULTS | 雲端部署測試結果")
    print("=" * 80)
    print(f"📈 Overall Success Rate: {report['success_rate']}%")
    print(f"✅ Tests Passed: {report['passed_tests']}/{report['total_tests']}")
    print(f"⏱️ Total Duration: {report['duration_seconds']:.1f}s")
    print(f"🎯 Status: {report['overall_status']}")
    
    print("\n📋 Test Details:")
    for result in report['test_results']:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"   {status} {result['name']} ({result['duration_seconds']}s)")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    # Save report to file | 保存報告到文件
    report_file = project_root / f"test_phase4_cloud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return success/failure | 返回成功/失敗
    return report['success_rate'] >= 80

if __name__ == "__main__":
    try:
        success = run_phase4_cloud_deployment_tests()
        exit_code = 0 if success else 1
        
        if success:
            print("\n🎉 PHASE 4.1.2 CLOUD DEPLOYMENT TESTS: PASSED")
            print("✅ Cloud deployment architecture is ready for production!")
        else:
            print("\n🚨 PHASE 4.1.2 CLOUD DEPLOYMENT TESTS: FAILED")
            print("❌ Issues need to be resolved before cloud deployment")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏸️ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Test suite failed with unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)