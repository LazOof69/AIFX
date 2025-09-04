# AIFX - Terraform Outputs Configuration
# AIFX - Terraform輸出配置

# ============================================================================
# VPC Outputs | VPC輸出
# ============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "intra_subnets" {
  description = "List of IDs of intra subnets"
  value       = module.vpc.intra_subnets
}

# ============================================================================
# EKS Cluster Outputs | EKS集群輸出
# ============================================================================

output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "EKS cluster version"
  value       = module.eks.cluster_version
}

output "cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = module.eks.cluster_platform_version
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "node_security_group_id" {
  description = "ID of the node shared security group"
  value       = module.eks.node_security_group_id
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

# ============================================================================
# EKS Node Groups Outputs | EKS節點組輸出
# ============================================================================

output "eks_managed_node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value       = module.eks.eks_managed_node_groups
}

output "eks_managed_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names created by EKS managed node groups"
  value       = module.eks.eks_managed_node_groups_autoscaling_group_names
}

# ============================================================================
# Database Outputs | 資料庫輸出
# ============================================================================

output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.aifx.identifier
}

output "db_instance_address" {
  description = "RDS instance hostname"
  value       = aws_db_instance.aifx.address
  sensitive   = true
}

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.aifx.endpoint
  sensitive   = true
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.aifx.port
}

output "db_instance_name" {
  description = "RDS instance database name"
  value       = aws_db_instance.aifx.db_name
}

output "db_instance_username" {
  description = "RDS instance master username"
  value       = aws_db_instance.aifx.username
  sensitive   = true
}

output "db_subnet_group_name" {
  description = "RDS subnet group name"
  value       = aws_db_subnet_group.aifx.name
}

output "db_parameter_group_name" {
  description = "RDS DB parameter group name"
  value       = aws_db_instance.aifx.parameter_group_name
}

# ============================================================================
# Cache Outputs | 快取輸出
# ============================================================================

output "elasticache_replication_group_id" {
  description = "ElastiCache replication group identifier"
  value       = aws_elasticache_replication_group.aifx.replication_group_id
}

output "elasticache_primary_endpoint_address" {
  description = "Address of the endpoint for the primary node in the replication group"
  value       = aws_elasticache_replication_group.aifx.primary_endpoint_address
  sensitive   = true
}

output "elasticache_reader_endpoint_address" {
  description = "Address of the endpoint for the reader node in the replication group"
  value       = aws_elasticache_replication_group.aifx.reader_endpoint_address
  sensitive   = true
}

output "elasticache_configuration_endpoint_address" {
  description = "Address of the replication group configuration endpoint"
  value       = aws_elasticache_replication_group.aifx.configuration_endpoint_address
  sensitive   = true
}

# ============================================================================
# Security Groups Outputs | 安全組輸出
# ============================================================================

output "node_security_group_one_id" {
  description = "Security group ID for node group one"
  value       = aws_security_group.node_group_one.id
}

output "node_security_group_two_id" {
  description = "Security group ID for node group two"
  value       = aws_security_group.node_group_two.id
}

output "rds_security_group_id" {
  description = "Security group ID for RDS"
  value       = aws_security_group.rds.id
}

output "elasticache_security_group_id" {
  description = "Security group ID for ElastiCache"
  value       = aws_security_group.elasticache.id
}

output "alb_security_group_id" {
  description = "Security group ID for Application Load Balancer"
  value       = aws_security_group.alb.id
}

# ============================================================================
# Configuration for kubectl | kubectl配置
# ============================================================================

output "configure_kubectl" {
  description = "Configure kubectl: make sure you're logged in with the correct AWS profile and run the following command to update your kubeconfig"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${module.eks.cluster_name}"
}

# ============================================================================
# Environment Information | 環境信息
# ============================================================================

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

# ============================================================================
# Cost Tracking | 成本跟踪
# ============================================================================

output "tags" {
  description = "A map of tags assigned to the resource"
  value       = local.tags
}

# ============================================================================
# Connection Strings | 連接字符串
# ============================================================================

output "database_connection_string" {
  description = "Database connection string (without password)"
  value       = "postgresql://${aws_db_instance.aifx.username}@${aws_db_instance.aifx.endpoint}/${aws_db_instance.aifx.db_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${aws_elasticache_replication_group.aifx.primary_endpoint_address}:6379"
  sensitive   = true
}

# ============================================================================
# Kubernetes Service Account | Kubernetes服務帳戶
# ============================================================================

output "aws_load_balancer_controller_role_arn" {
  description = "IAM role ARN for AWS Load Balancer Controller"
  value       = module.eks.cluster_service_accounts.aws-load-balancer-controller.iam_role_arn
  depends_on  = [module.eks]
}

output "external_dns_role_arn" {
  description = "IAM role ARN for External DNS"
  value       = module.eks.cluster_service_accounts.external-dns.iam_role_arn
  depends_on  = [module.eks]
}

output "cluster_autoscaler_role_arn" {
  description = "IAM role ARN for Cluster Autoscaler"
  value       = module.eks.cluster_service_accounts.cluster-autoscaler.iam_role_arn
  depends_on  = [module.eks]
}

# ============================================================================
# Monitoring Endpoints | 監控端點
# ============================================================================

output "prometheus_server_endpoint" {
  description = "Prometheus server endpoint (internal)"
  value       = "http://prometheus-server.monitoring.svc.cluster.local"
}

output "grafana_endpoint" {
  description = "Grafana endpoint (internal)"
  value       = "http://grafana.monitoring.svc.cluster.local"
}

# ============================================================================
# Application Deployment Information | 應用部署信息
# ============================================================================

output "aifx_namespace" {
  description = "Kubernetes namespace for AIFX application"
  value       = "aifx"
}

output "monitoring_namespace" {
  description = "Kubernetes namespace for monitoring stack"
  value       = "monitoring"
}

output "ingress_controller_namespace" {
  description = "Kubernetes namespace for ingress controller"
  value       = "ingress-nginx"
}