# AIFX - Terraform Variables Configuration
# AIFX - Terraform變數配置

# ============================================================================
# General Configuration | 通用配置
# ============================================================================

variable "aws_region" {
  description = "AWS region for infrastructure deployment"
  type        = string
  default     = "us-west-2"
  
  validation {
    condition = contains([
      "us-east-1", "us-east-2", "us-west-1", "us-west-2",
      "eu-west-1", "eu-west-2", "eu-central-1",
      "ap-southeast-1", "ap-southeast-2", "ap-northeast-1"
    ], var.aws_region)
    error_message = "AWS region must be a valid region with EKS support."
  }
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "project_name" {
  description = "Name of the AIFX project"
  type        = string
  default     = "aifx"
}

# ============================================================================
# EKS Configuration | EKS配置
# ============================================================================

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    min_size      = number
    max_size      = number
    desired_size  = number
    disk_size     = number
    labels        = map(string)
    taints = map(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  
  default = {
    aifx_nodes = {
      instance_types = ["t3.large"]
      min_size      = 2
      max_size      = 10
      desired_size  = 3
      disk_size     = 50
      labels        = { role = "general" }
      taints        = {}
    }
    
    aifx_ai_nodes = {
      instance_types = ["r5.xlarge"]
      min_size      = 1
      max_size      = 5
      desired_size  = 2
      disk_size     = 100
      labels        = { role = "ai-workload" }
      taints = {
        dedicated = {
          key    = "ai-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler for EKS"
  type        = bool
  default     = true
}

# ============================================================================
# Database Configuration | 資料庫配置
# ============================================================================

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r5.large"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS instance (GB)"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS instance (GB)"
  type        = number
  default     = 1000
}

variable "db_password" {
  description = "Password for RDS database"
  type        = string
  sensitive   = true
  
  validation {
    condition     = length(var.db_password) >= 8
    error_message = "Database password must be at least 8 characters long."
  }
}

variable "db_backup_retention_period" {
  description = "Backup retention period for RDS (days)"
  type        = number
  default     = 7
  
  validation {
    condition     = var.db_backup_retention_period >= 1 && var.db_backup_retention_period <= 35
    error_message = "Backup retention period must be between 1 and 35 days."
  }
}

variable "enable_db_deletion_protection" {
  description = "Enable deletion protection for RDS instance"
  type        = bool
  default     = true
}

# ============================================================================
# Cache Configuration | 快取配置
# ============================================================================

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r5.large"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters for Redis replication group"
  type        = number
  default     = 3
  
  validation {
    condition     = var.redis_num_cache_clusters >= 1 && var.redis_num_cache_clusters <= 6
    error_message = "Number of cache clusters must be between 1 and 6."
  }
}

variable "redis_parameter_group" {
  description = "Parameter group name for Redis"
  type        = string
  default     = "default.redis7"
}

# ============================================================================
# Networking Configuration | 網路配置
# ============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway for private subnets"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT gateway for all private subnets"
  type        = bool
  default     = false
}

# ============================================================================
# Security Configuration | 安全配置
# ============================================================================

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all supported services"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit for all supported services"
  type        = bool
  default     = true
}

# ============================================================================
# Monitoring Configuration | 監控配置
# ============================================================================

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs for EKS cluster"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 14
  
  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = true
}

# ============================================================================
# Backup Configuration | 備份配置
# ============================================================================

variable "backup_window" {
  description = "Backup window for RDS (UTC)"
  type        = string
  default     = "03:00-04:00"
  
  validation {
    condition     = can(regex("^[0-2][0-9]:[0-5][0-9]-[0-2][0-9]:[0-5][0-9]$", var.backup_window))
    error_message = "Backup window must be in format HH:MM-HH:MM (24-hour UTC)."
  }
}

variable "maintenance_window" {
  description = "Maintenance window for RDS"
  type        = string
  default     = "Sun:04:00-Sun:05:00"
}

# ============================================================================
# Cost Optimization | 成本優化
# ============================================================================

variable "enable_cost_allocation_tags" {
  description = "Enable cost allocation tags"
  type        = bool
  default     = true
}

variable "instance_lifecycle" {
  description = "Instance lifecycle (spot/on-demand)"
  type        = string
  default     = "on-demand"
  
  validation {
    condition     = contains(["spot", "on-demand"], var.instance_lifecycle)
    error_message = "Instance lifecycle must be either 'spot' or 'on-demand'."
  }
}

# ============================================================================
# Feature Flags | 功能標誌
# ============================================================================

variable "enable_load_balancer_controller" {
  description = "Enable AWS Load Balancer Controller"
  type        = bool
  default     = true
}

variable "enable_external_dns" {
  description = "Enable External DNS for automatic DNS management"
  type        = bool
  default     = true
}

variable "enable_cert_manager" {
  description = "Enable Cert Manager for automatic SSL certificate management"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring stack"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

# ============================================================================
# AIFX Specific Configuration | AIFX特定配置
# ============================================================================

variable "aifx_image_tag" {
  description = "Docker image tag for AIFX application"
  type        = string
  default     = "latest"
}

variable "aifx_replica_count" {
  description = "Number of AIFX application replicas"
  type        = number
  default     = 3
  
  validation {
    condition     = var.aifx_replica_count >= 1 && var.aifx_replica_count <= 20
    error_message = "Replica count must be between 1 and 20."
  }
}

variable "aifx_cpu_request" {
  description = "CPU request for AIFX pods"
  type        = string
  default     = "500m"
}

variable "aifx_memory_request" {
  description = "Memory request for AIFX pods"
  type        = string
  default     = "1Gi"
}

variable "aifx_cpu_limit" {
  description = "CPU limit for AIFX pods"
  type        = string
  default     = "2"
}

variable "aifx_memory_limit" {
  description = "Memory limit for AIFX pods"
  type        = string
  default     = "4Gi"
}

# ============================================================================
# Domain Configuration | 域名配置
# ============================================================================

variable "domain_name" {
  description = "Domain name for AIFX application"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ARN of SSL certificate for HTTPS"
  type        = string
  default     = ""
}