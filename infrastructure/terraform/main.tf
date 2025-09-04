# AIFX - Main Terraform Configuration
# AIFX - 主要Terraform配置
# Infrastructure as Code for AIFX production deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  # Backend configuration for state management | 狀態管理的後端配置
  backend "s3" {
    bucket         = "aifx-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "aifx-terraform-locks"
  }
}

# ============================================================================
# Provider Configuration | 提供者配置
# ============================================================================

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "AIFX"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "AIFX-Team"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ============================================================================
# Local Variables | 本地變數
# ============================================================================

locals {
  name = "aifx-${var.environment}"
  
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
  
  tags = {
    Project     = "AIFX"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# ============================================================================
# Data Sources | 數據源
# ============================================================================

data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# ============================================================================
# VPC Module | VPC模組
# ============================================================================

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = local.name
  cidr = local.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]
  intra_subnets   = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 52)]

  enable_nat_gateway = true
  single_nat_gateway = var.environment != "production"
  enable_vpn_gateway = false

  enable_dns_hostnames = true
  enable_dns_support   = true

  # Kubernetes integration | Kubernetes整合
  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = local.tags
}

# ============================================================================
# EKS Cluster | EKS集群
# ============================================================================

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.name
  cluster_version = "1.28"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS Managed Node Groups | EKS管理節點組
  eks_managed_node_groups = {
    # General purpose nodes for AIFX application | AIFX應用的通用節點
    aifx_nodes = {
      name = "aifx-nodes"
      
      instance_types = ["t3.large"]
      
      min_size     = 2
      max_size     = 10
      desired_size = 3

      pre_bootstrap_user_data = <<-EOT
        #!/bin/bash
        set -ex
        /etc/eks/bootstrap.sh ${local.name}
      EOT

      vpc_security_group_ids = [aws_security_group.node_group_one.id]
    }

    # High-memory nodes for AI models | AI模型的高記憶體節點
    aifx_ai_nodes = {
      name = "aifx-ai-nodes"
      
      instance_types = ["r5.xlarge"]
      
      min_size     = 1
      max_size     = 5
      desired_size = 2

      labels = {
        role = "ai-workload"
      }

      taints = {
        dedicated = {
          key    = "ai-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      vpc_security_group_ids = [aws_security_group.node_group_two.id]
    }
  }

  # Cluster access entry | 集群訪問條目
  access_entries = {
    aifx_admin = {
      kubernetes_groups = ["system:masters"]
      principal_arn     = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/AifxAdminRole"

      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }

  tags = local.tags
}

# ============================================================================
# Security Groups | 安全組
# ============================================================================

resource "aws_security_group" "node_group_one" {
  name_prefix = "${local.name}-node-group-one"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-node-group-one"
  })
}

resource "aws_security_group" "node_group_two" {
  name_prefix = "${local.name}-node-group-two"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-node-group-two"
  })
}

# ============================================================================
# RDS Database | RDS資料庫
# ============================================================================

resource "aws_db_subnet_group" "aifx" {
  name       = "${local.name}-db"
  subnet_ids = module.vpc.private_subnets

  tags = merge(local.tags, {
    Name = "${local.name}-db"
  })
}

resource "aws_security_group" "rds" {
  name        = "${local.name}-rds"
  description = "RDS security group for AIFX"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-rds"
  })
}

resource "aws_db_instance" "aifx" {
  identifier = "${local.name}-postgres"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.environment == "production" ? "db.r5.large" : "db.t3.micro"

  allocated_storage     = var.environment == "production" ? 100 : 20
  max_allocated_storage = var.environment == "production" ? 1000 : 100

  db_name  = "aifx"
  username = "aifx_user"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.aifx.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = var.environment == "production" ? 7 : 1
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"

  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"

  performance_insights_enabled = var.environment == "production"
  monitoring_interval         = var.environment == "production" ? 60 : 0

  tags = merge(local.tags, {
    Name = "${local.name}-postgres"
  })
}

# ============================================================================
# ElastiCache Redis | ElastiCache Redis快取
# ============================================================================

resource "aws_elasticache_subnet_group" "aifx" {
  name       = "${local.name}-cache"
  subnet_ids = module.vpc.private_subnets

  tags = local.tags
}

resource "aws_security_group" "elasticache" {
  name        = "${local.name}-elasticache"
  description = "ElastiCache security group for AIFX"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-elasticache"
  })
}

resource "aws_elasticache_replication_group" "aifx" {
  replication_group_id       = "${local.name}-redis"
  description                = "Redis cluster for AIFX"
  
  port                = 6379
  parameter_group_name = "default.redis7"
  node_type           = var.environment == "production" ? "cache.r5.large" : "cache.t3.micro"
  
  num_cache_clusters = var.environment == "production" ? 3 : 1
  
  subnet_group_name  = aws_elasticache_subnet_group.aifx.name
  security_group_ids = [aws_security_group.elasticache.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = local.tags
}

# ============================================================================
# Application Load Balancer | 應用負載均衡器
# ============================================================================

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Application Load Balancer security group"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-alb"
  })
}