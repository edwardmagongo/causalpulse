# The bucket that holds the state of `infra/`. Applied once, by hand, before
# anything else. Its own state is a local file, which git ignores: it names
# one bucket and holds no secret.
terraform {
  required_version = "~> 1.15"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

variable "state_bucket" {
  description = "The bucket's name, which must be unused in all of AWS."
  type        = string
}

variable "region" {
  description = "The AWS region the bucket is created in."
  type        = string
  default     = "eu-west-1"
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project = "causalpulse"
    }
  }
}

resource "aws_s3_bucket" "state" {
  bucket = var.state_bucket

  lifecycle {
    prevent_destroy = true
  }
}

# An earlier state can be brought back after a bad write.
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# The state names the account and its resources, so nothing about the bucket
# is public.
resource "aws_s3_bucket_public_access_block" "state" {
  bucket = aws_s3_bucket.state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "state_bucket" {
  description = "The name to give `terraform init -backend-config=\"bucket=...\"` in `infra/`."
  value       = aws_s3_bucket.state.bucket
}
