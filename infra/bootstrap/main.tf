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

# The region is written and is not a variable: it must be the one the
# backend block in `../versions.tf` names, which cannot follow a variable.
provider "aws" {
  region = "eu-west-1"

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

# An earlier state can be brought back after a bad write (for how long, see
# the lifecycle rule below).
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

# Every request must come over TLS. The public access block comes first: AWS
# refuses two changes to one bucket's settings at the same moment.
resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "TlsOnly"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]
      Condition = {
        Bool = { "aws:SecureTransport" = "false" }
      }
    }]
  })

  depends_on = [aws_s3_bucket_public_access_block.state]
}

# Each apply writes a new version of the state, and each lock writes and
# deletes a small object, which leaves a version behind. A version that has
# been replaced is removed 90 days later, so an earlier state can be brought
# back for 90 days and not for ever. The current version is never removed.
resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    id     = "expire-replaced-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  # A lifecycle rule for versions is refused on a bucket without versioning.
  depends_on = [aws_s3_bucket_versioning.state]
}

output "state_bucket" {
  description = "The name to give `terraform init -backend-config=\"bucket=...\"` in `infra/`."
  value       = aws_s3_bucket.state.bucket
}
