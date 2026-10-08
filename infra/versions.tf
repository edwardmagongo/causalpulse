terraform {
  required_version = "~> 1.15"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # The bucket is made by `bootstrap/` and named when the state is first set
  # up: `terraform init -backend-config="bucket=<name>"`. A backend block
  # cannot read a variable, so the region is written here; it is the region
  # of the bucket, which `bootstrap/` creates in the same default.
  backend "s3" {
    key          = "causalpulse/terraform.tfstate"
    region       = "eu-west-1"
    encrypt      = true
    use_lockfile = true
  }
}

locals {
  # Every resource that takes tags gets these, through the provider. The
  # tests read this local: a mocked plan does not know `tags_all`.
  default_tags = {
    Project = "causalpulse"
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = local.default_tags
  }
}

# The account, partition and region are read from whoever runs Terraform.
# No account number is written in this directory.
data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
  # The provider's region, as the provider reports it.
  region = data.aws_region.current.region
}
