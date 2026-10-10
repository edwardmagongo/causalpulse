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
  # cannot read a variable, so the region is written here. It is the region
  # `bootstrap/` creates the bucket in and the region of the provider below.
  backend "s3" {
    key          = "causalpulse/terraform.tfstate"
    region       = "eu-west-1"
    encrypt      = true
    use_lockfile = true
  }
}

locals {
  # Every resource that takes tags gets these, through the provider. The
  # tests read this local, because a mocked plan does not know `tags_all`,
  # and `tests/inventory.sh` checks that the provider is given it.
  default_tags = {
    Project = "causalpulse"
  }
}

# The region is written and is not a variable: the backend above cannot
# follow one, and the commands in `terraform.tfvars.example` name this region.
provider "aws" {
  region = "eu-west-1"

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
