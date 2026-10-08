# None of these has a default that is a real value. Set them in
# `terraform.tfvars`, which git ignores; `terraform.tfvars.example` shows how.

variable "region" {
  description = "The AWS region everything is created in."
  type        = string
  default     = "eu-west-1"
}

variable "site_name" {
  description = "The site's public name, which the proxy asks a certificate for."
  type        = string

  # The name is written into a shell script and an environment file.
  validation {
    condition     = can(regex("^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$", var.site_name))
    error_message = "The site name may hold only lowercase letters, digits, dots and hyphens."
  }
}

variable "repository" {
  description = "The GitHub repository whose main branch may deploy, as owner/name."
  type        = string

  # The name becomes part of the subject the pipeline's role trusts, so it
  # may not hold a colon, a star or a question mark.
  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.repository))
    error_message = "The repository must be owner/name, in letters, digits, dots, hyphens and underscores."
  }
}

variable "budget_email" {
  description = "Where the budget's two warnings are sent."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+$", var.budget_email))
    error_message = "The budget's address must be an e-mail address."
  }
}

variable "ami_id" {
  description = "The Ubuntu LTS image for x86 the machine is created from. Pinned: see terraform.tfvars.example."
  type        = string

  validation {
    condition     = can(regex("^ami-[0-9a-f]{8,17}$", var.ami_id))
    error_message = "The image must be an AMI id."
  }
}
