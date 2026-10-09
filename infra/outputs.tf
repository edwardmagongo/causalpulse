# What the runbook and the repository's variables on GitHub are filled in from.

output "site_address" {
  description = "The server's fixed public address, for the site's A record."
  value       = aws_eip.server.public_ip
}

output "site_name" {
  description = "The site's public name, as given in terraform.tfvars: the SITE_NAME variable on GitHub."
  value       = var.site_name
}

output "data_volume_id" {
  description = "The volume that holds the database and the certificates: what a restore replaces."
  value       = aws_ebs_volume.data.id
}

output "instance_id" {
  description = "The server, as the deploy workflow names it."
  value       = aws_instance.server.id
}

output "pipeline_role_arn" {
  description = "The role the deploy workflow assumes."
  value       = aws_iam_role.pipeline.arn
}

output "deploy_document_name" {
  description = "The one command the deploy workflow may send."
  value       = aws_ssm_document.deploy.name
}

output "app_repository_url" {
  description = "Where the app image is pushed."
  value       = aws_ecr_repository.app.repository_url
}

output "web_repository_url" {
  description = "Where the dashboard image is pushed."
  value       = aws_ecr_repository.web.repository_url
}

output "registry" {
  description = "The registry's host, which both image names begin with."
  value       = local.registry
}
