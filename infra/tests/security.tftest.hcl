# What the machine exposes and how it is built, read from a plan against a
# mocked provider: no credentials, no account, nothing created.
#
# A mocked plan knows what the configuration says and not what AWS would
# answer. Where a test needs an identifier AWS would give (the data volume's
# id, inside the first-boot script), it is supplied below.

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "000000000000"
    }
  }

  mock_data "aws_partition" {
    defaults = {
      partition  = "aws"
      dns_suffix = "amazonaws.com"
    }
  }

  mock_data "aws_region" {
    defaults = {
      region = "eu-west-1"
    }
  }

  mock_data "aws_subnets" {
    defaults = {
      ids = ["subnet-b", "subnet-a"]
    }
  }

  mock_data "aws_subnet" {
    defaults = {
      availability_zone = "eu-west-1a"
    }
  }
}

override_resource {
  target          = aws_ebs_volume.data
  override_during = plan
  values = {
    id = "vol-00000000000000000"
  }
}

# A key pair's name is one of the things AWS may answer with, so a plan
# does not know it when the configuration is silent. Here the mock's answer
# is "none"; a name written in the configuration is still what the plan shows.
override_resource {
  target          = aws_instance.server
  override_during = plan
  values = {
    key_name = null
  }
}

variables {
  site_name    = "causalpulse.example"
  repository   = "owner/name"
  budget_email = "you@example.com"
  ami_id       = "ami-00000000000000000"
}

run "inbound_is_http_and_https_over_ipv4_only" {
  command = plan

  assert {
    condition     = length(aws_security_group.server.ingress) == 2
    error_message = "The security group must have exactly two inbound rules."
  }

  assert {
    condition     = toset([for rule in aws_security_group.server.ingress : rule.from_port]) == toset([80, 443])
    error_message = "The inbound rules must be for ports 80 and 443, and no other."
  }

  assert {
    condition = alltrue([
      for rule in aws_security_group.server.ingress :
      rule.from_port == rule.to_port && rule.protocol == "tcp"
    ])
    error_message = "Each inbound rule must be one TCP port, not a range."
  }

  assert {
    condition = alltrue([
      for rule in aws_security_group.server.ingress :
      rule.cidr_blocks == tolist(["0.0.0.0/0"])
    ])
    error_message = "Each inbound rule must be from 0.0.0.0/0 and nowhere else."
  }

  assert {
    condition = alltrue([
      for rule in concat(tolist(aws_security_group.server.ingress), tolist(aws_security_group.server.egress)) :
      try(length(rule.ipv6_cidr_blocks), 0) == 0
    ])
    error_message = "No rule may name an IPv6 range."
  }

  assert {
    condition = alltrue([
      for rule in aws_security_group.server.ingress :
      try(length(rule.security_groups), 0) == 0 && try(length(rule.prefix_list_ids), 0) == 0 && rule.self != true
    ])
    error_message = "No inbound rule may admit a security group or a prefix list."
  }
}

run "the_machine_has_no_key_no_ipv6_and_version_2_metadata" {
  command = plan

  assert {
    condition     = aws_instance.server.key_name == null
    error_message = "The instance must have no key pair."
  }

  assert {
    condition     = aws_instance.server.ipv6_address_count == 0
    error_message = "The instance must have no IPv6 address."
  }

  assert {
    condition     = aws_instance.server.metadata_options[0].http_tokens == "required"
    error_message = "The metadata service must require version 2 (http_tokens = required)."
  }

  assert {
    condition     = aws_instance.server.metadata_options[0].http_put_response_hop_limit == 1
    error_message = "The metadata service's hop limit must be 1, so that a container cannot use the instance's role."
  }

  assert {
    condition     = aws_instance.server.credit_specification[0].cpu_credits == "standard"
    error_message = "The CPU credit mode must be standard: unlimited is billed."
  }

  assert {
    condition     = aws_instance.server.instance_type == "t3.medium"
    error_message = "The instance must be a t3.medium."
  }

  assert {
    condition     = aws_instance.server.ami == var.ami_id
    error_message = "The instance must be created from the pinned image."
  }

  assert {
    condition     = aws_instance.server.user_data_replace_on_change == false
    error_message = "A changed first-boot script must not replace the machine."
  }

  assert {
    condition     = aws_instance.server.iam_instance_profile == "causalpulse-server"
    error_message = "The instance must run under the server's own profile."
  }
}

run "both_disks_are_encrypted_and_sized" {
  command = plan

  assert {
    condition     = aws_ebs_volume.data.encrypted == true
    error_message = "The data volume must be encrypted."
  }

  assert {
    condition     = aws_ebs_volume.data.size == 20
    error_message = "The data volume must be 20 GB."
  }

  assert {
    condition     = aws_instance.server.root_block_device[0].encrypted == true
    error_message = "The root disk must be encrypted."
  }

  assert {
    condition     = aws_instance.server.root_block_device[0].volume_size == 30
    error_message = "The root disk must be 30 GB."
  }

  assert {
    condition     = aws_volume_attachment.data.device_name == "/dev/sdf"
    error_message = "The data volume must be attached as /dev/sdf."
  }
}

run "only_the_data_volume_is_snapshotted" {
  command = plan

  assert {
    condition     = aws_ebs_volume.data.tags["Backup"] == "causalpulse-data"
    error_message = "The data volume must carry the tag the snapshot policy looks for."
  }

  assert {
    condition     = aws_dlm_lifecycle_policy.data.policy_details[0].target_tags == tomap({ Backup = "causalpulse-data" })
    error_message = "The snapshot policy must look for the data volume's own tag and no other."
  }

  assert {
    condition     = !contains(keys(aws_instance.server.root_block_device[0].tags), "Backup")
    error_message = "The root disk must not carry the snapshot policy's tag."
  }

  assert {
    condition     = aws_dlm_lifecycle_policy.data.policy_details[0].schedule[0].retain_rule[0].count == 7
    error_message = "Seven snapshots must be kept."
  }

  assert {
    condition     = aws_dlm_lifecycle_policy.data.policy_details[0].schedule[0].create_rule[0].times == tolist(["03:00"])
    error_message = "The snapshot must be taken at 03:00 UTC."
  }

  assert {
    condition     = aws_dlm_lifecycle_policy.data.state == "ENABLED"
    error_message = "The snapshot policy must be enabled."
  }
}

run "the_deploy_command_takes_one_guarded_parameter" {
  command = plan

  assert {
    condition     = local.deploy_document.parameters.Sha.allowedPattern == "^[0-9a-f]{40}$"
    error_message = "The Sha parameter's allowedPattern must be ^[0-9a-f]{40}$: the value is put into a shell line."
  }

  assert {
    condition     = keys(local.deploy_document.parameters) == ["Sha"]
    error_message = "The deploy document must take the Sha parameter and no other."
  }

  assert {
    condition     = local.deploy_document.schemaVersion == "2.2" && length(local.deploy_document.mainSteps) == 1 && local.deploy_document.mainSteps[0].action == "aws:runShellScript"
    error_message = "The deploy document must be schema 2.2 with one aws:runShellScript step."
  }

  assert {
    condition     = aws_ssm_document.deploy.content == jsonencode(local.deploy_document) && aws_ssm_document.deploy.document_type == "Command"
    error_message = "The document resource must be a Command whose content is the tested local."
  }

  assert {
    condition     = length(regexall("\\{\\{", local.deploy_script)) == length(regexall("[\":]\\{\\{ Sha \\}\\}\"", local.deploy_script))
    error_message = "Every substitution in the deploy script must be {{ Sha }}, inside double quotes."
  }

  assert {
    condition     = element(local.deploy_document.mainSteps[0].inputs.runCommand, -1) == "exec sh deploy/deploy.sh \"$registry\" \"{{ Sha }}\""
    error_message = "The deploy script must end by running deploy/deploy.sh with the registry and the SHA."
  }

  assert {
    condition     = strcontains(local.deploy_script, "registry=\"000000000000.dkr.ecr.eu-west-1.amazonaws.com\"")
    error_message = "The registry's host must be the caller's account and the region."
  }
}

run "everything_is_tagged_for_the_project" {
  command = plan

  assert {
    condition     = local.default_tags == { Project = "causalpulse" }
    error_message = "The provider's default tags must be Project = causalpulse."
  }

  # The two places the provider's default tags do not reach.
  assert {
    condition     = aws_instance.server.root_block_device[0].tags["Project"] == "causalpulse"
    error_message = "The root disk must be tagged Project = causalpulse."
  }

  assert {
    condition     = aws_dlm_lifecycle_policy.data.policy_details[0].schedule[0].tags_to_add["Project"] == "causalpulse"
    error_message = "The snapshots must be tagged Project = causalpulse."
  }
}

run "images_are_scanned_and_five_are_kept" {
  command = plan

  assert {
    condition     = aws_ecr_repository.app.name == "causalpulse-app" && aws_ecr_repository.web.name == "causalpulse-web"
    error_message = "The repositories must be causalpulse-app and causalpulse-web."
  }

  assert {
    condition     = aws_ecr_repository.app.image_scanning_configuration[0].scan_on_push && aws_ecr_repository.web.image_scanning_configuration[0].scan_on_push
    error_message = "Both repositories must scan on push."
  }

  assert {
    condition     = local.image_lifecycle.rules[0].selection.countNumber == 5 && local.image_lifecycle.rules[0].selection.tagStatus == "any"
    error_message = "The lifecycle rule must keep five images."
  }

  assert {
    condition     = aws_ecr_lifecycle_policy.app.policy == jsonencode(local.image_lifecycle) && aws_ecr_lifecycle_policy.web.policy == jsonencode(local.image_lifecycle)
    error_message = "Both repositories must use the tested lifecycle rule."
  }
}

run "the_budget_warns_twice_at_80_percent_of_50_dollars" {
  command = plan

  assert {
    condition     = aws_budgets_budget.monthly.limit_amount == "50" && aws_budgets_budget.monthly.limit_unit == "USD" && aws_budgets_budget.monthly.time_unit == "MONTHLY"
    error_message = "The budget must be 50 US dollars a month."
  }

  assert {
    condition     = toset([for n in aws_budgets_budget.monthly.notification : n.notification_type]) == toset(["ACTUAL", "FORECASTED"])
    error_message = "The budget must warn on actual and on forecast spending."
  }

  assert {
    condition = alltrue([
      for n in aws_budgets_budget.monthly.notification :
      n.threshold == 80 && n.threshold_type == "PERCENTAGE" && n.subscriber_email_addresses == toset([var.budget_email])
    ])
    error_message = "Each warning must be at 80 percent, to the budget's address."
  }
}

run "a_failed_host_is_recovered" {
  command = plan

  assert {
    condition     = aws_cloudwatch_metric_alarm.recover.metric_name == "StatusCheckFailed_System" && aws_cloudwatch_metric_alarm.recover.namespace == "AWS/EC2"
    error_message = "The alarm must watch the system status check."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.recover.alarm_actions == toset(["arn:aws:automate:eu-west-1:ec2:recover"])
    error_message = "The alarm's one action must be EC2's recover."
  }
}

run "the_first_boot_script_is_given_whole" {
  command = plan

  assert {
    condition     = length(local.first_boot) < 16384
    error_message = "User data may not exceed 16 KB."
  }

  assert {
    condition     = strcontains(local.first_boot, "site_name='causalpulse.example'") && strcontains(local.first_boot, "volume_id='vol-00000000000000000'")
    error_message = "The first-boot script must be given the site's name and the data volume's id."
  }

  assert {
    condition     = startswith(local.first_boot, "#!/bin/bash\n")
    error_message = "cloud-init runs user data as a script only when it begins with an interpreter line."
  }
}

run "a_site_name_that_could_break_out_of_a_quote_is_refused" {
  command = plan

  variables {
    site_name = "x'; reboot; '"
  }

  expect_failures = [var.site_name]
}
