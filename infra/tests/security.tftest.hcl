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

  # The subnet the owner names, as AWS would describe a default one.
  mock_data "aws_subnet" {
    defaults = {
      vpc_id                          = "vpc-00000000000000000"
      availability_zone               = "eu-west-1a"
      map_public_ip_on_launch         = true
      assign_ipv6_address_on_creation = false
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
# The network interface's id is given so that its tags can be read.
override_resource {
  target          = aws_instance.server
  override_during = plan
  values = {
    key_name                     = null
    primary_network_interface_id = "eni-00000000000000000"
  }
}

# The group's id, so that a test can say the machine is in this group and in
# no other. `tests/inventory.sh` is what says there is no other group.
override_resource {
  target          = aws_security_group.server
  override_during = plan
  values = {
    id = "sg-00000000000000000"
  }
}

variables {
  site_name    = "causalpulse.example"
  repository   = "owner/name"
  budget_email = "you@example.com"
  ami_id       = "ami-00000000000000000"
  subnet_id    = "subnet-00000000000000000"

  create_github_oidc_provider = true
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

  assert {
    condition     = aws_instance.server.vpc_security_group_ids == toset(["sg-00000000000000000"])
    error_message = "The machine must be in the one security group read above, and in no other."
  }

  assert {
    condition     = aws_security_group.server.vpc_id == "vpc-00000000000000000"
    error_message = "The security group must be in the VPC of the subnet the owner named."
  }
}

run "the_machine_is_where_the_owner_said" {
  command = plan

  assert {
    condition     = data.aws_subnet.server.id == "subnet-00000000000000000" && aws_instance.server.subnet_id == "subnet-00000000000000000"
    error_message = "The machine must be placed in var.subnet_id, which is read and not looked for."
  }

  assert {
    condition     = aws_ebs_volume.data.availability_zone == "eu-west-1a"
    error_message = "The data volume must be created in the subnet's zone."
  }
}

run "a_subnet_that_gives_no_public_address_is_refused" {
  command = plan

  override_data {
    target = data.aws_subnet.server
    values = {
      vpc_id                  = "vpc-00000000000000000"
      availability_zone       = "eu-west-1a"
      map_public_ip_on_launch = false
    }
  }

  expect_failures = [data.aws_subnet.server]
}

run "a_subnet_id_that_is_not_one_is_refused" {
  command = plan

  variables {
    subnet_id = "eu-west-1a"
  }

  expect_failures = [var.subnet_id]
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
    condition     = aws_instance.server.instance_type == "c7i-flex.large"
    error_message = "The instance must be a c7i-flex.large."
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

  assert {
    condition     = aws_volume_attachment.data.stop_instance_before_detaching == true && aws_volume_attachment.data.force_detach != true
    error_message = "The data volume must be detached from a stopped machine, and never by force: the database is on it."
  }
}

run "only_the_data_volume_is_snapshotted" {
  command = plan

  assert {
    condition     = aws_ebs_volume.data.tags["Backup"] == "causalpulse-data"
    error_message = "The data volume must carry the tag the snapshot policy looks for."
  }

  # Volumes, found by tag. With INSTANCE here the policy would look for a
  # machine carrying the tag, find none, and take no snapshot at all.
  assert {
    condition     = aws_dlm_lifecycle_policy.data.policy_details[0].resource_types == tolist(["VOLUME"])
    error_message = "The snapshot policy must target volumes."
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
    condition     = length(regexall("\\{\\{|\\}\\}", local.deploy_script)) == 2 && contains(local.deploy_script_lines, "sha=\"{{ Sha }}\"")
    error_message = "The deploy script must hold one substitution, the line sha=\"{{ Sha }}\", and no other pair of braces."
  }

  assert {
    condition     = local.deploy_document.mainSteps[0].inputs.runCommand == split("\n", trimspace(local.deploy_script))
    error_message = "The document must run the tested script, line for line."
  }
}

# The script is read as text: nothing here runs it. `make check-infra` also
# gives it to `sh -n` and `shellcheck` (`tests/render-deploy-command.sh`).
# What runs it is `tests/test_deploy_command.py` at the repository's root,
# against stand-ins, and `make smoke-deploy`, against a real Docker.
run "a_deploy_returns_its_reason_and_its_status" {
  command = plan

  assert {
    condition     = contains(local.deploy_script_lines, "sh deploy/deploy.sh \"$registry\" \"$sha\"") && length(regexall("deploy\\.sh", join("\n", local.deploy_script_code))) == 1
    error_message = "The deploy script must be run once, with the registry and the SHA, and not by exec: what follows it returns its output."
  }

  assert {
    condition     = contains(local.deploy_script_lines, ") >\"$run\" 2>&1") && contains(local.deploy_script_lines, "run=\"$root/deploy-run.log\"")
    error_message = "Both streams of the whole deploy must go to this run's file."
  }

  assert {
    condition     = length(regexall(">&2|2>[^&]", join("\n", local.deploy_script_code))) == 0
    error_message = "The script must say nothing on standard error, of which Systems Manager returns the first 8,000 characters only."
  }

  assert {
    condition = (
      index(local.deploy_script_lines, ") >\"$run\" 2>&1") + 1 == index(local.deploy_script_lines, "status=$?")
      && index(local.deploy_script_lines, "status=$?") < index(local.deploy_script_lines, "tail -c 20000 \"$run\"")
      && element(local.deploy_script_lines, -1) == "exit \"$status\""
    )
    error_message = "The status must be taken straight after the deploy, the end of this run's output printed, and the status returned last."
  }

  assert {
    condition     = length(regexall("tail -c [0-9]+ \"\\$run\"", local.deploy_script)) == 1 && length(regexall("(^|[^a-z])cat ", join("\n", local.deploy_script_code))) == 1 && contains(local.deploy_script_lines, "cat \"$run\"")
    error_message = "Standard output must carry the last 20,000 bytes of this run and no more: Systems Manager returns the first 24,000 characters."
  }

  assert {
    condition     = contains(local.deploy_script_lines, "} >>\"$log\" || echo \"deploy-command: could not add this run to $log\"") && contains(local.deploy_script_lines, "if [ \"$(wc -c <\"$log\")\" -gt 1048576 ]; then") && contains(local.deploy_script_lines, "tail -c 524288 \"$log\" >\"$log.new\" && mv \"$log.new\" \"$log\" || echo \"deploy-command: could not shorten $log\"")
    error_message = "Each run must be added to the history, and the history cut to its last half once it passes a megabyte."
  }

  assert {
    condition     = aws_ssm_document.deploy.content == jsonencode(local.deploy_document) && local.deploy_document.mainSteps[0].inputs.timeoutSeconds == "1800"
    error_message = "The document's time-out is the only limit on the pulls and the migration: 1,800 seconds, as the workflow's wait and the deploy script's deadlines are sized for."
  }

  assert {
    condition     = contains(local.deploy_script_lines, "docker pull --quiet \"$registry/causalpulse-app:$sha\"")
    error_message = "The document's own pull must be quiet."
  }
}

run "a_machine_whose_first_boot_did_not_finish_is_refused" {
  command = plan

  assert {
    condition = (
      contains(local.deploy_script_lines, "marker=/var/lib/causalpulse/first-boot-done")
      && contains(local.deploy_script_lines, "if [ ! -e \"$marker\" ]; then echo \"deploy-command: the first boot of this machine did not finish ($marker is missing), so nothing was deployed: follow 'The first boot did not finish' in docs/runbook.md\"; exit 76; fi")
    )
    error_message = "A machine without the first boot's marker must be refused with status 76 and a sentence that names the runbook's procedure."
  }

  # Before the lock file, the first thing the script makes, and so before
  # anything under the root.
  assert {
    condition     = index(local.deploy_script_lines, "marker=/var/lib/causalpulse/first-boot-done") < index(local.deploy_script_lines, "exec 9>\"$root/deploy.lock\"") && [for line in local.deploy_script_code : line if strcontains(line, "$marker")][0] == element(local.deploy_script_code, index(local.deploy_script_code, "exec 9>\"$root/deploy.lock\"") - 2)
    error_message = "The marker must be looked for before the lock file is made."
  }

  # The first-boot script writes the same path, on the root disk, last.
  assert {
    condition     = strcontains(local.first_boot, "\nmarker=/var/lib/causalpulse/first-boot-done\n") && endswith(trimspace(local.first_boot), "install -d -m 755 \"$(dirname \"$marker\")\"\ndate -u +%Y-%m-%dT%H:%M:%SZ >\"$marker.new\"\nmv \"$marker.new\" \"$marker\"\nsay \"done\"") && length(regexall("\\$marker", local.first_boot)) == 4
    error_message = "The first-boot script must write the marker the deploy command looks for, as its last act and nowhere else."
  }
}

run "one_deploy_at_a_time_and_nothing_of_the_records_is_removed" {
  command = plan

  assert {
    condition     = index(local.deploy_script_lines, "exec 9>\"$root/deploy.lock\"") + 1 == index(local.deploy_script_lines, "flock -n 9 || { echo \"deploy-command: another deploy is running; this one did nothing\"; exit 75; }") && index(local.deploy_script_lines, "exec 9>\"$root/deploy.lock\"") < index(local.deploy_script_lines, "(")
    error_message = "The lock must be taken before anything is written, and a second deploy must say so on standard output and exit 75."
  }

  # Every line that removes or renames, in order. A new one is a decision.
  assert {
    condition = [for line in local.deploy_script_code : line if length(regexall("(^|[ ;&|])(rm|mv|rmdir|unlink|truncate|find|dd|cp|install|tee|ln)( |$)|>>?[\"/$]", line)) > 0] == [
      "exec 9>\"$root/deploy.lock\"",
      "rm -rf \"$root/release.new\"",
      "if [ -e \"$root/release\" ]; then mv \"$root/release\" \"$root/release.new\"; fi",
      "rm -rf \"$root/release\"",
      "mv \"$root/release.old\" \"$root/release\"",
      "rm -rf \"$root/release.new\"",
      "aws ecr get-login-password --region \"eu-west-1\" | docker login --username AWS --password-stdin \"$registry\" >/dev/null",
      "trap 'docker rm -f \"$id\" >/dev/null' EXIT",
      "docker cp \"$id:/deploy/.\" \"$root/release.new/\"",
      "docker rm -f \"$id\" >/dev/null",
      "if [ -e \"$root/release\" ]; then mv \"$root/release\" \"$root/release.old\"; fi",
      "if [ -e \"$root/release\" ]; then rm -rf \"$root/release\"; fi",
      "mv \"$root/release.new\" \"$root/release\"",
      ") >\"$run\" 2>&1",
      "rm -rf \"$root/release.new\"",
      "if [ -e \"$root/release.old\" ]; then mv \"$root/release.old\" \"$root/release.new\" || echo \"deploy-command: the release directory set aside could not be renamed; the next run puts it back\"; fi",
      "if { [ ! -e \"$root/release\" ] || mv \"$root/release\" \"$root/release.new\"; } && rm -rf \"$root/release\" && mv \"$root/release.old\" \"$root/release\"; then",
      "rm -rf \"$root/release.new\"",
      "} >>\"$run\" 2>&1",
      "' \"$run\" >\"$run.clean\"; then",
      "mv \"$run.clean\" \"$run\"",
      "rm -f \"$run\" \"$run.clean\"",
      "} >>\"$log\" || echo \"deploy-command: could not add this run to $log\"",
      "tail -c 524288 \"$log\" >\"$log.new\" && mv \"$log.new\" \"$log\" || echo \"deploy-command: could not shorten $log\"",
    ]
    error_message = "The script removes, renames or writes over something this list does not hold. It may touch deploy.lock, deploy-run.log, deploy.log and the release directories, and nothing else."
  }

  # `release.old` is whole whenever it exists only because it is made by a
  # rename and leaves by one. `release` itself is removed in three places and
  # no other, each between two renames, where a release cannot be: in the
  # swap, and in the two put-backs, where it has just been renamed away.
  assert {
    condition = (
      length([for line in local.deploy_script_code : line if length(regexall("rm -[rf]+ [^;&|]*release\\.old", line)) > 0]) == 0
      && length(regexall("rm -[rf]+ \"\\$root/release\"", join("\n", local.deploy_script_code))) == 3
      && index(local.deploy_script_lines, "if [ -e \"$root/release\" ]; then mv \"$root/release\" \"$root/release.new\"; fi") + 1 == index(local.deploy_script_lines, "rm -rf \"$root/release\"")
      && index(local.deploy_script_lines, "rm -rf \"$root/release\"") + 1 == index(local.deploy_script_lines, "mv \"$root/release.old\" \"$root/release\"")
      && length(regexall("mv \"\\$root/release\" \"\\$root/release\\.new\"; \\} && rm -rf \"\\$root/release\" && mv \"\\$root/release\\.old\" \"\\$root/release\"; then", join("\n", local.deploy_script_code))) == 1
      && index(local.deploy_script_lines, "if [ -e \"$root/release\" ]; then mv \"$root/release\" \"$root/release.old\"; fi") + 1 == index(local.deploy_script_lines, "if [ -e \"$root/release\" ]; then rm -rf \"$root/release\"; fi")
      && index(local.deploy_script_lines, "if [ -e \"$root/release\" ]; then rm -rf \"$root/release\"; fi") + 1 == index(local.deploy_script_lines, "mv \"$root/release.new\" \"$root/release\"")
    )
    error_message = "release.old must never be removed in place, and release itself only between two renames: the swap's, and each put-back's."
  }

  # The statuses after which the new release directory stays. Every other
  # one puts back the directory that was there.
  assert {
    condition     = [for line in local.deploy_script_code : line if endswith(line, ")") && !strcontains(line, "(")] == ["0 | 30 | 40)", "*)"] && index(local.deploy_script_lines, "status=$?") < index(local.deploy_script_lines, "case $status in")
    error_message = "The new release directory must stay after status 0, 30 or 40 and after no other."
  }

  assert {
    condition     = length(regexall("current-sha|previous-sha|deploy\\.env", local.deploy_script)) == 0 && [for line in local.deploy_script_code : line if strcontains(line, ".env")] == ["if awk -v envfile=\"$root/.env\" -v quote=\"'\" '", "if [ ! -r \"$root/.env\" ]; then echo \"deploy-command: $root/.env is missing or cannot be read\"; fi"]
    error_message = "The script must not name current-sha, previous-sha or deploy.env, and must name the environment file twice: to read it, and to say that it could not be read."
  }

  assert {
    condition     = contains(local.deploy_script_lines, "if (got < 0) exit 1") && contains(local.deploy_script_lines, "while ((got = (getline line < envfile)) > 0) {")
    error_message = "An environment file that cannot be read must fail the search, so that nothing is printed unsearched."
  }

  assert {
    condition     = length(regexall("set -[a-z]*x|xtrace|compose config|printenv|(^|[ ;])env( |$)", join("\n", local.deploy_script_code))) == 0
    error_message = "The script must not trace its commands or print settings."
  }

  assert {
    condition     = index(local.deploy_script_lines, "' \"$run\" >\"$run.clean\"; then") < index(local.deploy_script_lines, "cat \"$run\"") && index(local.deploy_script_lines, "' \"$run\" >\"$run.clean\"; then") < index(local.deploy_script_lines, "tail -c 20000 \"$run\"") && contains(local.deploy_script_lines, "if (length(value) >= 8) {")
    error_message = "The environment file's values must be taken out of the run's output before it is kept or printed."
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

  # The three places the provider's default tags do not reach.
  assert {
    condition     = aws_instance.server.root_block_device[0].tags["Project"] == "causalpulse"
    error_message = "The root disk must be tagged Project = causalpulse."
  }

  assert {
    condition     = { for key, tag in aws_ec2_tag.server_network_interface : key => tag.value } == { Name = "causalpulse-server", Project = "causalpulse" } && alltrue([for key, tag in aws_ec2_tag.server_network_interface : tag.key == key && tag.resource_id == "eni-00000000000000000"])
    error_message = "The machine's network interface must be tagged Project = causalpulse and named."
  }

  # `volume_tags` on the instance would also take charge of the data
  # volume's tags, and remove the one the snapshot policy looks for.
  assert {
    condition     = aws_instance.server.volume_tags == null
    error_message = "The instance must not set volume_tags: the data volume's tags are its own."
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

  # A SHA names one image: a tag that is there cannot be pushed again.
  assert {
    condition     = aws_ecr_repository.app.image_tag_mutability == "IMMUTABLE" && aws_ecr_repository.web.image_tag_mutability == "IMMUTABLE"
    error_message = "Both repositories must have immutable tags."
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
    condition     = aws_budgets_budget.monthly.limit_amount == "100" && aws_budgets_budget.monthly.limit_unit == "USD" && aws_budgets_budget.monthly.time_unit == "MONTHLY"
    error_message = "The budget must be 100 US dollars a month."
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

run "the_outputs_are_what_the_runbook_and_the_workflow_are_filled_in_from" {
  command = plan

  assert {
    condition     = output.site_name == "causalpulse.example" && output.data_volume_id == "vol-00000000000000000" && output.deploy_document_name == "causalpulse-deploy" && output.registry == "000000000000.dkr.ecr.eu-west-1.amazonaws.com"
    error_message = "The outputs must give the site's name, the data volume's id, the deploy document's name and the registry's host."
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

# Read as text: no test here runs the script.
run "the_first_boot_script_waits_and_checks" {
  command = plan

  assert {
    condition     = length(regexall("apt-get", local.first_boot)) == length(regexall("apt-get -o DPkg::Lock::Timeout=300 ", local.first_boot)) && length(regexall("apt-get", local.first_boot)) > 0
    error_message = "Every apt-get must wait for dpkg's lock (-o DPkg::Lock::Timeout=300)."
  }

  # The lists and the packages are one unit, tried again whole, and a fetch
  # of the lists that was cut is an error and not a warning.
  assert {
    condition     = strcontains(local.first_boot, "until apt_get -o APT::Update::Error-Mode=any update -q && apt_get install -y -q \"$@\"; do") && length(regexall("apt_get (-o [A-Za-z:=-]+ )?(update|install)", local.first_boot)) == 2 && length(regexall("(^|\n) *install_packages [a-z]", local.first_boot)) == 2
    error_message = "Every fetch of the package lists and every install must be the one retried unit in install_packages, with APT::Update::Error-Mode=any."
  }

  assert {
    condition     = strcontains(local.first_boot, "signatures=$(wipefs \"$device\") || wipe_status=$?\n        if [ \"$wipe_status\" -ne 0 ]; then\n            die ") && length(regexall("wipefs \"", local.first_boot)) == 1
    error_message = "wipefs's status must be looked at before its silence is taken to mean an empty volume."
  }

  assert {
    condition     = strcontains(local.first_boot, "\"$work/aws/install\" --update >/dev/null") && length(regexall("aws/install", local.first_boot)) == 1
    error_message = "The AWS CLI installer must be run with --update."
  }
}

run "a_site_name_that_could_break_out_of_a_quote_is_refused" {
  command = plan

  variables {
    site_name = "x'; reboot; '"
  }

  expect_failures = [var.site_name]
}
