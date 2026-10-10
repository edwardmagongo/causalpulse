locals {
  # The first-boot script. The data volume's id is given to it so that it
  # finds that volume and no other.
  first_boot = templatefile("${path.module}/../deploy/first-boot.sh.tftpl", {
    site_name = var.site_name
    volume_id = aws_ebs_volume.data.id
  })

  # The tag only the data volume carries, which the snapshot policy looks for.
  data_volume_backup_tag = {
    Backup = "causalpulse-data"
  }
}

resource "aws_instance" "server" {
  ami           = var.ami_id
  instance_type = "t3.medium"
  subnet_id     = data.aws_subnet.server.id

  vpc_security_group_ids = [aws_security_group.server.id]
  iam_instance_profile   = aws_iam_instance_profile.server.name

  # No key pair is named, so there is no SSH key on the machine. No IPv6
  # address either.
  ipv6_address_count = 0

  # The metadata service answers version 2 only, and its answer does not
  # travel past the machine's own network stack: a container on Docker's
  # bridge is one hop further, and cannot take the instance's role.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  # A burst beyond the earned credits is throttled, not billed.
  credit_specification {
    cpu_credits = "standard"
  }

  # The system and Docker's images. The provider's default tags do not reach
  # this disk, so they are given here. `volume_tags` is not used: it would
  # also claim the tags of the data volume, which is a resource of its own.
  root_block_device {
    volume_type = "gp3"
    volume_size = 30
    encrypted   = true

    tags = merge(local.default_tags, {
      Name = "causalpulse-root"
    })
  }

  user_data                   = local.first_boot
  user_data_replace_on_change = false

  tags = {
    Name = "causalpulse-server"
  }

  # A changed first-boot script or a newer image does not replace a running
  # machine. docs/runbook.md says how to apply either on purpose.
  lifecycle {
    ignore_changes = [user_data, ami]
  }
}

# The provider's default tags do not reach the machine's network interface
# either, which EC2 creates with it. These follow a replaced machine to its
# new interface.
resource "aws_ec2_tag" "server_network_interface" {
  for_each = merge(local.default_tags, {
    Name = "causalpulse-server"
  })

  resource_id = aws_instance.server.primary_network_interface_id
  key         = each.key
  value       = each.value
}

# Docker's named volumes: the database and the proxy's certificates. It is
# separate from the instance so that replacing the instance keeps it, and
# Terraform refuses to destroy it.
resource "aws_ebs_volume" "data" {
  availability_zone = data.aws_subnet.server.availability_zone
  type              = "gp3"
  size              = 20
  encrypted         = true

  tags = merge(local.data_volume_backup_tag, {
    Name = "causalpulse-data"
  })

  lifecycle {
    prevent_destroy = true
  }
}

# The device name is what EC2 is told. On this instance type the volume
# appears as an NVMe device, and the first-boot script finds it by its id.
resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.data.id
  instance_id = aws_instance.server.id

  # Detaching from a running machine would pull the disk from under the
  # database.
  stop_instance_before_detaching = true
}

# EC2 moves the instance to healthy hardware when the host under it fails.
resource "aws_cloudwatch_metric_alarm" "recover" {
  alarm_name        = "causalpulse-server-recover"
  alarm_description = "Recover the CausalPulse server when its host fails the system status check"

  namespace   = "AWS/EC2"
  metric_name = "StatusCheckFailed_System"
  dimensions = {
    InstanceId = aws_instance.server.id
  }

  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1

  alarm_actions = ["arn:${local.partition}:automate:${local.region}:ec2:recover"]
}

# What the machine itself may do: be managed by Systems Manager, and pull the
# two images.
#
# `AmazonSSMManagedInstanceCore` is AWS's own policy and is wider than this
# machine needs. Version 2 of it, as AWS publishes it, allows on every
# resource two actions that read Parameter Store, `ssm:GetParameter` and
# `ssm:GetParameters`, and the account is shared: without the `Deny` below,
# root on this public machine could read the other project's parameters,
# decrypted where they are encrypted under the account's default key. A
# `Deny` wins over any `Allow`, in this policy or the managed one.
#
# The agent works without them. A command and a session reach it through the
# `ssmmessages:` and `ec2messages:` actions. It reads Parameter Store when a
# document or a command names a parameter (`{{ssm:...}}` or
# `{{ssm-secure:...}}`), or when something on the machine (the CloudWatch
# agent, for one) is told to take its configuration from one. The deploy
# document names none, and nothing else here does either.
#
# Those two are every action of that policy that reads Parameter Store. The
# others that read it (`ssm:GetParametersByPath`, `ssm:GetParameterHistory`,
# `ssm:DescribeParameters`) are not allowed by it or by anything else on this
# role, so there is nothing of theirs to deny; a later version of AWS's
# policy could add one, and the `Deny` below would not cover it.
#
# Left as AWS grants them, on every resource: `ssm:GetDocument` and
# `ssm:DescribeDocument` (they belong to how the agent is sent documents to
# run, and by them the machine can read any document in the account),
# `ssm:GetManifest`, `ssm:DescribeAssociation`, `ssm:ListAssociations`,
# `ssm:ListInstanceAssociations`, `ssm:GetDeployablePatchSnapshotForInstance`,
# the `ssm:Put...` and `ssm:Update...` actions by which the agent reports,
# and the `ssmmessages:` and `ec2messages:` actions.
locals {
  server_trust = {
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  }

  server_managed_policies = [
    "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore",
  ]

  server_policy = {
    Version = "2012-10-17"
    Statement = [
      {
        # The registry's login token is not a repository's, and AWS accepts
        # no narrower resource for it.
        Sid      = "RegistryLogin"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      {
        Sid    = "PullTheTwoImages"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Resource = local.repository_arns
      },
      {
        Sid    = "NoParameterStore"
        Effect = "Deny"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
        ]
        Resource = ["*"]
      },
    ]
  }
}

resource "aws_iam_role" "server" {
  name               = "causalpulse-server"
  description        = "The CausalPulse server: Systems Manager, and pulling its two images"
  assume_role_policy = jsonencode(local.server_trust)
}

# The name is from before the policy also denied anything, and is kept: a
# role's inline policy cannot be renamed, only replaced.
resource "aws_iam_role_policy" "server" {
  name   = "pull-images"
  role   = aws_iam_role.server.name
  policy = jsonencode(local.server_policy)
}

resource "aws_iam_role_policy_attachment" "server" {
  for_each   = toset(local.server_managed_policies)
  role       = aws_iam_role.server.name
  policy_arn = each.value
}

# Anything attached to the role by other means is removed at the next apply.
resource "aws_iam_role_policy_attachments_exclusive" "server" {
  role_name   = aws_iam_role.server.name
  policy_arns = local.server_managed_policies
}

resource "aws_iam_role_policies_exclusive" "server" {
  role_name    = aws_iam_role.server.name
  policy_names = [aws_iam_role_policy.server.name]
}

resource "aws_iam_instance_profile" "server" {
  name = "causalpulse-server"
  role = aws_iam_role.server.name
}
