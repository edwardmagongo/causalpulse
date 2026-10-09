# A snapshot of the data volume each night; the last seven are kept. Data
# Lifecycle Manager starts it within the hour after 03:00 UTC, so it is not
# certain to be finished, or even begun, by 04:00, when an automatic reboot
# may happen. Either way it is a snapshot of a disk in use: what a machine
# that lost power would find. The policy finds the volume by the tag only it
# carries, so the root disk is not snapshotted.
#
# A volume made from one of these snapshots does not carry the tag until it
# is given it, and without the tag it is not snapshotted. Nothing here can
# change that: the policy finds volumes by tag and by nothing else. But the
# tags are Terraform's (`aws_ebs_volume.data` in `machine.tf`), so once the
# restored volume has been imported in the old one's place, the next apply
# puts the tag on it. docs/runbook.md has the restore.
#
# The role is the snapshot service's and this account's only. AWS's own
# policy for the service lets the role create and delete snapshots anywhere
# in the account, the other project's included; it is not narrowed here.
locals {
  # The condition is the one AWS's guide for this service recommends: only a
  # lifecycle policy of this account may have the service assume the role.
  backup_trust = {
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dlm.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = {
        StringEquals = { "aws:SourceAccount" = local.account_id }
      }
    }]
  }

  backup_managed_policy = "arn:${local.partition}:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

resource "aws_iam_role" "backup" {
  name               = "causalpulse-backup"
  description        = "Data Lifecycle Manager, snapshotting the CausalPulse data volume"
  assume_role_policy = jsonencode(local.backup_trust)
}

resource "aws_iam_role_policy_attachment" "backup" {
  role       = aws_iam_role.backup.name
  policy_arn = local.backup_managed_policy
}

# As for the other two roles: anything else given to this one by other means
# is removed at the next apply. It has the one managed policy and no inline
# policy.
resource "aws_iam_role_policy_attachments_exclusive" "backup" {
  role_name   = aws_iam_role.backup.name
  policy_arns = [aws_iam_role_policy_attachment.backup.policy_arn]
}

resource "aws_iam_role_policies_exclusive" "backup" {
  role_name    = aws_iam_role.backup.name
  policy_names = []
}

resource "aws_dlm_lifecycle_policy" "data" {
  description        = "CausalPulse data volume nightly"
  execution_role_arn = aws_iam_role.backup.arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]
    target_tags    = local.data_volume_backup_tag

    schedule {
      name = "nightly"

      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = ["03:00"]
      }

      retain_rule {
        count = 7
      }

      copy_tags = true
      tags_to_add = merge(local.default_tags, {
        Name = "causalpulse-data-nightly"
      })
    }
  }

  # The role must be able to act before the policy first runs.
  depends_on = [aws_iam_role_policy_attachment.backup]
}
