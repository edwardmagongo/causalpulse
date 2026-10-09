# A snapshot of the data volume each night; the last seven are kept. Data
# Lifecycle Manager starts it within the hour after 03:00 UTC, so it is not
# certain to be finished, or even begun, by 04:00, when an automatic reboot
# may happen. Either way it is a snapshot of a disk in use: what a machine
# that lost power would find. The policy finds the volume by the tag only it
# carries, so the root disk is not snapshotted.
locals {
  backup_trust = {
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dlm.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  }
}

resource "aws_iam_role" "backup" {
  name               = "causalpulse-backup"
  description        = "Data Lifecycle Manager, snapshotting the CausalPulse data volume"
  assume_role_policy = jsonencode(local.backup_trust)
}

resource "aws_iam_role_policy_attachment" "backup" {
  role       = aws_iam_role.backup.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
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
