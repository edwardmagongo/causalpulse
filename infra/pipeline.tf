# The pipeline's identity. GitHub Actions presents a token GitHub signed; AWS
# exchanges it for this role's short-lived credentials. There is no access
# key, and the role can push two images and run one command on one machine.
locals {
  github_oidc_host = "token.actions.githubusercontent.com"

  # An account holds one identity provider for GitHub Actions, and this
  # account is shared. Whether it is created here or read is the owner's
  # answer (`var.create_github_oidc_provider`), given after one look at the
  # account. This is the provider in use, whichever it is.
  github_oidc_provider_arn = (
    var.create_github_oidc_provider
    ? aws_iam_openid_connect_provider.github[0].arn
    : data.aws_iam_openid_connect_provider.github[0].arn
  )

  # One repository, one branch. The comparison in the trust policy is
  # StringEquals, in which a star is a star and matches nothing else.
  #
  # The subject is compared whole. A job that declares an `environment:`
  # presents `repo:<owner>/<name>:environment:<name>` instead, and is
  # refused. That is intended: the deploy job uses no GitHub environment, and
  # the branch is the condition this role rests on. A pull request, a tag,
  # another branch and another repository are refused in the same way.
  pipeline_subject = "repo:${var.repository}:ref:refs/heads/main"

  pipeline_trust = {
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.github_oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.github_oidc_host}:sub" = local.pipeline_subject
          "${local.github_oidc_host}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  }

  pipeline_policy = {
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
        # Pushing an image, and asking whether a tag is already there.
        Sid    = "PushAndReadTheTwoImages"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeImages",
          "ecr:GetDownloadUrlForLayer",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart",
        ]
        Resource = local.repository_arns
      },
      {
        # SendCommand is checked against the document and against the
        # target, so both are named and nothing else is: no other document
        # on this machine, and this document on no other machine.
        Sid      = "SendTheDeployCommand"
        Effect   = "Allow"
        Action   = ["ssm:SendCommand"]
        Resource = [aws_ssm_document.deploy.arn, aws_instance.server.arn]
      },
      {
        # Reading how a command went. The IAM reference lists no resource
        # type for this action, so a narrower resource would deny it. The
        # role can therefore read the output of any run-command in this
        # account and region whose id it is given, not only its own. It is
        # not given `ssm:ListCommandInvocations`, by which it could find the
        # ids of commands it did not send: the workflow waits on the id its
        # own `SendCommand` returned.
        Sid      = "ReadTheCommandResult"
        Effect   = "Allow"
        Action   = ["ssm:GetCommandInvocation"]
        Resource = ["*"]
      },
    ]
  }
}

# Created only where the account has none. A second for the same address is
# refused by AWS part of the way through an apply; and one created here is
# removed by a `terraform destroy` here, from under anything else in the
# account that has come to use it.
resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url            = "https://${local.github_oidc_host}"
  client_id_list = ["sts.amazonaws.com"]
}

# Otherwise the account's own is read, and nothing about it is changed. It
# must already list the audience `sts.amazonaws.com`, which is the one
# GitHub's own action for AWS credentials asks for.
data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1

  url = "https://${local.github_oidc_host}"
}

resource "aws_iam_role" "pipeline" {
  name               = "causalpulse-pipeline"
  description        = "GitHub Actions, deploying CausalPulse from the main branch"
  assume_role_policy = jsonencode(local.pipeline_trust)
}

resource "aws_iam_role_policy" "pipeline" {
  name   = "push-and-deploy"
  role   = aws_iam_role.pipeline.name
  policy = jsonencode(local.pipeline_policy)
}

# Anything else given to the role by other means is removed at the next
# apply: no managed policy, and no inline policy but the one above.
resource "aws_iam_role_policy_attachments_exclusive" "pipeline" {
  role_name   = aws_iam_role.pipeline.name
  policy_arns = []
}

resource "aws_iam_role_policies_exclusive" "pipeline" {
  role_name    = aws_iam_role.pipeline.name
  policy_names = [aws_iam_role_policy.pipeline.name]
}
