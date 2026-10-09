# Who may do what: the pipeline's role, the machine's role. Read from a plan
# against a mocked provider: no credentials, no account, nothing created.
#
# Every policy is built from a local with `jsonencode`, and the tests read
# the local and check that the resource is given exactly it. The identifiers
# AWS would give (ARNs) are supplied below, so that the tests can tell one
# resource from another.

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
  target          = aws_instance.server
  override_during = plan
  values = {
    id  = "i-00000000000000000"
    arn = "arn:aws:ec2:eu-west-1:000000000000:instance/i-00000000000000000"
  }
}

override_resource {
  target          = aws_ssm_document.deploy
  override_during = plan
  values = {
    arn = "arn:aws:ssm:eu-west-1:000000000000:document/causalpulse-deploy"
  }
}

override_resource {
  target          = aws_ecr_repository.app
  override_during = plan
  values = {
    arn = "arn:aws:ecr:eu-west-1:000000000000:repository/causalpulse-app"
  }
}

override_resource {
  target          = aws_ecr_repository.web
  override_during = plan
  values = {
    arn = "arn:aws:ecr:eu-west-1:000000000000:repository/causalpulse-web"
  }
}

# The account's identity provider for GitHub Actions, in each of the two
# ways it can come to be in use. In an account the two would have one ARN,
# since it is made from the address; here they are told apart on purpose.
override_resource {
  target          = aws_iam_openid_connect_provider.github
  override_during = plan
  values = {
    arn = "arn:aws:iam::000000000000:oidc-provider/created-here"
  }
}

override_data {
  target = data.aws_iam_openid_connect_provider.github
  values = {
    arn = "arn:aws:iam::000000000000:oidc-provider/already-there"
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

run "the_pipeline_role_trusts_one_branch_of_one_repository" {
  command = plan

  assert {
    condition     = length(local.pipeline_trust.Statement) == 1
    error_message = "The pipeline role's trust policy must have one statement."
  }

  assert {
    condition     = keys(local.pipeline_trust.Statement[0].Condition) == ["StringEquals"]
    error_message = "The trust policy must compare with StringEquals and nothing else: a pattern operator would let a star match."
  }

  assert {
    condition     = local.pipeline_trust.Statement[0].Condition.StringEquals["token.actions.githubusercontent.com:sub"] == "repo:owner/name:ref:refs/heads/main"
    error_message = "The trusted subject must be exactly repo:<repository>:ref:refs/heads/main."
  }

  assert {
    condition     = local.pipeline_trust.Statement[0].Condition.StringEquals["token.actions.githubusercontent.com:aud"] == "sts.amazonaws.com"
    error_message = "The trusted audience must be sts.amazonaws.com."
  }

  assert {
    condition     = length(local.pipeline_trust.Statement[0].Condition.StringEquals) == 2
    error_message = "The trust policy must test the subject and the audience, and nothing else."
  }

  assert {
    condition     = local.pipeline_trust.Statement[0].Action == "sts:AssumeRoleWithWebIdentity" && local.pipeline_trust.Statement[0].Principal == { Federated = "arn:aws:iam::000000000000:oidc-provider/created-here" }
    error_message = "The role must be assumable through GitHub's OIDC provider only: here, the one this configuration creates."
  }

  assert {
    condition     = length(aws_iam_openid_connect_provider.github) == 1 && length(data.aws_iam_openid_connect_provider.github) == 0
    error_message = "With create_github_oidc_provider = true the provider must be created, and not also read."
  }

  assert {
    condition     = aws_iam_role.pipeline.assume_role_policy == jsonencode(local.pipeline_trust)
    error_message = "The pipeline role must be given exactly the tested trust policy."
  }

  assert {
    condition     = aws_iam_openid_connect_provider.github[0].url == "https://token.actions.githubusercontent.com" && aws_iam_openid_connect_provider.github[0].client_id_list == toset(["sts.amazonaws.com"])
    error_message = "The OIDC provider must be GitHub's, for the audience sts.amazonaws.com."
  }
}

# An account holds one provider for an address, and this account is shared.
run "an_identity_provider_the_account_already_has_is_read_and_not_created" {
  command = plan

  variables {
    create_github_oidc_provider = false
  }

  assert {
    condition     = length(aws_iam_openid_connect_provider.github) == 0 && length(data.aws_iam_openid_connect_provider.github) == 1
    error_message = "With create_github_oidc_provider = false no provider may be created: a second for the same address is refused by AWS, and the first is another project's."
  }

  assert {
    condition     = data.aws_iam_openid_connect_provider.github[0].url == "https://token.actions.githubusercontent.com"
    error_message = "The provider that is read must be GitHub's."
  }

  assert {
    condition     = local.pipeline_trust.Statement[0].Principal == { Federated = "arn:aws:iam::000000000000:oidc-provider/already-there" }
    error_message = "The role must trust the provider the account already has."
  }

  assert {
    condition     = aws_iam_role.pipeline.assume_role_policy == jsonencode(local.pipeline_trust) && local.pipeline_trust.Statement[0].Condition.StringEquals["token.actions.githubusercontent.com:sub"] == "repo:owner/name:ref:refs/heads/main" && keys(local.pipeline_trust.Statement[0].Condition) == ["StringEquals"]
    error_message = "The trust must be the same one branch of one repository whichever provider is in use."
  }
}

run "a_repository_name_with_a_star_is_refused" {
  command = plan

  variables {
    repository = "owner/*"
  }

  expect_failures = [var.repository]
}

run "the_pipeline_may_push_two_images_and_send_one_command" {
  command = plan

  assert {
    condition = alltrue([
      for action in flatten(local.pipeline_policy.Statement[*].Action) : contains([
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeImages",
        "ecr:GetDownloadUrlForLayer",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
      ], action)
    ])
    error_message = "The pipeline policy holds an action outside the list: pushing and reading images, ssm:SendCommand, and reading the result of a command whose id is known."
  }

  assert {
    condition = alltrue([
      for action in flatten(local.pipeline_policy.Statement[*].Action) :
      !strcontains(action, "*") && !strcontains(action, "?") && !startswith(action, "iam:") && !startswith(action, "ec2:") && action != "ssm:StartSession"
    ])
    error_message = "The pipeline policy must hold no wildcard action, no iam: or ec2: action, and no ssm:StartSession."
  }

  assert {
    condition     = alltrue([for statement in local.pipeline_policy.Statement : statement.Effect == "Allow" && !can(statement.NotAction) && !can(statement.NotResource)])
    error_message = "The pipeline policy must be plain Allow statements, with no NotAction or NotResource."
  }

  assert {
    condition = alltrue([
      for statement in local.pipeline_policy.Statement :
      statement.Resource == [aws_ssm_document.deploy.arn, aws_instance.server.arn]
      if contains(statement.Action, "ssm:SendCommand")
    ])
    error_message = "ssm:SendCommand must be allowed on exactly the deploy document and the one instance."
  }

  assert {
    condition     = length([for statement in local.pipeline_policy.Statement : statement if contains(statement.Action, "ssm:SendCommand")]) == 1 && [for statement in local.pipeline_policy.Statement : statement.Action if contains(statement.Action, "ssm:SendCommand")] == [["ssm:SendCommand"]]
    error_message = "ssm:SendCommand must be in one statement, alone."
  }

  assert {
    condition = alltrue([
      for statement in local.pipeline_policy.Statement :
      statement.Resource == [aws_ecr_repository.app.arn, aws_ecr_repository.web.arn]
      if anytrue([for action in statement.Action : startswith(action, "ecr:") && action != "ecr:GetAuthorizationToken"])
    ])
    error_message = "Every repository action must be allowed on exactly the two repositories."
  }

  assert {
    condition = alltrue([
      for statement in local.pipeline_policy.Statement :
      alltrue([for action in statement.Action : contains(["ecr:GetAuthorizationToken", "ssm:GetCommandInvocation"], action)])
      if contains(statement.Resource, "*")
    ])
    error_message = "Only ecr:GetAuthorizationToken and ssm:GetCommandInvocation may be allowed on every resource."
  }

  assert {
    condition     = aws_iam_role_policy.pipeline.policy == jsonencode(local.pipeline_policy)
    error_message = "The pipeline role must be given exactly the tested policy."
  }

  assert {
    condition     = length(aws_iam_role_policy_attachments_exclusive.pipeline.policy_arns) == 0 && aws_iam_role_policies_exclusive.pipeline.policy_names == toset(["push-and-deploy"])
    error_message = "The pipeline role must have no managed policy and no inline policy but the tested one."
  }
}

run "the_machine_may_pull_two_images_and_be_managed" {
  command = plan

  assert {
    condition = alltrue([
      for action in flatten([for statement in local.server_policy.Statement : statement.Action if statement.Effect == "Allow"]) : contains([
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
      ], action)
    ])
    error_message = "The server's policy must allow ECR read actions only."
  }

  assert {
    condition = alltrue([
      for statement in local.server_policy.Statement :
      statement.Resource == [aws_ecr_repository.app.arn, aws_ecr_repository.web.arn] || statement.Action == ["ecr:GetAuthorizationToken"]
      if statement.Effect == "Allow"
    ])
    error_message = "The server's policy must name the two repositories, and every resource only for ecr:GetAuthorizationToken."
  }

  # The managed policy allows these two on every resource, and the account is
  # shared. They are refused whole: every resource, no condition.
  assert {
    condition = [for statement in local.server_policy.Statement : statement if statement.Effect != "Allow"] == [{
      Sid      = "NoParameterStore"
      Effect   = "Deny"
      Action   = ["ssm:GetParameter", "ssm:GetParameters"]
      Resource = ["*"]
    }]
    error_message = "The server's policy must deny ssm:GetParameter and ssm:GetParameters on every resource, without a condition, and deny nothing else."
  }

  assert {
    condition     = alltrue([for statement in local.server_policy.Statement : contains(["Allow", "Deny"], statement.Effect) && !can(statement.NotAction) && !can(statement.NotResource) && !can(statement.Condition) && !can(statement.Principal)])
    error_message = "The server's policy must be plain statements, with no NotAction, NotResource or Condition."
  }

  assert {
    condition     = keys(aws_iam_role_policy_attachment.server) == ["arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"] && alltrue([for arn, attachment in aws_iam_role_policy_attachment.server : attachment.policy_arn == arn && attachment.role == "causalpulse-server"])
    error_message = "The one managed policy attached to the server's role must be AmazonSSMManagedInstanceCore."
  }

  assert {
    condition     = local.server_managed_policies == ["arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"]
    error_message = "The server's one managed policy must be AmazonSSMManagedInstanceCore."
  }

  assert {
    condition     = aws_iam_role_policy_attachments_exclusive.server.policy_arns == toset(local.server_managed_policies) && aws_iam_role_policies_exclusive.server.policy_names == toset(["pull-images"])
    error_message = "The server's role must have the one managed policy and the one inline policy, and no other."
  }

  assert {
    condition     = aws_iam_role_policy.server.policy == jsonencode(local.server_policy) && aws_iam_role_policy.server.role == "causalpulse-server"
    error_message = "The server's role must be given exactly the tested policy."
  }

  assert {
    condition     = aws_iam_role.server.assume_role_policy == jsonencode(local.server_trust) && local.server_trust.Statement[0].Principal == { Service = "ec2.amazonaws.com" }
    error_message = "The server's role must be assumable by EC2 only."
  }

  assert {
    condition     = aws_iam_instance_profile.server.role == "causalpulse-server"
    error_message = "The instance profile must carry the server's role."
  }
}

# The other managed policy in this directory: AWS's own for the service that
# takes the snapshots. `tests/inventory.sh` refuses a third attachment.
run "the_snapshot_role_is_the_snapshot_services_and_no_more" {
  command = plan

  assert {
    condition     = aws_iam_role.backup.assume_role_policy == jsonencode(local.backup_trust) && local.backup_trust.Statement == [{ Effect = "Allow", Principal = { Service = "dlm.amazonaws.com" }, Action = "sts:AssumeRole" }]
    error_message = "The snapshot role must be assumable by Data Lifecycle Manager only."
  }

  assert {
    condition     = aws_iam_role_policy_attachment.backup.policy_arn == "arn:aws:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole" && aws_iam_role_policy_attachment.backup.role == "causalpulse-backup"
    error_message = "The snapshot role's one managed policy must be AWSDataLifecycleManagerServiceRole."
  }
}
