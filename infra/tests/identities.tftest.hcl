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

override_resource {
  target          = aws_iam_openid_connect_provider.github
  override_during = plan
  values = {
    arn = "arn:aws:iam::000000000000:oidc-provider/token.actions.githubusercontent.com"
  }
}

variables {
  site_name    = "causalpulse.example"
  repository   = "owner/name"
  budget_email = "you@example.com"
  ami_id       = "ami-00000000000000000"
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
    condition     = local.pipeline_trust.Statement[0].Action == "sts:AssumeRoleWithWebIdentity" && local.pipeline_trust.Statement[0].Principal == { Federated = aws_iam_openid_connect_provider.github.arn }
    error_message = "The role must be assumable through GitHub's OIDC provider only."
  }

  assert {
    condition     = aws_iam_role.pipeline.assume_role_policy == jsonencode(local.pipeline_trust)
    error_message = "The pipeline role must be given exactly the tested trust policy."
  }

  assert {
    condition     = aws_iam_openid_connect_provider.github.url == "https://token.actions.githubusercontent.com" && aws_iam_openid_connect_provider.github.client_id_list == toset(["sts.amazonaws.com"])
    error_message = "The OIDC provider must be GitHub's, for the audience sts.amazonaws.com."
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
        "ssm:ListCommandInvocations",
      ], action)
    ])
    error_message = "The pipeline policy holds an action outside the list: pushing and reading images, ssm:SendCommand, and reading a command's result."
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
      alltrue([for action in statement.Action : contains(["ecr:GetAuthorizationToken", "ssm:GetCommandInvocation", "ssm:ListCommandInvocations"], action)])
      if contains(statement.Resource, "*")
    ])
    error_message = "Only ecr:GetAuthorizationToken and the two actions that read a command's result may be allowed on every resource."
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
      for action in flatten(local.server_policy.Statement[*].Action) : contains([
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
      ], action)
    ])
    error_message = "The server's policy must hold ECR read actions only."
  }

  assert {
    condition = alltrue([
      for statement in local.server_policy.Statement :
      statement.Effect == "Allow" && (statement.Resource == [aws_ecr_repository.app.arn, aws_ecr_repository.web.arn] || statement.Action == ["ecr:GetAuthorizationToken"])
    ])
    error_message = "The server's policy must name the two repositories, and every resource only for ecr:GetAuthorizationToken."
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
