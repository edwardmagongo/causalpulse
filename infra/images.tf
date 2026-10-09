# The two images, tagged by commit SHA. Tags can be written again: a deploy
# that is run a second time for the same commit pushes the same tag.
locals {
  repository_names = {
    app = "causalpulse-app"
    web = "causalpulse-web"
  }

  repository_arns = [aws_ecr_repository.app.arn, aws_ecr_repository.web.arn]

  # The registry's host, which is the account's and the region's.
  registry = "${local.account_id}.dkr.ecr.${local.region}.${data.aws_partition.current.dns_suffix}"

  # Five images are kept in each repository. Not established: whether a push
  # that carries build provenance, which is an index and several manifests,
  # counts as one image here or as more. If it counts as more, fewer than
  # five deploys are kept. To be looked at after the first two deploys.
  image_lifecycle = {
    rules = [{
      rulePriority = 1
      description  = "Keep the five most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  }
}

resource "aws_ecr_repository" "app" {
  name                 = local.repository_names.app
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "web" {
  name                 = local.repository_names.web
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy     = jsonencode(local.image_lifecycle)
}

resource "aws_ecr_lifecycle_policy" "web" {
  repository = aws_ecr_repository.web.name
  policy     = jsonencode(local.image_lifecycle)
}
