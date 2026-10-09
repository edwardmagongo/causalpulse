# The two images, tagged by commit SHA. A tag is written once: the registry
# refuses a second push of a tag it has, so a SHA names one image for as long
# as the image is kept, and what a rollback goes back to cannot be changed
# after the fact by anything that holds the pipeline's role. The deploy
# workflow never pushes a tag twice: it asks for each image's tag first and
# builds and pushes only the one that is missing, so a run that pushed one
# image and failed on the other is run again and pushes the other.
#
# An image that is bad under its SHA cannot be pushed again by the pipeline,
# whose role may neither overwrite a tag nor delete an image
# (`infra/pipeline.tf`; `tests/identities.tftest.hcl` holds its actions).
# The usual answer is a new commit. To build the same commit again, an
# administrator of the account deletes the tag first
# (`aws ecr batch-delete-image`), and the next deploy of that SHA finds it
# missing and builds it.
locals {
  repository_names = {
    app = "causalpulse-app"
    web = "causalpulse-web"
  }

  repository_arns = [aws_ecr_repository.app.arn, aws_ecr_repository.web.arn]

  # The registry's host, which is the account's and the region's.
  registry = "${local.account_id}.dkr.ecr.${local.region}.${data.aws_partition.current.dns_suffix}"

  # Five images are kept in each repository. The deploy workflow pushes each
  # without build provenance or a bill of materials, so a tag is one manifest
  # and not an index of several.
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
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "web" {
  name                 = local.repository_names.web
  image_tag_mutability = "IMMUTABLE"

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
