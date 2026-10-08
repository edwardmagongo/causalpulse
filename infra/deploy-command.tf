# The one command the pipeline may run on the server. It logs Docker in to
# the registry with the machine's own role, takes the deploy files out of the
# app image for the commit, and runs `deploy/deploy.sh` from them.
#
# `{{ Sha }}` is replaced by Systems Manager, as text, before the shell reads
# the line. That is safe only because the parameter's pattern admits forty
# lowercase hexadecimal digits and nothing else: Systems Manager refuses a
# value that does not match. `tests/security.tftest.hcl` guards the pattern.
# Do not loosen it, and do not add a parameter without one.
locals {
  deploy_document_name = "causalpulse-deploy"

  deploy_script = <<-EOT
    set -eu
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    HOME=$${HOME:-/root}
    export PATH HOME
    registry="${local.registry}"
    root=/srv/causalpulse
    exec 9>"$root/deploy.lock"
    flock -n 9 || { echo "another deploy is running" >&2; exit 75; }
    aws ecr get-login-password --region "${local.region}" | docker login --username AWS --password-stdin "$registry" >/dev/null
    docker pull "$registry/${local.repository_names.app}:{{ Sha }}"
    rm -rf "$root/release.new"
    mkdir -p "$root/release.new"
    id=$(docker create "$registry/${local.repository_names.app}:{{ Sha }}")
    trap 'docker rm -f "$id" >/dev/null' EXIT
    docker cp "$id:/deploy/." "$root/release.new/"
    docker rm -f "$id" >/dev/null
    trap - EXIT
    rm -rf "$root/release"
    mv "$root/release.new" "$root/release"
    cd "$root/release"
    exec sh deploy/deploy.sh "$registry" "{{ Sha }}"
  EOT

  deploy_document = {
    schemaVersion = "2.2"
    description   = "Deploy one commit of CausalPulse from its images"
    parameters = {
      Sha = {
        type           = "String"
        description    = "The commit to deploy: forty lowercase hexadecimal digits."
        allowedPattern = "^[0-9a-f]{40}$"
      }
    }
    mainSteps = [{
      action = "aws:runShellScript"
      name   = "deploy"
      inputs = {
        # Two image pulls, the migration, and the deploy script's 300
        # seconds of waiting for health, with room to spare.
        timeoutSeconds = "1800"
        runCommand     = split("\n", trimspace(local.deploy_script))
      }
    }]
  }
}

resource "aws_ssm_document" "deploy" {
  name            = local.deploy_document_name
  document_type   = "Command"
  document_format = "JSON"
  target_type     = "/AWS::EC2::Instance"
  content         = jsonencode(local.deploy_document)
}
