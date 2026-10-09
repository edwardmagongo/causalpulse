# The one command the pipeline may run on the server. It logs Docker in to
# the registry with the machine's own role, takes the deploy files out of the
# app image for the commit, and runs `deploy/deploy.sh` from them.
#
# `{{ Sha }}` is replaced by Systems Manager, as text, before the shell reads
# the line. That is safe only because the parameter's pattern admits forty
# lowercase hexadecimal digits and nothing else: Systems Manager refuses a
# value that does not match. `tests/security.tftest.hcl` guards the pattern.
# Do not loosen it, and do not add a parameter without one. The value is
# written once, into `sha`, and no other line may hold two braces together.
#
# What a deploy says. Systems Manager hands back the first 24,000 characters
# of standard output and the first 8,000 of standard error, and the deploy
# script says everything on standard error and the reason for a failure last.
# So everything from the registry login to the end of the deploy script goes
# to one file, both streams together; that file is added to
# `/srv/causalpulse/deploy.log`, the history; and the last 20,000 bytes of
# this run are printed on standard output. The command's status is the
# script's (see its header), or that of the step before it that failed.
#
# Before the output is kept or printed, every value in the environment file
# that is eight characters or longer is replaced in it by the name of its
# setting. The deploy script prints the services' own last log lines after a
# failure, and what a service logs is not this file's to promise. A shorter
# value is not looked for: it would be found in ordinary words. If that
# search itself fails, nothing of the run is kept or printed.
#
# The files this touches under `/srv/causalpulse` are `deploy.lock`,
# `deploy-run.log`, `deploy.log` and the three `release` directories. It
# reads `.env` and never writes it, and it does not name `current-sha` or
# `previous-sha`, which are the deploy script's.
#
# `tests/render-deploy-command.sh` prints the script for `sh -n` and
# `shellcheck`. Nothing under `infra/` runs it: that it behaves as described
# was tried by hand with stand-ins for `docker`, `aws` and `flock`, and is
# otherwise unproved until the first deploy.
locals {
  deploy_document_name = "causalpulse-deploy"

  deploy_script = <<-EOT
    set -eu
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    HOME=$${HOME:-/root}
    export PATH HOME
    registry="${local.registry}"
    sha="{{ Sha }}"
    root=/srv/causalpulse
    run="$root/deploy-run.log"
    log="$root/deploy.log"
    # The logs are root's alone.
    umask 077
    exec 9>"$root/deploy.lock"
    flock -n 9 || { echo "deploy-command: another deploy is running; this one did nothing"; exit 75; }
    # A subshell, so that one redirection takes everything. It is not the left
    # side of an `||`, where `set -e` would be switched off inside it.
    set +e
    (
        set -eu
        umask 022
        aws ecr get-login-password --region "${local.region}" | docker login --username AWS --password-stdin "$registry" >/dev/null
        docker pull --quiet "$registry/${local.repository_names.app}:$sha"
        rm -rf "$root/release.new" "$root/release.old"
        mkdir -p "$root/release.new"
        id=$(docker create "$registry/${local.repository_names.app}:$sha")
        trap 'docker rm -f "$id" >/dev/null' EXIT
        docker cp "$id:/deploy/." "$root/release.new/"
        docker rm -f "$id" >/dev/null
        trap - EXIT
        # Two renames, so the path the proxy's Caddyfile is bound from is
        # missing for the moment between them and not for as long as it takes
        # to remove a directory tree.
        if [ -e "$root/release" ]; then mv "$root/release" "$root/release.old"; fi
        mv "$root/release.new" "$root/release"
        rm -rf "$root/release.old"
        cd "$root/release"
        sh deploy/deploy.sh "$registry" "$sha"
    ) >"$run" 2>&1
    status=$?
    set -e
    if awk -v envfile="$root/.env" -v quote="'" '
        BEGIN {
            while ((getline line < envfile) > 0) {
                at = index(line, "=")
                if (at < 2 || line ~ /^[ \t]*#/) continue
                value = substr(line, at + 1)
                first = substr(value, 1, 1)
                if (length(value) > 1 && (first == "\"" || first == quote) && substr(value, length(value)) == first) {
                    value = substr(value, 2, length(value) - 2)
                }
                if (length(value) >= 8) {
                    count++
                    values[count] = value
                    names[count] = substr(line, 1, at - 1)
                }
            }
        }
        {
            for (i = 1; i <= count; i++) {
                kept = ""
                while ((at = index($0, values[i])) > 0) {
                    kept = kept substr($0, 1, at - 1) "[" names[i] " is not shown]"
                    $0 = substr($0, at + length(values[i]))
                }
                $0 = kept $0
            }
            print
        }
    ' "$run" >"$run.clean"; then
        mv "$run.clean" "$run"
    else
        rm -f "$run" "$run.clean"
        echo "deploy-command: this run's output could not be searched for the environment file's values, so none of it is kept or shown; the exit status was $status"
        exit "$status"
    fi
    # The history, kept to about a megabyte: past that, its last half.
    {
        echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) $sha, exit status $status"
        cat "$run"
    } >>"$log" || echo "deploy-command: could not add this run to $log"
    if [ "$(wc -c <"$log")" -gt 1048576 ]; then
        tail -c 524288 "$log" >"$log.new" && mv "$log.new" "$log" || echo "deploy-command: could not shorten $log"
    fi
    tail -c 20000 "$run"
    echo "deploy-command: exit status $status; the whole of this run is in $run, and earlier runs in $log"
    exit "$status"
  EOT

  # The script's lines without their indentation, and those of them that are
  # not comments. The tests read these two; nothing else does.
  deploy_script_lines = [for line in split("\n", trimspace(local.deploy_script)) : trimspace(line)]
  deploy_script_code  = [for line in local.deploy_script_lines : line if !startswith(line, "#")]

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
