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
# A machine whose first boot did not finish is refused first, with status 76:
# `deploy/first-boot.sh.tftpl` writes `/var/lib/causalpulse/first-boot-done`
# as its last act, on the root disk, and without that file this touches
# nothing, the lock file included.
#
# The release directory. `/srv/causalpulse/release` holds the files of the
# commit that is running, and the proxy's Caddyfile is mounted from it by
# path: Docker looks the path up again each time the container starts. The
# new commit's files are copied to `release.new`; the directory in place is
# renamed to `release.old` and the new one into its place; and when the
# deploy script has returned, its status decides which of the two stays:
#    0, 30, 40  the new one. At 0 and 40 the stack is on the new commit, and
#               at 30 `compose up` was run with the new files and the stack
#               is as that left it.
#   anything else  the one that was there. At 2, 10 and 20 the script's
#               header says nothing was replaced; a login, pull or copy that
#               failed ended the run before the directories were renamed, so
#               there is nothing to put back; and 1, which the script does
#               not expect to give, is treated as not deployed, as its
#               records treat it. A first deploy has no directory to put
#               back and keeps the new one whatever the status.
# `release.old` is only ever made by renaming `release` and only ever leaves
# by a rename, so it is whole whenever it exists, and `release.new` is what
# may be half-written or half-removed. A run that finds `release.old` at its
# start follows a run that was cut off (a reboot, the time-out below) before
# it decided: it puts `release.old` back, on the reasoning that a run is cut
# off in the pulls or the migration, which have no limit of their own, more
# easily than in the few minutes after them. If that run had in fact
# replaced the stack, the directory then holds the commit before the one
# running, until the next deploy that ends with 0, 30 or 40. A run cut off
# by a reboot is the other way about: the proxy restarts on the files in
# `release` as they stand, and a later deploy of the commit that was running
# does not recreate it, so it keeps the Caddyfile it read at boot until the
# runbook's `docker restart`. None of this
# was tried on the server: `tests/test_deploy_command.py` (beside the
# repository's root) runs this script against stand-ins, each of these
# states among them, and `make smoke-deploy` runs it against a real Docker,
# which is not a Linux server's.
#
# What a deploy says. Systems Manager hands back the first 24,000 characters
# of standard output and the first 8,000 of standard error, and the deploy
# script says everything on standard error and the reason for a failure last.
# So everything from the registry login to the end of the deploy script goes
# to one file, both streams together; that file is added to
# `/srv/causalpulse/deploy.log`, the history; and the last 20,000 bytes of
# this run are printed on standard output, followed after a failure by how
# full the two disks are. The command's status is the script's (see its
# header), or that of the step before it that failed.
#
# Before the output is kept or printed, every value in the environment file
# that is eight characters or longer is replaced in it by the name of its
# setting; the site's name is one of them. The deploy script prints the
# services' own last log lines after a failure, and what a service logs is
# not this file's to promise. A shorter value is not looked for: it would be
# found in ordinary words. If that search fails, or the environment file
# cannot be read, nothing of the run is kept or printed: the status is, and
# whether the file is there. A run that is cut off before the search (a
# reboot, the time-out) leaves `deploy-run.log` as the services wrote it,
# readable by root alone, until the next run writes over it.
#
# The files this touches under `/srv/causalpulse` are `deploy.lock`,
# `deploy-run.log`, `deploy.log` and the three `release` directories. It
# reads `.env` and never writes it, and it does not name `current-sha`,
# `previous-sha` or `deploy.env`, which are the deploy script's.
#
# `tests/render-deploy-command.sh` prints the script, for `sh -n` and
# `shellcheck` and, with other values for the five lines that name the
# search path, the registry, the SHA, the root and the marker, for the tests
# that run it. Those values are put in when the text is rendered: the script
# reads no setting of its own from the environment, so nothing sent through
# Systems Manager can move its root.
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
    marker=/var/lib/causalpulse/first-boot-done
    run="$root/deploy-run.log"
    log="$root/deploy.log"
    if [ ! -e "$marker" ]; then echo "deploy-command: the first boot of this machine did not finish ($marker is missing), so nothing was deployed: follow 'The first boot did not finish' in docs/runbook.md"; exit 76; fi
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
        # What a run that was cut off left: see the top of this file. As in
        # the swap below, whatever Docker made at the path before the second
        # rename is removed; a path made between that `rm` and the `mv` is
        # not (the window is narrowed to that instant, not closed).
        if [ -e "$root/release.old" ]; then
            rm -rf "$root/release.new"
            if [ -e "$root/release" ]; then mv "$root/release" "$root/release.new"; fi
            rm -rf "$root/release"
            mv "$root/release.old" "$root/release"
            echo "deploy-command: an earlier run was cut off; the release directory it had set aside is back in place"
        fi
        rm -rf "$root/release.new"
        aws ecr get-login-password --region "${local.region}" | docker login --username AWS --password-stdin "$registry" >/dev/null
        docker pull --quiet "$registry/${local.repository_names.app}:$sha"
        mkdir -p "$root/release.new"
        id=$(docker create "$registry/${local.repository_names.app}:$sha")
        trap 'docker rm -f "$id" >/dev/null' EXIT
        docker cp "$id:/deploy/." "$root/release.new/"
        docker rm -f "$id" >/dev/null
        trap - EXIT
        # Two renames, so the path the proxy's Caddyfile is bound from is
        # missing for the moment between them and not for as long as it takes
        # to copy or remove a directory tree. A proxy that starts in that
        # moment has Docker make the path as empty directories, which are
        # not a release and would have the second rename move the new files
        # inside them: so whatever is at the path by then is removed (a path
        # made between that `rm` and the `mv` is not).
        if [ -e "$root/release" ]; then mv "$root/release" "$root/release.old"; fi
        if [ -e "$root/release" ]; then rm -rf "$root/release"; fi
        mv "$root/release.new" "$root/release"
        cd "$root/release"
        sh deploy/deploy.sh "$registry" "$sha"
    ) >"$run" 2>&1
    status=$?
    # Which release directory stays: see the top of this file. `release.old`
    # leaves by a rename, never by being removed where it is. Where it is put
    # back, `release` has just been renamed away, so the `rm -rf` between the
    # two renames finds nothing, or what Docker made there in that instant.
    {
        rm -rf "$root/release.new"
        case $status in
            0 | 30 | 40)
                if [ -e "$root/release.old" ]; then mv "$root/release.old" "$root/release.new" || echo "deploy-command: the release directory set aside could not be renamed; the next run puts it back"; fi
                ;;
            *)
                if [ -e "$root/release.old" ]; then
                    if { [ ! -e "$root/release" ] || mv "$root/release" "$root/release.new"; } && rm -rf "$root/release" && mv "$root/release.old" "$root/release"; then
                        echo "deploy-command: the release directory holds again the files it held before this run"
                    else
                        echo "deploy-command: the release directory this run set aside could not be put back; the next run tries again"
                    fi
                fi
                ;;
        esac
        rm -rf "$root/release.new"
    } >>"$run" 2>&1
    set -e
    if awk -v envfile="$root/.env" -v quote="'" '
        BEGIN {
            while ((got = (getline line < envfile)) > 0) {
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
            # A file that could not be opened is not a file with no values.
            if (got < 0) exit 1
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
        if [ ! -r "$root/.env" ]; then echo "deploy-command: $root/.env is missing or cannot be read"; fi
        df -h / "$root" || true
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
    # A full disk is the failure that says least about itself.
    if [ "$status" -ne 0 ]; then df -h / "$root" || true; fi
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
