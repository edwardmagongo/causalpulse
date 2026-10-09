#!/bin/sh
# Deploy one commit's images on the server.
#
#   sh deploy/deploy.sh <registry> <sha>
#
# Run from a release directory: the files the app image carries under
# `/deploy`, copied out of the image for <sha>. This script is one of them,
# and it works from the directory above its own, wherever it is called from.
#
# It pulls `<registry>/causalpulse-app:<sha>` and
# `<registry>/causalpulse-web:<sha>`, runs the migration, brings the stack up
# on them, waits for every long-running service to be well and to stay so,
# and only then records <sha>. It creates no image itself, and it does not
# log in to the registry: whoever calls it has done that.
#
# From the environment:
#   CP_DEPLOY_ROOT             holds `.env`, `current-sha`, `previous-sha`
#                              and `deploy.env` (default /srv/causalpulse)
#   CP_DEPLOY_PROJECT          the compose project (default causalpulse-prod)
#   CP_DEPLOY_HEALTH_DEADLINE  seconds to wait for health (default 300): a
#                              whole number of at most six digits, with no
#                              leading zero
#
# The deadline bounds the two waits for health and nothing else. The pull,
# the migration (which may wait for a lock) and `compose up` (which waits for
# the services others depend on) have no limit of this script's: whoever
# calls it sets the limit on the whole. The last wait can also end later than
# the deadline, by the one confirming look described at `settle` below.
#
# Exit status:
#    0  every service is well and `current-sha` holds <sha>
#    1  a command failed where this script did not expect one to (`set -e`)
#    2  the arguments or the deadline are not what is described above
#   10  the release directory cannot be entered; a file of the release or the
#       environment file is missing; `current-sha`, `previous-sha` or
#       `deploy.env` is there and is not a file; the Caddyfile's digest could
#       not be computed; or
#       an image could not be pulled or is not on the machine afterwards
#   20  the migration failed, or Postgres was not there for it to run against
#   30  the stack did not come up, or was not well by the deadline
#   40  the stack is well on <sha> and `previous-sha`, `deploy.env` or
#       `current-sha` could not be written
#
# It does not roll back. After 2, 10 or 20 no application service has been
# replaced. After 30 or 40 the stack is as the attempt left it, and
# `current-sha` still holds what it held before.
#
# `deploy.env` is for a person, and this script does not read it. It holds
# the three settings the override needs, as they were for the last deploy
# that ended well, so that Compose can be run by hand from the release
# directory with `--env-file` given twice: the environment file, then this
# one. It holds no secret.
#
# Everything this script says goes to standard error and begins `deploy:`.
set -eu

LC_ALL=C
export LC_ALL

say() {
    echo "deploy: $*" >&2
}

# is_hex VALUE LENGTH: that many lowercase hexadecimal digits and nothing else.
is_hex() {
    case $1 in
        *[!0-9a-f]*) return 1 ;;
    esac
    [ "${#1}" -eq "$2" ]
}

if [ $# -ne 2 ]; then
    say "usage: sh deploy/deploy.sh <registry> <sha>"
    exit 2
fi
registry=$1
sha=$2
# Not empty, not something Docker would read as an option or as a path from
# the root, and no character outside the set.
case $registry in
    '' | -* | /* | *[!A-Za-z0-9._:/-]*)
        say "the registry may hold only letters, digits and . _ : / - and may not begin with - or /"
        exit 2
        ;;
esac
if ! is_hex "$sha" 40; then
    say "the SHA must be 40 lowercase hexadecimal digits"
    exit 2
fi

root=${CP_DEPLOY_ROOT:-/srv/causalpulse}
project=${CP_DEPLOY_PROJECT:-causalpulse-prod}
deadline=${CP_DEPLOY_HEALTH_DEADLINE:-300}
# Refused here, because the shell's arithmetic would stop the script later on
# a number it reads as octal (`08`) or cannot hold.
case $deadline in
    *[!0-9]* | 0?* | ???????*)
        say "CP_DEPLOY_HEALTH_DEADLINE must be a whole number of seconds, at most 999999, with no leading zero"
        exit 2
        ;;
esac

# How long after every service first reads well the confirming look is taken.
# Docker asks a new container for its health `interval` (5 s in
# `docker-compose.prod.yml`) after it starts, and one good answer makes it
# `healthy`; it takes twelve bad ones in a row to undo that. So a process
# that dies after its first check reads `healthy` at every start, and what
# tells is the second look: the same container, not restarted in between, has
# then lived at least `interval` + `settle` seconds. That is 30 s here, the
# `start_period` the same file allows a service to finish starting in. A
# process that dies later than that is not caught by this script.
settle=25

unset CDPATH
if ! cd "$(dirname "$0")/.." 2>/dev/null; then
    say "the release directory above $0 cannot be entered"
    exit 10
fi

for file in docker-compose.prod.yml deploy/compose.server.yml deploy/Caddyfile "$root/.env"; do
    if [ ! -f "$file" ]; then
        say "$file is missing"
        exit 10
    fi
done
for name in current-sha previous-sha deploy.env; do
    if [ -e "$root/$name" ] && [ ! -f "$root/$name" ]; then
        say "$root/$name is not a file"
        exit 10
    fi
done

# The SHA-256 of a file's content, or nothing. `sha256sum` is what the server
# has; `shasum` is what a Mac may have in its place.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        line=$(sha256sum <"$1" 2>/dev/null) || line=
    else
        line=$(shasum -a 256 <"$1" 2>/dev/null) || line=
    fi
    echo "${line%% *}"
}

# The proxy's configuration is one file, mounted from this directory, and
# Compose compares a container's settings, which hold that file's path and
# not its content. The override puts this digest in a label on `proxy`, so
# that a Caddyfile whose content changed is a changed setting.
CP_CADDYFILE_DIGEST=$(sha256_of deploy/Caddyfile)
if ! is_hex "$CP_CADDYFILE_DIGEST" 64; then
    say "the SHA-256 of deploy/Caddyfile could not be computed with sha256sum or shasum"
    exit 10
fi

# record_of NAME: the SHA that file under the root holds, or nothing.
record_of() {
    [ -f "$root/$1" ] || return 0
    held=$(cat "$root/$1" 2>/dev/null) || held=
    if is_hex "$held" 40; then
        echo "$held"
    else
        say "$root/$1 does not hold a SHA and is ignored"
    fi
}

# record NAME VALUE: that file under the root, written whole or not at all.
record() {
    target=${root:?}/${1:?}
    if { printf '%s\n' "$2" >"$target.new"; } 2>/dev/null &&
        mv -f "$target.new" "$target" 2>/dev/null; then
        return 0
    fi
    rm -f "$target.new" 2>/dev/null || true
    return 1
}

# The last SHA recorded, and the one whose images are kept beside this
# deploy's so that going back to it needs no registry: the SHA recorded
# before this one. For a deploy of the SHA already recorded, that is what
# `previous-sha` holds.
recorded=$(record_of current-sha)
if [ "$sha" = "$recorded" ]; then
    before=$(record_of previous-sha)
else
    before=$recorded
fi

app=$registry/causalpulse-app
web=$registry/causalpulse-web

# The override reads these three.
CP_IMAGE_REGISTRY=$registry
CP_IMAGE_TAG=$sha
export CP_IMAGE_REGISTRY CP_IMAGE_TAG CP_CADDYFILE_DIGEST

compose() {
    docker compose -p "$project" --env-file "$root/.env" \
        -f docker-compose.prod.yml -f deploy/compose.server.yml "$@"
}

# look SERVICE: where its container stands, as Docker reports it. One of
#   absent                             the service has no container
#   unknown                            Docker did not answer for it
#   <state> <health> <restarts>/<id>   with `unchecked` for the health of a
#                                      container that has no health check
look() {
    id=$(compose ps -a -q "$1" 2>/dev/null | head -n 1) || id=
    if [ -z "$id" ]; then
        echo absent
        return 0
    fi
    docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}unchecked{{end}} {{.RestartCount}}/{{.Id}}' "$id" 2>/dev/null ||
        echo unknown
}

# wait_healthy CONFIRM SERVICE...: until each service's container is running
# and healthy, or the deadline has passed. With CONFIRM above 0 that is not
# enough: the look taken CONFIRM seconds after one that found them all so
# must find each of them running and healthy again, in the same container,
# with the same count of restarts. That second look is taken even when it
# falls after the deadline.
wait_healthy() {
    confirm=$1
    shift
    give_up=$(($(date +%s) + deadline))
    # What the look before this one saw, when it found every service well.
    settled=
    while :; do
        unwell=
        moved=
        seen=
        for service in "$@"; do
            report=$(look "$service")
            case $report in
                'running healthy '*)
                    mark="$service=${report#running healthy }"
                    seen="$seen $mark"
                    case "$settled " in
                        *" $mark "*) ;;
                        *) moved="$moved $service=restarted-or-replaced" ;;
                    esac
                    ;;
                'running '*)
                    report=${report#running }
                    unwell="$unwell $service=${report%% *}"
                    ;;
                *) unwell="$unwell $service=${report%% *}" ;;
            esac
        done
        if [ -z "$unwell" ]; then
            if [ "$confirm" -eq 0 ] || [ -z "$moved" ]; then
                return 0
            fi
            if [ -z "$settled" ]; then
                settled=$seen
                sleep "$confirm"
                continue
            fi
        fi
        if [ "$(date +%s)" -ge "$give_up" ]; then
            if [ -n "$unwell" ]; then
                say "not healthy after $deadline seconds:$unwell"
            else
                say "not healthy after $deadline seconds:$moved"
            fi
            return 1
        fi
        if [ -z "$unwell" ]; then
            settled=$seen
            sleep "$confirm"
        else
            settled=
            sleep 2
        fi
    done
}

# What a person needs to see after a failure: each container's state and the
# last lines each service wrote.
show_state() {
    say "the state of the stack, and the last 40 lines of each service's log, follow"
    compose ps -a >&2 || true
    compose logs --no-color --tail 40 >&2 || true
}

# 1. The images. Whoever called this has pulled the app image already, to
#    take this script out of it, so this fetches the dashboard's image and,
#    of the app image, nothing. Compose may not treat a failed pull as an
#    error for every service, so each image is also looked for afterwards.
say "pulling the images for $sha"
if ! compose pull migrate api web >&2; then
    say "the images for $sha could not be pulled"
    exit 10
fi
for image in "$app:$sha" "$web:$sha"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        say "$image is not on this machine after the pull"
        exit 10
    fi
done

# 2. The migration, before any service is replaced. It runs as one container
#    of its own, beside the services that are running, and starts and stops
#    no other. Postgres is started here only when it is not running, which is
#    a first deploy or a stack that was stopped; a Postgres that is running
#    is left as it is. One look is enough here: a Postgres that does not stay
#    up fails the migration.
if [ -z "$(compose ps -q --status running postgres 2>/dev/null)" ]; then
    say "postgres is not running: starting it for the migration"
    if ! compose up -d --no-build postgres >&2; then
        say "postgres could not be started, so the migration was not run"
        show_state
        exit 20
    fi
fi
if ! wait_healthy 0 postgres; then
    say "postgres is not healthy, so the migration was not run"
    show_state
    exit 20
fi
say "running the migration"
if ! compose run --rm --no-deps -T --pull never migrate </dev/null >&2; then
    say "the migration failed: no application service has been replaced"
    show_state
    exit 20
fi

# 3. The stack, on the new images. Compose replaces the containers whose
#    image or settings changed and leaves the others running. The proxy's
#    image does not change with the SHA: the proxy is replaced when the
#    Caddyfile's digest, which the override makes one of its settings, is not
#    the one its container was created with, and that is how this release's
#    Caddyfile comes into use, going forward or going back.
say "bringing the stack up on $sha"
if ! compose up -d --no-build --remove-orphans >&2; then
    say "the stack did not come up on $sha"
    show_state
    exit 30
fi
say "waiting for every service to be healthy, and to be so again $settle seconds later"
if ! wait_healthy "$settle" postgres redis api worker web proxy; then
    say "the stack is not healthy on $sha"
    show_state
    exit 30
fi

# 4. The records, each written whole or not at all, and `current-sha` last.
#    A deploy of the SHA already recorded writes neither SHA, so
#    `previous-sha` goes on holding the SHA that was recorded before it.
#    `deploy.env` is written every time: see the top of this file.
if [ "$sha" != "$recorded" ] && [ -n "$recorded" ] && ! record previous-sha "$recorded"; then
    say "the stack is healthy on $sha, and $root/previous-sha could not be written"
    exit 40
fi
if ! record deploy.env "# Written by deploy/deploy.sh when a deploy ends well: what the override
# (deploy/compose.server.yml) needs to name the images and the proxy's label
# of the commit that is running. This file holds no secret.
CP_IMAGE_REGISTRY=$CP_IMAGE_REGISTRY
CP_IMAGE_TAG=$CP_IMAGE_TAG
CP_CADDYFILE_DIGEST=$CP_CADDYFILE_DIGEST"; then
    say "the stack is healthy on $sha, and $root/deploy.env could not be written"
    exit 40
fi
if [ "$sha" != "$recorded" ] && ! record current-sha "$sha"; then
    say "the stack is healthy on $sha, and $root/current-sha could not be written"
    exit 40
fi

# 5. Older images, so the disk does not fill: every tag of the two
#    repositories but this SHA's and the one recorded before it. No other
#    image on the machine is removed. A failure here is said and no more.
for repository in "$app" "$web"; do
    if ! tags=$(docker image ls --format '{{.Tag}}' "$repository"); then
        say "could not list the images of $repository, so none was removed"
        continue
    fi
    for tag in $tags; do
        case $tag in
            "$sha" | "$before" | '<none>') continue ;;
        esac
        docker image rm "$repository:$tag" >/dev/null 2>&1 ||
            say "could not remove $repository:$tag"
    done
done

say "$sha is running"
