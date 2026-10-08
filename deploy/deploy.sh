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
# on them, waits for every long-running service to be healthy, and only then
# records <sha> as the one running. It creates no image itself, and it does
# not log in to the registry: whoever calls it has done that.
#
# From the environment:
#   CP_DEPLOY_ROOT             holds `.env` and `current-sha`
#                              (default /srv/causalpulse)
#   CP_DEPLOY_PROJECT          the compose project (default causalpulse-prod)
#   CP_DEPLOY_HEALTH_DEADLINE  seconds to wait for health (default 300)
#
# Exit status:
#    0  every service is healthy and `current-sha` holds <sha>
#    2  the arguments or the deadline are not what is described above
#   10  a file of the release, the environment file or an image is missing
#   20  the migration failed, or Postgres was not there for it to run against
#   30  the stack did not come up, or was not healthy by the deadline
#    1  the stack is healthy and the SHA could not be recorded
#
# It does not roll back. After 20 no application service has been replaced.
# After 30 the stack is as the attempt left it, and `current-sha` still names
# the commit that ran before.
#
# Everything this script says goes to standard error and begins `deploy:`.
set -eu

LC_ALL=C
export LC_ALL

say() {
    echo "deploy: $*" >&2
}

# Forty lowercase hexadecimal digits and nothing else.
is_sha() {
    case $1 in
        *[!0-9a-f]*) return 1 ;;
    esac
    [ "${#1}" -eq 40 ]
}

if [ $# -ne 2 ]; then
    say "usage: sh deploy/deploy.sh <registry> <sha>"
    exit 2
fi
registry=$1
sha=$2
case $registry in
    '' | *[!A-Za-z0-9._:/-]*)
        say "the registry may hold only letters, digits and . _ : / -"
        exit 2
        ;;
esac
if ! is_sha "$sha"; then
    say "the SHA must be 40 lowercase hexadecimal digits"
    exit 2
fi

root=${CP_DEPLOY_ROOT:-/srv/causalpulse}
project=${CP_DEPLOY_PROJECT:-causalpulse-prod}
deadline=${CP_DEPLOY_HEALTH_DEADLINE:-300}
case $deadline in
    *[!0-9]*)
        say "CP_DEPLOY_HEALTH_DEADLINE must be a whole number of seconds"
        exit 2
        ;;
esac

unset CDPATH
cd "$(dirname "$0")/.."

for file in docker-compose.prod.yml deploy/compose.server.yml deploy/Caddyfile "$root/.env"; do
    if [ ! -f "$file" ]; then
        say "$file is missing"
        exit 10
    fi
done

# The commit running now, if one was ever recorded. Its images are kept, so
# that going back to it needs no registry.
previous=
if [ -f "$root/current-sha" ]; then
    previous=$(cat "$root/current-sha") || previous=
    if ! is_sha "$previous"; then
        say "$root/current-sha does not hold a SHA and is ignored"
        previous=
    fi
fi

app=$registry/causalpulse-app
web=$registry/causalpulse-web

# The override reads these two.
CP_IMAGE_REGISTRY=$registry
CP_IMAGE_TAG=$sha
export CP_IMAGE_REGISTRY CP_IMAGE_TAG

compose() {
    docker compose -p "$project" --env-file "$root/.env" \
        -f docker-compose.prod.yml -f deploy/compose.server.yml "$@"
}

# The health of a service's container, as Docker reports it.
health() {
    id=$(compose ps -a -q "$1" 2>/dev/null | head -n 1) || id=
    if [ -z "$id" ]; then
        echo absent
        return 0
    fi
    docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}unchecked{{end}}' "$id" 2>/dev/null ||
        echo unknown
}

# wait_healthy SERVICE...: until each is healthy, or the deadline has passed.
wait_healthy() {
    give_up=$(($(date +%s) + deadline))
    while :; do
        pending=
        for service in "$@"; do
            state=$(health "$service")
            [ "$state" = healthy ] || pending="$pending $service=$state"
        done
        [ -n "$pending" ] || return 0
        if [ "$(date +%s)" -ge "$give_up" ]; then
            say "not healthy after $deadline seconds:$pending"
            return 1
        fi
        sleep 2
    done
}

# What a person needs to see after a failure: each container's state and the
# last lines each service wrote.
show_state() {
    say "the state of the stack, and the last 40 lines of each service's log, follow"
    compose ps -a >&2 || true
    compose logs --no-color --tail 40 >&2 || true
}

# 1. The images. Whoever called this has usually pulled them already, to take
#    this script out of one; then this fetches nothing. Compose does not treat
#    a failed pull as an error for every service, so each image is also looked
#    for afterwards.
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
#    is left as it is.
if [ -z "$(compose ps -q --status running postgres 2>/dev/null)" ]; then
    say "postgres is not running: starting it for the migration"
    if ! compose up -d --no-build postgres >&2; then
        say "postgres could not be started, so the migration was not run"
        show_state
        exit 20
    fi
fi
if ! wait_healthy postgres; then
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
#    image or settings changed and leaves the others running.
say "bringing the stack up on $sha"
if ! compose up -d --no-build --remove-orphans >&2; then
    say "the stack did not come up on $sha"
    show_state
    exit 30
fi
if ! wait_healthy postgres redis api worker web proxy; then
    say "the stack is not healthy on $sha"
    show_state
    exit 30
fi

# 4. The record, written whole or not at all.
if ! printf '%s\n' "$sha" >"$root/current-sha.new" || ! mv -f "$root/current-sha.new" "$root/current-sha"; then
    rm -f "$root/current-sha.new"
    say "the stack is healthy on $sha, and $root/current-sha could not be written"
    exit 1
fi

# 5. Older images, so the disk does not fill: every tag of the two
#    repositories but this SHA's and the one that ran before it, and then
#    whatever that left without a tag. A failure here is said and no more.
for repository in "$app" "$web"; do
    if ! tags=$(docker image ls --format '{{.Tag}}' "$repository"); then
        say "could not list the images of $repository, so none was removed"
        continue
    fi
    for tag in $tags; do
        case $tag in
            "$sha" | "$previous" | '<none>') continue ;;
        esac
        docker image rm "$repository:$tag" >/dev/null 2>&1 ||
            say "could not remove $repository:$tag"
    done
done
docker image prune -f >/dev/null 2>&1 || say "could not remove the images left without a tag"

say "$sha is running"
