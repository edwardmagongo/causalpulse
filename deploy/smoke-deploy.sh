#!/bin/sh
# The deploy, end to end, on this machine: `make smoke-deploy`.
#
# A throwaway registry stands in for the server's, a temporary directory for
# `/srv/causalpulse`, and the real `deploy/deploy.sh`, taken out of the image
# it deploys, runs against the local Docker. Five cases, in order:
#
#   1. a first deploy, from nothing, brings the stack up healthy;
#   2. a deploy of a second tag, whose Caddyfile has one more response
#      header, replaces the first: the header is served, and the proxy's
#      container is a new one while Postgres's and Redis's are not;
#   3. a deploy of the first tag again works, creates no image, and the
#      header is gone;
#   4. a deploy of a tag whose worker ends eight seconds after each start,
#      long enough for Docker to call it healthy in between, fails with
#      status 30, says which service, leaves the recorded SHA alone, and
#      leaves the proxy's container, whose Caddyfile did not change, the one
#      it was;
#   5. a deploy of the SHA already recorded keeps the images of the SHA
#      recorded before it and leaves the proxy's container the one it was;
#      and a deploy whose migration fails ends with status 20 and leaves
#      every running container the one it was, not restarted.
#
# What is not run here: a pull that fails. The script's status 10 and its
# look for each image after the pull are tested against a stand-in for
# Docker only (`tests/test_deploy_files.py`), so what the real Compose does
# when an image cannot be pulled is not established by this gate.
#
# Its own compose project and its own image names. The proxy is on 8091 and
# 8444, the ports `make smoke-prod` uses, so the two do not run together.
# Nor do two runs of this gate, which share every name: it stops at the
# start if it finds its containers running, and there is no lock beyond
# that, so two runs started in the same few seconds are not kept apart.
# Whatever happens, the project, its volumes, the registry's container, the
# temporary directory and every image name made here are removed at the end,
# and nothing else on the machine is named. What outlasts it is the public
# images it pulled if they were not here: `registry:2`, and the stack's
# Postgres, Redis and Caddy.
#
# Not run in CI: it creates the two images, and the minutes are counted.
set -eu

project=causalpulse-deploy-smoke
http_port=8091
https_port=8444
registry=127.0.0.1:5005
registry_container=$project-registry
# The images as created here, before they are given the registry's names.
local_app=$project-app
local_web=$project-web
# Marks the containers made here that belong to no compose project.
label=causalpulse.gate=deploy-smoke
# The response header that the second tag's Caddyfile adds.
header=X-Smoke-Deploy

say() {
    echo "smoke-deploy: $*" >&2
}

unset CDPATH
cd "$(dirname "$0")/.."

# The settings are read from the file written below, and a name set in this
# shell would be taken in its place.
unset POSTGRES_PASSWORD CP_SESSION_SECRET CP_SITE_ADDRESS CP_HTTP_PORT CP_HTTPS_PORT \
    CP_LIMIT_SIGNIN_FAILURES CP_LIMIT_UPLOADS_PER_DAY CP_LIMIT_ANALYSES_PER_DAY \
    CP_LIMIT_ANALYSES_ACTIVE CP_IMAGE_REGISTRY CP_IMAGE_TAG CP_DEPLOY_ROOT CP_DEPLOY_PROJECT \
    CP_DEPLOY_HEALTH_DEADLINE COMPOSE_PROJECT_NAME COMPOSE_FILE COMPOSE_ENV_FILES

work=

if ! docker info >/dev/null 2>&1; then
    say "Docker is not answering"
    exit 1
fi

# Everything this gate makes in Docker, found by the project's label, this
# gate's label and this gate's names, so that it needs no compose file and
# works on what an interrupted run left.
remove_everything() {
    for id in $(docker ps -a -q --filter "label=com.docker.compose.project=$project") \
        $(docker ps -a -q --filter "label=$label"); do
        docker rm -f -v "$id" >/dev/null 2>&1 || true
    done
    docker rm -f -v "$registry_container" >/dev/null 2>&1 || true
    for volume in $(docker volume ls -q --filter "label=com.docker.compose.project=$project"); do
        docker volume rm "$volume" >/dev/null 2>&1 || true
    done
    for network in $(docker network ls -q --filter "label=com.docker.compose.project=$project"); do
        docker network rm "$network" >/dev/null 2>&1 || true
    done
    for repository in "$registry/causalpulse-app" "$registry/causalpulse-web" "$local_app" "$local_web"; do
        for tag in $(docker image ls --format '{{.Tag}}' "$repository"); do
            [ "$tag" = '<none>' ] || docker image rm "$repository:$tag" >/dev/null 2>&1 || true
        done
    done
    return 0
}

finish() {
    status=$?
    # A second interrupt does not stop the removal half way.
    trap - EXIT
    trap '' INT TERM HUP
    set +e
    say "removing the project, its volumes, the registry and the images made here"
    remove_everything
    [ -z "$work" ] || rm -rf "$work"
    if [ "$status" -eq 0 ]; then
        say "all five cases passed"
    else
        say "FAILED (status $status)"
    fi
    exit "$status"
}

# Before anything is removed, and before the trap that removes: the names
# are shared with every other run of this gate, so a run that is going
# elsewhere would be destroyed by this one's first step and by its last.
if [ -n "$(docker ps -q --filter "label=com.docker.compose.project=$project")" ] ||
    [ -n "$(docker ps -q --filter "label=$label")" ]; then
    say "containers of this gate are running: another run of it is going, or one was killed"
    say "if none is going, remove them and run this again:"
    say "  docker rm -f -v \$(docker ps -q --filter label=com.docker.compose.project=$project)"
    say "  docker rm -f -v \$(docker ps -q --filter label=$label)"
    exit 1
fi

trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# The last 40 lines each of the project's containers wrote.
show_logs() {
    for id in $(docker ps -a -q --filter "label=com.docker.compose.project=$project"); do
        echo "---- $(docker inspect -f '{{.Name}} ({{.Config.Image}}): {{.State.Status}}' "$id" 2>/dev/null)" >&2
        docker logs --tail 40 "$id" >&2 2>&1 || true
    done
}

number=0
failed() {
    say "case $number: FAILED: $*"
    show_logs
    exit 1
}
passed() {
    echo "ok $number: $*"
}

# The running container of a service: its id, or nothing.
container() {
    docker ps -q \
        --filter "label=com.docker.compose.project=$project" \
        --filter "label=com.docker.compose.service=$1" \
        --filter "label=com.docker.compose.oneoff=False" | head -n 1
}

# The image a service's running container was created from, as it was named.
image_of() {
    id=$(container "$1")
    [ -n "$id" ] || return 0
    docker inspect -f '{{.Config.Image}}' "$id"
}

# When a service's running container was last started, as Docker gives it.
started() {
    id=$(container "$1")
    [ -n "$id" ] || return 0
    docker inspect -f '{{.State.StartedAt}}' "$id"
}

# What a record under the root holds: `current-sha` unless another is named.
recorded() {
    [ ! -f "$root/${1:-current-sha}" ] || cat "$root/${1:-current-sha}"
}

# The benchmark study's page, which needs no session: through the proxy,
# over HTTPS, with the certificate of Caddy's own authority accepted.
site_status() {
    curl -sk -o /dev/null -w '%{http_code}' --max-time 10 "https://localhost:$https_port/study" || true
}

# What the site sends in the header the second tag's Caddyfile adds, or
# nothing. Over HTTP/2 the name arrives in lowercase.
header_served() {
    curl -sk -o /dev/null -D - --max-time 10 "https://localhost:$https_port/study" |
        tr -d '\r' | grep -i "^$header:" | sed 's/^[^:]*: *//' || true
}

# The site answers 200, now or within a minute.
site_answers() {
    tries=0
    while [ "$(site_status)" != 200 ]; do
        tries=$((tries + 1))
        [ "$tries" -lt 30 ] || return 1
        sleep 2
    done
}

# deploy SHA [DEADLINE]: what the server's command does. Pull the two images,
# copy the deploy files out of the app image into the release directory, and
# run the script that came out. Leaves the script's status in
# `deploy_status` and what it wrote in `$log`, which is also shown.
deploy() {
    deploy_status=0
    : >"$log"
    say "deploying $1"
    {
        docker pull -q "$registry/causalpulse-app:$1" &&
            docker pull -q "$registry/causalpulse-web:$1" &&
            carrier=$(docker create --label "$label" "$registry/causalpulse-app:$1") &&
            rm -rf "$root/release" &&
            mkdir "$root/release" &&
            docker cp "$carrier:/deploy/." "$root/release" &&
            docker rm "$carrier" >/dev/null
    } || failed "the images for $1 could not be pulled, or the deploy files not copied out of the app image"
    (
        cd "$root/release" &&
            CP_DEPLOY_ROOT=$root CP_DEPLOY_PROJECT=$project CP_DEPLOY_HEALTH_DEADLINE=${2:-300} \
                sh deploy/deploy.sh "$registry" "$1"
    ) >"$log" 2>&1 || deploy_status=$?
    cat "$log"
    say "deploy.sh ended with status $deploy_status"
}

# An image made from the app image with one file replaced, as root, and the
# services' user restored.
#   variant NAME FILE LINE...
variant() {
    name=$1
    file=$2
    shift 2
    quoted=
    for line in "$@"; do
        quoted="$quoted '$line'"
    done
    docker build -q -t "$name" - <<EOF >/dev/null
FROM $local_app:working
USER root
RUN printf '%s\n'$quoted > $file && chmod 755 $file
USER app
EOF
}

publish() {
    docker tag "$1" "$2" && docker push -q "$2" >/dev/null && docker image rm "$2" >/dev/null
}

# ---- Preparation ----

say "removing what an earlier run may have left"
remove_everything

work=$(mktemp -d)
# The real path, which is the one Docker Desktop is allowed to mount: the
# proxy's configuration is mounted from the release directory under it.
work=$(cd "$work" && pwd -P)
root=$work/root
log=$work/deploy.log
mkdir "$root"
(
    umask 077
    {
        echo "CP_SITE_ADDRESS=localhost:$https_port"
        echo "CP_HTTP_PORT=$http_port"
        echo "CP_HTTPS_PORT=$https_port"
        echo "POSTGRES_PASSWORD=$(openssl rand -hex 24)"
        echo "CP_SESSION_SECRET=$(openssl rand -hex 32)"
    } >"$root/.env"
)

# Four made-up SHAs. A is the working tree's images; B is the same with one
# line more in the Caddyfile its app image carries; C and D are broken on
# purpose, each in one place.
sha_a=$(openssl rand -hex 20)
sha_b=$(openssl rand -hex 20)
sha_c=$(openssl rand -hex 20)
sha_d=$(openssl rand -hex 20)

say "starting the registry on $registry"
docker run -d --name "$registry_container" --label "$label" -p "$registry:5000" registry:2 >/dev/null
tries=0
until curl -sf -o /dev/null --max-time 5 "http://$registry/v2/"; do
    tries=$((tries + 1))
    [ "$tries" -lt 30 ] || failed "the registry did not answer on $registry"
    sleep 1
done

say "creating the app and web images from the working tree"
docker build -t "$local_app:working" .
docker build -t "$local_web:working" web

# B: the Caddyfile the image carries, with one response header added after
# the line that sets another. The second command fails the build if the
# first added nothing.
docker build -q -t "$local_app:caddyfile-changed" - <<EOF >/dev/null
FROM $local_app:working
USER root
RUN sed -i '/X-Content-Type-Options/a header $header "changed"' /deploy/deploy/Caddyfile \\
    && grep -q '$header' /deploy/deploy/Caddyfile
USER app
EOF
# C: the migration and the API work, and the worker's process ends eight
# seconds after it starts, each time Docker starts it again. Its health
# check, which does not look at the process, is first run five seconds in
# and passes, so the container reads healthy for the seconds in between: a
# deploy that looked at its health once would take it for well.
variant "$local_app:worker-broken" /app/packages/cp-worker/src/cp_worker/__main__.py \
    'import time' 'time.sleep(8)' \
    'raise SystemExit("smoke-deploy: this worker is broken on purpose")'
# D: the migration's command, in the image's virtual environment, replaced by
# one that fails.
variant "$local_app:migration-broken" /app/.venv/bin/cp-api-migrate \
    '#!/bin/sh' 'echo "smoke-deploy: this migration fails on purpose" >&2' 'exit 1'

say "pushing four tags of each image, and removing them here so that each deploy pulls"
publish "$local_app:working" "$registry/causalpulse-app:$sha_a"
publish "$local_app:caddyfile-changed" "$registry/causalpulse-app:$sha_b"
publish "$local_app:worker-broken" "$registry/causalpulse-app:$sha_c"
publish "$local_app:migration-broken" "$registry/causalpulse-app:$sha_d"
for sha in "$sha_a" "$sha_b" "$sha_c" "$sha_d"; do
    publish "$local_web:working" "$registry/causalpulse-web:$sha"
done

# ---- 1. A first deploy, from nothing ----
number=1
deploy "$sha_a"
[ "$deploy_status" -eq 0 ] || failed "the first deploy ended with status $deploy_status, not 0"
for file in docker-compose.prod.yml deploy/compose.server.yml deploy/Caddyfile deploy/deploy.sh; do
    cmp -s "$file" "$root/release/$file" ||
        failed "$file out of the image is not the working tree's"
done
[ "$(recorded)" = "$sha_a" ] || failed "current-sha holds '$(recorded)', not A ($sha_a)"
for service in api worker; do
    [ "$(image_of "$service")" = "$registry/causalpulse-app:$sha_a" ] ||
        failed "$service runs '$(image_of "$service")', not the app image for A"
done
[ "$(image_of web)" = "$registry/causalpulse-web:$sha_a" ] ||
    failed "web runs '$(image_of web)', not the web image for A"
site_answers || failed "https://localhost:$https_port/study answered $(site_status), not 200"
[ -z "$(header_served)" ] || failed "the site sends $header on A, whose Caddyfile does not set it"
proxy_first=$(container proxy)
postgres_first=$(container postgres)
redis_first=$(container redis)
[ -n "$proxy_first" ] && [ -n "$postgres_first" ] && [ -n "$redis_first" ] ||
    failed "the proxy, Postgres or Redis has no running container after the first deploy"
passed "a first deploy from nothing is healthy, on files taken out of the image, and records A"

# ---- 2. A second tag, with a changed Caddyfile, replaces the first ----
number=2
deploy "$sha_b"
[ "$deploy_status" -eq 0 ] || failed "the deploy of B ended with status $deploy_status, not 0"
if [ "$(grep -c "$header" "$root/release/deploy/Caddyfile")" -ne 1 ] ||
    ! grep -v "$header" "$root/release/deploy/Caddyfile" | cmp -s - deploy/Caddyfile; then
    failed "B's Caddyfile is not the working tree's with one line more"
fi
for file in docker-compose.prod.yml deploy/compose.server.yml deploy/deploy.sh; do
    cmp -s "$file" "$root/release/$file" || failed "$file out of B's image is not the working tree's"
done
[ "$(recorded)" = "$sha_b" ] || failed "current-sha holds '$(recorded)', not B ($sha_b)"
[ "$(recorded previous-sha)" = "$sha_a" ] ||
    failed "previous-sha holds '$(recorded previous-sha)', not A ($sha_a)"
[ "$(image_of api)" = "$registry/causalpulse-app:$sha_b" ] ||
    failed "api runs '$(image_of api)', not the app image for B"
[ "$(image_of web)" = "$registry/causalpulse-web:$sha_b" ] ||
    failed "web runs '$(image_of web)', not the web image for B"
site_answers || failed "the site answered $(site_status), not 200, after the deploy of B"
[ "$(header_served)" = changed ] ||
    failed "the site sends '$(header_served)' in $header, not 'changed': B's Caddyfile is not in use"
proxy_second=$(container proxy)
[ -n "$proxy_second" ] && [ "$proxy_second" != "$proxy_first" ] ||
    failed "the proxy's container is '$proxy_second', and was '$proxy_first' before the Caddyfile changed"
# Their images and settings are the same in B, so nothing replaces them.
[ "$(container postgres)" = "$postgres_first" ] ||
    failed "Postgres's container was replaced by a deploy that changed nothing of its own"
[ "$(container redis)" = "$redis_first" ] ||
    failed "Redis's container was replaced by a deploy that changed nothing of its own"
passed "a deploy of a second tag replaces the first, records B, and serves B's Caddyfile from a new proxy"

# ---- 3. The first tag again: a rollback, and no image is created ----
number=3
deploy "$sha_a"
[ "$deploy_status" -eq 0 ] || failed "the deploy of A again ended with status $deploy_status, not 0"
if grep -E -i -q 'building|load build definition|exporting to image|\[[a-z]+ +[0-9]+/[0-9]+\]' "$log"; then
    failed "the deploy's output shows an image being created"
fi
[ "$(recorded)" = "$sha_a" ] || failed "current-sha holds '$(recorded)', not A ($sha_a)"
[ "$(image_of api)" = "$registry/causalpulse-app:$sha_a" ] ||
    failed "api runs '$(image_of api)', not the app image for A"
[ "$(recorded previous-sha)" = "$sha_b" ] ||
    failed "previous-sha holds '$(recorded previous-sha)', not B ($sha_b)"
site_answers || failed "the site answered $(site_status), not 200, after going back to A"
[ -z "$(header_served)" ] ||
    failed "the site still sends $header after going back to A: A's Caddyfile is not in use"
proxy_third=$(container proxy)
[ -n "$proxy_third" ] && [ "$proxy_third" != "$proxy_second" ] ||
    failed "the proxy's container is '$proxy_third', and was '$proxy_second' before the Caddyfile changed back"
passed "a deploy of the first tag again works, creates no image, records A and serves A's Caddyfile"

# ---- 4. A worker that does not stay up: fails at the health wait ----
number=4
deploy "$sha_c" 60
[ "$deploy_status" -eq 30 ] || failed "the deploy of the broken C ended with status $deploy_status, not 30"
grep -q '^deploy: not healthy after 60 seconds:.* worker=' "$log" ||
    failed "the deploy did not say that the worker was not healthy"
[ "$(recorded)" = "$sha_a" ] || failed "current-sha holds '$(recorded)' after a failed deploy, not A ($sha_a)"
site_answers || failed "the site answered $(site_status), not 200, after the failed deploy"
# C's images are new and its Caddyfile is A's.
[ "$(container proxy)" = "$proxy_third" ] ||
    failed "the proxy's container was replaced by a deploy whose Caddyfile did not change"
# What makes this case the one it claims to be: between two of its ends the
# worker does read running and healthy, and the deploy refused it all the
# same. Docker waits longer before each new start, up to a minute.
tries=0
until [ "$(docker inspect -f '{{.State.Status}} {{.State.Health.Status}}' "$(container worker)" 2>/dev/null)" = 'running healthy' ]; do
    tries=$((tries + 1))
    [ "$tries" -lt 150 ] ||
        failed "the broken worker never read healthy in 150 seconds, so this case did not test a worker that is healthy and does not stay up"
    sleep 1
done
passed "a deploy whose worker reads healthy and does not stay up ends with status 30, names the worker, and leaves A recorded"

# ---- 5. The recorded SHA again keeps the previous images; a failed migration changes nothing ----
number=5
# First back to a healthy stack, as a person would after case 4. `current-sha`
# is A already and `previous-sha` is B, so this is a deploy of the SHA
# already recorded: A's images and B's stay, and C's go.
deploy "$sha_a"
[ "$deploy_status" -eq 0 ] || failed "the deploy of A after the broken C ended with status $deploy_status, not 0"
site_answers || failed "the site answered $(site_status), not 200, before the failing migration"
[ "$(recorded)" = "$sha_a" ] || failed "current-sha holds '$(recorded)', not A ($sha_a)"
[ "$(recorded previous-sha)" = "$sha_b" ] ||
    failed "previous-sha holds '$(recorded previous-sha)' after a deploy of the SHA already recorded, not B ($sha_b)"
for repository in "$registry/causalpulse-app" "$registry/causalpulse-web"; do
    if docker image inspect "$repository:$sha_c" >/dev/null 2>&1; then
        failed "$repository for C is still here after a deploy that should have removed it"
    fi
    docker image inspect "$repository:$sha_a" >/dev/null 2>&1 ||
        failed "$repository for A, which is running, was removed"
    docker image inspect "$repository:$sha_b" >/dev/null 2>&1 ||
        failed "$repository for B, the SHA recorded before A, was removed"
done
[ "$(container proxy)" = "$proxy_third" ] ||
    failed "the proxy's container was replaced by a deploy whose Caddyfile did not change"
before=
for service in postgres redis api worker web proxy; do
    id=$(container "$service")
    [ -n "$id" ] || failed "$service has no running container before the failing migration"
    before="$before $service=$id@$(started "$service")"
done
deploy "$sha_d"
[ "$deploy_status" -eq 20 ] || failed "the deploy of D ended with status $deploy_status, not 20"
grep -q 'smoke-deploy: this migration fails on purpose' "$log" ||
    failed "the deploy failed before the migration's own command ran"
after=
for service in postgres redis api worker web proxy; do
    after="$after $service=$(container "$service")@$(started "$service")"
done
[ "$after" = "$before" ] ||
    failed "the running containers changed or were restarted: before:$before after:$after"
[ "$(recorded)" = "$sha_a" ] || failed "current-sha holds '$(recorded)' after a failed migration, not A ($sha_a)"
site_answers || failed "the site answered $(site_status), not 200, after the failed migration"
passed "a deploy of the recorded SHA keeps the previous one's images, and a failed migration ends with status 20 and leaves every running container the one it was"
