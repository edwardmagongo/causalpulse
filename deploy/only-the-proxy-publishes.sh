#!/bin/sh
# Fail unless `proxy` is the only service of a compose project with a port on
# the host.
#
#   sh deploy/only-the-proxy-publishes.sh <compose project name>
#
# Read from the containers themselves, as Docker created them, and not from
# the compose file: `HostConfig.PortBindings` is what a `ports:` entry becomes.
# A container on the host's own network (`network_mode: host`) has no entry
# there and every port it listens on is the host's, so that fails too, for
# any service, the proxy included.
# Trying the ports from outside would say nothing, because another stack on
# the machine (the demo one) may be answering on them.
set -eu

project=${1:?the name of a compose project}
containers=$(docker ps -a -q --filter "label=com.docker.compose.project=$project")
if [ -z "$containers" ]; then
    echo "only-the-proxy-publishes: no containers in the project $project" >&2
    exit 1
fi

# One line a container: its service, how many of its ports are bound, and the
# network it is on.
# shellcheck disable=SC2086  # the ids are separate arguments
docker inspect -f '{{index .Config.Labels "com.docker.compose.service"}} {{len .HostConfig.PortBindings}} {{.HostConfig.NetworkMode}}' $containers |
    awk -v project="$project" '
        { seen[$1] = 1 }
        $3 == "host" {
            print "only-the-proxy-publishes: " $1 " is on the network of the host" > "/dev/stderr"
            bad = 1
        }
        $1 == "proxy" { proxy = $2 }
        $1 != "proxy" && $2 > 0 {
            print "only-the-proxy-publishes: " $1 " has " $2 " port(s) on the host" > "/dev/stderr"
            bad = 1
        }
        END {
            # Without these a project that never started would pass.
            count = split("proxy web api worker postgres redis", wanted, " ")
            for (i = 1; i <= count; i++) {
                if (!(wanted[i] in seen)) {
                    print "only-the-proxy-publishes: " project " has no container for " wanted[i] > "/dev/stderr"
                    bad = 1
                }
            }
            if (proxy < 1) {
                print "only-the-proxy-publishes: the proxy has no port on the host" > "/dev/stderr"
                bad = 1
            }
            exit bad
        }
    '
