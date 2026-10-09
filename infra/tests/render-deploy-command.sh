#!/bin/sh
# Print the deploy document's shell script, with example values where
# Terraform and Systems Manager each put one in.
#
#   sh infra/tests/render-deploy-command.sh > deploy-command.sh
#   sh infra/tests/render-deploy-command.sh root=DIR marker=FILE path=DIRS \
#       registry=HOST sha=SHA > deploy-command.sh
#
# It reads the text between `deploy_script = <<-EOT` and `EOT` in
# `infra/deploy-command.tf` and does by hand what the two would do: it does
# not run Terraform. `make check-infra` gives the result to `sh -n` and
# `shellcheck`. It stops if a Terraform expression it does not know is left.
#
# The arguments are for the tests that run the script away from a server
# (`deploy/smoke-deploy.sh`, and `tests/` beside the repository's root). Each
# replaces one whole line of the script, which must be there exactly once:
#   root      where the script keeps its files, for /srv/causalpulse
#   marker    the file the first boot leaves
#   path      the command search path
#   registry  the registry's host
#   sha       the commit
# The script itself reads none of these from its environment, so this is the
# only place they can be changed, and the document Terraform creates holds
# the lines as `infra/deploy-command.tf` has them.
set -eu

unset CDPATH
source=$(cd "$(dirname "$0")/.." && pwd)/deploy-command.tf

example_registry=000000000000.dkr.ecr.eu-west-1.amazonaws.com
example_sha=0123456789abcdef0123456789abcdef01234567

new_root=
new_marker=
new_path=
new_registry=
new_sha=
for argument in "$@"; do
    value=${argument#*=}
    case $value in
        '' | *"'"* | *'
'*)
            echo "render-deploy-command: a value may not be empty or hold a single quote or a new line: ${argument%%=*}" >&2
            exit 2
            ;;
    esac
    case $argument in
        root=*) new_root=$value ;;
        marker=*) new_marker=$value ;;
        path=*) new_path=$value ;;
        registry=*) new_registry=$value ;;
        sha=*) new_sha=$value ;;
        *)
            echo "render-deploy-command: usage: render-deploy-command.sh [root=DIR] [marker=FILE] [path=DIRS] [registry=HOST] [sha=SHA]" >&2
            exit 2
            ;;
    esac
done

# The dollar signs below are Terraform's and sed's, not this shell's.
# shellcheck disable=SC2016
script=$(
    sed -n '/deploy_script = <<-EOT$/,/^  EOT$/p' "$source" |
        sed -e '1d' -e '$d' -e 's/^    //' \
            -e 's/\${local\.registry}/'"$example_registry"'/g' \
            -e 's/\${local\.region}/eu-west-1/g' \
            -e 's/\${local\.repository_names\.app}/causalpulse-app/g' \
            -e 's/{{ Sha }}/'"$example_sha"'/g'
)
if [ -z "$script" ]; then
    echo "render-deploy-command: no script was found in $source" >&2
    exit 1
fi
# What is left of Terraform's is the doubled dollar, which stands for one.
# shellcheck disable=SC2016
if printf '%s\n' "$script" | sed 's/\$\${/x/g' | grep -q -e '\${' -e '%{' -e '{{' -e '}}'; then
    echo "render-deploy-command: the script holds an expression this cannot render" >&2
    exit 1
fi
script=$(printf '%s\n' "$script" | sed 's/\$\${/${/g')

# The five lines that may be replaced, as the script has them.
root_line='root=/srv/causalpulse'
marker_line='marker=/var/lib/causalpulse/first-boot-done'
path_line='PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
registry_line="registry=\"$example_registry\""
sha_line="sha=\"$example_sha\""

# Each must be there once, whether or not it is being replaced this time: a
# script that no longer has it is not what a test that asks for another root
# would be running.
for line in "$root_line" "$marker_line" "$path_line" "$registry_line" "$sha_line"; do
    found=$(printf '%s\n' "$script" | grep -c -x -F -e "$line" || true)
    if [ "$found" -ne 1 ]; then
        echo "render-deploy-command: the script holds the line '$line' $found times, not once" >&2
        exit 1
    fi
done

# replace NAME VALUE LINE: the line, or NAME='VALUE' when a value was given.
# Single quotes, so that no character of a value is the shell's or sed's.
replace() {
    if [ -n "$2" ]; then
        printf "%s='%s'\n" "$1" "$2"
    else
        printf '%s\n' "$3"
    fi
}

printf '%s\n' "$script" | while IFS= read -r line; do
    case $line in
        "$root_line") replace root "$new_root" "$line" ;;
        "$marker_line") replace marker "$new_marker" "$line" ;;
        "$path_line") replace PATH "$new_path" "$line" ;;
        "$registry_line") replace registry "$new_registry" "$line" ;;
        "$sha_line") replace sha "$new_sha" "$line" ;;
        *) printf '%s\n' "$line" ;;
    esac
done
