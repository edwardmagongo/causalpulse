#!/bin/sh
# Print the deploy document's shell script, with example values where
# Terraform and Systems Manager each put one in.
#
#   sh infra/tests/render-deploy-command.sh > deploy-command.sh
#
# It reads the text between `deploy_script = <<-EOT` and `EOT` in
# `infra/deploy-command.tf` and does by hand what the two would do: it does
# not run Terraform. `make check-infra` gives the result to `sh -n` and
# `shellcheck`. It stops if a Terraform expression it does not know is left.
set -eu

unset CDPATH
source=$(cd "$(dirname "$0")/.." && pwd)/deploy-command.tf

# The dollar signs below are Terraform's and sed's, not this shell's.
# shellcheck disable=SC2016
script=$(
    sed -n '/deploy_script = <<-EOT$/,/^  EOT$/p' "$source" |
        sed -e '1d' -e '$d' -e 's/^    //' \
            -e 's/\${local\.registry}/000000000000.dkr.ecr.eu-west-1.amazonaws.com/g' \
            -e 's/\${local\.region}/eu-west-1/g' \
            -e 's/\${local\.repository_names\.app}/causalpulse-app/g' \
            -e 's/{{ Sha }}/0123456789abcdef0123456789abcdef01234567/g'
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
printf '%s\n' "$script" | sed 's/\$\${/${/g'
