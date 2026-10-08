#!/bin/sh
# Print the first-boot script as Terraform renders it, with example values.
#
#   sh infra/tests/render-first-boot.sh > first-boot.sh
#
# It evaluates the template alone, in an empty directory: no provider, no
# state and no credentials are involved. `make check-infra` gives the result
# to `bash -n` and `shellcheck`, which cannot read the template itself.
set -eu

unset CDPATH
template=$(cd "$(dirname "$0")/../../deploy" && pwd)/first-boot.sh.tftpl
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$template" "$work/first-boot.sh.tftpl"
cd "$work"

# The console prints a string in quotes, so the script travels in base64.
# No provider is loaded; the credentials are taken away all the same.
encoded=$(
    echo 'base64encode(templatefile("first-boot.sh.tftpl", {site_name = "causalpulse.example", volume_id = "vol-0123456789abcdef0"}))' |
        env -u AWS_PROFILE -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
            AWS_EC2_METADATA_DISABLED=true AWS_SHARED_CREDENTIALS_FILE=/dev/null AWS_CONFIG_FILE=/dev/null \
            terraform console
)
case $encoded in
    '"'*'"') ;;
    *)
        echo "render-first-boot: terraform did not render the template" >&2
        exit 1
        ;;
esac
printf %s "$encoded" | tr -d '"' | openssl base64 -d -A
