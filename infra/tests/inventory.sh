#!/bin/sh
# What `terraform test` cannot see: a test reads the resources it names, and
# a resource it does not name passes it by. This reads the files themselves.
#
#   sh infra/tests/inventory.sh          check; exit 1 and say why on a failure
#   sh infra/tests/inventory.sh --list   print the blocks that are declared
#
# It fails unless all of these hold:
#
# 1. The `resource`, `data` and `provider` blocks in `infra/*.tf` and
#    `infra/bootstrap/*.tf` are exactly the lines of `inventory.txt`, beside
#    this file. Adding a resource means adding its line there, on purpose,
#    where a reviewer sees it.
# 2. Terraform's files are where this looks: no `.tf` or `.tf.json` file
#    elsewhere under `infra/`, no JSON or override file, no `module` block
#    (its resources would not be listed) and no `import` block (it would
#    take charge of something this directory did not create).
# 3. The data volume and the state bucket have `prevent_destroy = true`.
# 4. Each provider has `default_tags`, and the main one is given
#    `local.default_tags`, whose value `security.tftest.hcl` reads. Each
#    provider's region, and the backend's, is written as `eu-west-1`: the
#    backend cannot follow a variable, so nothing else may either.
# 5. Each lock file holds two `h1:` hashes: one for each platform it was
#    locked for (`linux_amd64`, `darwin_arm64`).
# 6. The state bucket refuses a request that is not over TLS, and removes a
#    replaced version after 90 days. `bootstrap/` has no tests of its own.
# 7. The two answers only the owner can give, after a look at the account,
#    have no default: which subnet, and whether the account already has an
#    identity provider for GitHub Actions. The subnet that is named is
#    refused at the plan if it would give the machine an IPv6 address; no
#    test can say so, because the mock that would is one of the texts below.
# 8. None of the texts refused below appears in any file under `infra/`.
#    This file and `inventory.txt` are not read; nor are the lock files,
#    `.terraform/` or a state file.
#
# A line whose first character, after any indentation, is `#` or `//` is a
# comment and is not read. A comment that follows code on its line is read
# as code, and so is a `/* */` comment: a refused text is refused there too.
#
# It reads text and does not understand Terraform. It can be got round by
# someone who sets out to, for one by building a name from pieces. It is for
# the change that is made without thinking, and for the reviewer's eye.
set -eu

LC_ALL=C
export LC_ALL
unset CDPATH

here=$(cd "$(dirname "$0")" && pwd)
infra=$(cd "$here/.." && pwd)
expected=$here/inventory.txt
failed=0

fail() {
    echo "inventory: $*" >&2
    failed=1
}

# code FILE: the file with each comment line emptied, so that the lines keep
# their numbers.
code() {
    sed -E 's%^[[:space:]]*(#|//).*$%%' "$1"
}

# blocks: one line for each resource, data source and provider declared, as
# `<directory> <kind> <type> <name>` (a provider has no name). A line that
# begins like a declaration and is not written in the usual form is printed
# as `unreadable`, which is in no list.
blocks() {
    for dir in "$infra" "$infra/bootstrap"; do
        for file in "$dir"/*.tf; do
            [ -f "$file" ] || continue
            code "$file" | awk -v dir="infra${dir#"$infra"}" -v file="${file#"$infra"/}" '
                /^[ \t]*(resource|data|provider|module|import|ephemeral)([^A-Za-z0-9_]|$)/ {
                    if (match($0, /^(resource|data) "[A-Za-z0-9_]+" "[A-Za-z0-9_-]+" [{][}]?$/)) {
                        split($0, part, "\"")
                        split($0, word, " ")
                        print dir, word[1], part[2], part[4]
                    } else if (match($0, /^provider "[A-Za-z0-9_]+" [{]$/)) {
                        split($0, part, "\"")
                        print dir, "provider", part[2]
                    } else {
                        print dir, "unreadable", file ":" FNR
                    }
                }
            '
        done
    done | sort
}

# block FILE HEADER: the lines of the block that begins with exactly HEADER,
# to the first line that is a closing brace alone (`terraform fmt`, which
# `make check-infra` runs first, puts a top-level one there).
block() {
    code "$1" | awk -v header="$2" '
        $0 == header { inside = 1 }
        inside { print }
        inside && $0 == "}" { exit }
    '
}

if [ "${1:-}" = --list ]; then
    blocks
    exit 0
fi

# 1. The blocks declared are the blocks listed.
if [ ! -f "$expected" ]; then
    fail "$expected is missing"
else
    listed=$(grep -v -E '^[[:space:]]*(#|$)' "$expected" | sort)
    declared=$(blocks)
    if [ "$listed" != "$declared" ]; then
        fail "the blocks declared under infra/ are not the ones listed in tests/inventory.txt"
        tmp=$(mktemp -d)
        printf '%s\n' "$listed" >"$tmp/listed"
        printf '%s\n' "$declared" >"$tmp/declared"
        diff "$tmp/listed" "$tmp/declared" | sed -n -e 's/^< /  listed and not declared: /p' -e 's/^> /  declared and not listed: /p' >&2 || true
        rm -rf "$tmp"
    fi
fi

# 2. Nothing Terraform reads is out of this script's sight.
strays=$(
    find "$infra" -name .terraform -prune -o -type f \( -name '*.tf' -o -name '*.tf.json' -o -name '*.tofu' \) -print |
        while IFS= read -r file; do
            relative=${file#"$infra"/}
            # Each pattern opens with a bracket: without it, the bash that
            # is macOS's `sh` ends the command substitution at the first `)`.
            case $relative in
                (*.tf.json | *.tofu | override.tf | */override.tf | *_override.tf) echo "$relative" ;;
                (*/*/*) echo "$relative" ;;
                (bootstrap/*.tf) ;;
                (*/*) echo "$relative" ;;
            esac
        done
)
if [ -n "$strays" ]; then
    fail "Terraform files where this check does not read them, or of a kind it does not read:"
    printf '%s\n' "$strays" | sed 's/^/  /' >&2
fi
for dir in "$infra" "$infra/bootstrap"; do
    for file in "$dir"/*.tf; do
        [ -f "$file" ] || continue
        found=$(code "$file" | grep -n -E '^[[:space:]]*(module|import|ephemeral)([^A-Za-z0-9_]|$)' || true)
        if [ -n "$found" ]; then
            fail "${file#"$infra"/} declares a module, an import or an ephemeral resource:"
            printf '%s\n' "$found" | sed 's/^/  /' >&2
        fi
    done
done

# 3. Terraform refuses to destroy the two things that hold what cannot be
#    made again.
block "$infra/machine.tf" 'resource "aws_ebs_volume" "data" {' | grep -q -E '^    prevent_destroy = true$' ||
    fail "the data volume (machine.tf, aws_ebs_volume.data) must have prevent_destroy = true"
block "$infra/bootstrap/main.tf" 'resource "aws_s3_bucket" "state" {' | grep -q -E '^    prevent_destroy = true$' ||
    fail "the state bucket (bootstrap/main.tf, aws_s3_bucket.state) must have prevent_destroy = true"

# 4. Everything that takes tags is tagged for the project.
provider=$(block "$infra/versions.tf" 'provider "aws" {')
if ! printf '%s\n' "$provider" | grep -q -E '^  default_tags [{]$' ||
    ! printf '%s\n' "$provider" | grep -q -E '^    tags = local\.default_tags$'; then
    fail "the provider (versions.tf) must have default_tags with tags = local.default_tags"
fi
provider=$(block "$infra/bootstrap/main.tf" 'provider "aws" {')
if ! printf '%s\n' "$provider" | grep -q -E '^  default_tags [{]$' ||
    ! printf '%s\n' "$provider" | grep -q -E '^      Project = "causalpulse"$'; then
    fail "the provider (bootstrap/main.tf) must have default_tags with Project = \"causalpulse\""
fi

for file in versions.tf bootstrap/main.tf; do
    block "$infra/$file" 'provider "aws" {' | grep -q -E '^  region = "eu-west-1"$' ||
        fail "the provider ($file) must have region = \"eu-west-1\", written and not from a variable"
done
block "$infra/versions.tf" 'terraform {' | grep -q -E '^    region[[:space:]]*= "eu-west-1"$' ||
    fail "the backend (versions.tf) must have region = \"eu-west-1\""

# 5. The provider is locked for the platform of a Linux runner and of the
#    owner's machine.
for lock in "$infra/.terraform.lock.hcl" "$infra/bootstrap/.terraform.lock.hcl"; do
    if [ ! -f "$lock" ]; then
        fail "${lock#"$infra"/} is missing"
    elif [ "$(grep -c -E '^[[:space:]]*"h1:' "$lock")" -ne 2 ]; then
        fail "${lock#"$infra"/} must hold two h1: hashes (terraform providers lock -platform=linux_amd64 -platform=darwin_arm64)"
    fi
done

# 6. The state bucket.
policy=$(block "$infra/bootstrap/main.tf" 'resource "aws_s3_bucket_policy" "state" {')
# The dollar sign in what follows is Terraform's, not this shell's.
# shellcheck disable=SC2016
if ! printf '%s\n' "$policy" | grep -q -E '^      Effect    = "Deny"$' ||
    ! printf '%s\n' "$policy" | grep -q -E '^      Principal = "\*"$' ||
    ! printf '%s\n' "$policy" | grep -q -E '^      Action    = "s3:\*"$' ||
    ! printf '%s\n' "$policy" | grep -q -F 'Resource  = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]' ||
    ! printf '%s\n' "$policy" | grep -q -F 'Bool = { "aws:SecureTransport" = "false" }' ||
    [ "$(printf '%s\n' "$policy" | grep -c -E 'Effect[[:space:]]*=')" -ne 1 ]; then
    fail "the state bucket's policy (bootstrap/main.tf) must be one statement that denies every request not made over TLS"
fi
rule=$(block "$infra/bootstrap/main.tf" 'resource "aws_s3_bucket_lifecycle_configuration" "state" {')
if ! printf '%s\n' "$rule" | grep -q -E '^    status = "Enabled"$' ||
    ! printf '%s\n' "$rule" | grep -q -E '^    filter \{\}$' ||
    ! printf '%s\n' "$rule" | grep -q -E '^      noncurrent_days = 90$' ||
    printf '%s\n' "$rule" | grep -q -E '^[[:space:]]*expiration[[:space:]]'; then
    fail "the state bucket's lifecycle rule (bootstrap/main.tf) must remove replaced versions after 90 days, of every object, and never a current one"
fi

# 7. The owner's two answers are asked for, not assumed.
for name in subnet_id create_github_oidc_provider; do
    variable=$(block "$infra/variables.tf" "variable \"$name\" {")
    if [ -z "$variable" ]; then
        fail "variables.tf must declare the variable $name"
    elif printf '%s\n' "$variable" | grep -q -E '^[[:space:]]*default[[:space:]]*='; then
        fail "the variable $name (variables.tf) must have no default: the owner answers it"
    fi
done

block "$infra/network.tf" 'data "aws_subnet" "server" {' | grep -q -E '^      condition     = !self\.assign_ipv6_address_on_creation$' ||
    fail "the subnet (network.tf, data.aws_subnet.server) must have the postcondition !self.assign_ipv6_address_on_creation"

# 8. The texts refused. Each is an extended regular expression, matched
#    without regard to case, with what it means beside it.
#
# refuse REGEX MEANING [ALLOWED]: fail for each line of code under infra/
# that matches REGEX once every match of ALLOWED has been taken out of it.
refuse() {
    regex=$1
    meaning=$2
    allowed=${3:-}
    find "$infra" -name .terraform -prune -o -type f \
        ! -name .terraform.lock.hcl ! -name '*.tfstate' ! -name '*.tfstate.*' \
        ! -path "$here/inventory.sh" ! -path "$here/inventory.txt" -print |
        sort |
        while IFS= read -r file; do
            code "$file" |
                if [ -n "$allowed" ]; then sed -E "s/$allowed//g"; else cat; fi |
                grep -n -i -E -e "$regex" |
                sed "s|^|${file#"$infra"/}:|" || true
        done >"$hits"
    if [ -s "$hits" ]; then
        fail "$meaning:"
        sed 's/^/  /' "$hits" >&2
    fi
}

hits=$(mktemp)
trap 'rm -f "$hits"' EXIT

refuse 'aws_key_pair' \
    'a key pair (the machine has no SSH key)'
refuse 'key_name[[:space:]]*=([^=]|$)' \
    'key_name given a value (only null is allowed)' \
    'key_name[[:space:]]*=[[:space:]]*null[[:space:]]*$'
refuse 'managed_policy_arns|inline_policy' \
    'a policy given to a role inside the role, where the tests do not read it'
refuse 'aws_iam_role_policy_attachment' \
    'a managed policy attached to a role, other than the two that are tested (the server, the snapshot role)' \
    'aws_iam_role_policy_attachments_exclusive|"aws_iam_role_policy_attachment" "(server|backup)"|aws_iam_role_policy_attachment\.(server|backup)([^A-Za-z0-9_]|$)'
refuse 'AdministratorAccess|PowerUserAccess|IAMFullAccess' \
    'one of the policies of AWS that allow nearly everything'
refuse 'StringLike|ArnLike' \
    'a condition that matches by pattern (the trust policy compares with StringEquals)'
refuse 'ipv6[A-Za-z0-9_]*"?[[:space:]]*[=:]([^=]|$)' \
    'something about IPv6 given a value (only 0, [], null and false are allowed)' \
    'ipv6[A-Za-z0-9_]*"?[[:space:]]*[=:][[:space:]]*(0|\[\]|null|false)[[:space:]]*,?[[:space:]]*$'
refuse '::/0' \
    'every IPv6 address'
refuse 'port"?[[:space:]]*[=:][[:space:]]*"?22"?([^0-9]|$)' \
    'port 22 (the machine is reached through Systems Manager)'
refuse 'aws_vpc_security_group_(ingress|egress)_rule|aws_security_group_rule' \
    'a security group rule outside the group (the tests read the rules written inside it)'
refuse 'random_password' \
    'a secret made by Terraform (it would be kept in the state)'

if [ "$failed" -ne 0 ]; then
    echo "inventory: failed" >&2
    exit 1
fi
echo "inventory: the declared blocks are the listed ones, and nothing refused is present"
