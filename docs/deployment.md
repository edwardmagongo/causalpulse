# Deployment

Everything needed to put the production stack on one small AWS server and to deploy a commit to
it. **No server is running**: the configuration is written and checked, but nothing is live in
AWS. [`runbook.md`](runbook.md) has the procedures, each marked as not yet carried out.

What this phase added:

- **Terraform under `infra/`**, applied by hand from a developer's machine and never by the
  pipeline: one `c7i-flex.large` in `eu-west-1` with a fixed IPv4 address, ports 80 and 443 open and
  nothing else, no SSH key and no port 22; a 30 GB root disk and a separate 20 GB data volume
  that holds the database, the certificates and the environment file, and that Terraform refuses
  to destroy; two private image repositories; a nightly snapshot of the data volume, seven kept;
  a budget that e-mails at 80% of 100 US dollars a month; a role for the pipeline; and the one
  Systems Manager document the pipeline can run. Its state is in a private S3 bucket made by
  `infra/bootstrap/`. DNS is not in Terraform: the site's one A record is added by hand.
- **A first-boot script** (`deploy/first-boot.sh.tftpl`), run once by cloud-init: it mounts the
  data volume (formatting it only when two probes agree it is empty), places Docker's named
  volumes on it, installs Docker and the AWS CLI, makes a 2 GB swap file, writes the environment
  file if there is none, and leaves Ubuntu's security upgrades on with a reboot at 04:00 UTC.
  Its last act writes a marker file on the root disk, and a deploy is refused on a machine that
  does not have it, so a first boot that failed is found out before a deploy and not by one.
- **A deploy workflow** (`.github/workflows/deploy.yml`), started by hand.
- **A deploy script** (`deploy/deploy.sh`) and a compose override (`deploy/compose.server.yml`).
  The app image now carries, under `/deploy`, the production compose file, the override, the
  Caddyfile and the script, laid out as in the repository; nothing in the application reads that
  directory. A deploy copies those files out of the image it has just pulled, so the server needs
  no access to the repository, and a deploy of an older commit runs that commit's script against
  that commit's files. The script pulls the two images, runs the migration before any service is
  replaced, brings the stack up without building, waits for every service to be healthy and to be
  so again 25 seconds later in the same container, and only then records the commit. It does not
  roll back by itself. The command that wraps it on the server keeps the previous commit's files
  until the script has returned and puts them back when the script replaced nothing, so that the
  proxy's Caddyfile on disk is the running commit's after a failed migration or a failed pull.
- **Image tags that are written once.** Both repositories refuse a second push of a tag they
  have, so a commit's SHA names one image for as long as the image is kept.

## Two gates, and what each proves

```bash
make check-infra    # Terraform, the scripts and the workflows, with no AWS account
make smoke-deploy   # the real deploy script against a throwaway registry; not run in CI
```

`make check-infra` runs Terraform with every source of AWS credentials taken away. It checks
the formatting; compares the resources declared with a committed list
(`infra/tests/inventory.txt`), so that one added or removed is seen by a reviewer, and refuses
certain text anywhere under `infra/` (a key pair, a rule for port 22, a wide AWS policy among
them); validates both configurations; runs 25 tests against a mocked provider
(`infra/tests/*.tftest.hcl`), which assert, among other things, that the only inbound ports are
80 and 443 over IPv4, that the machine has no key and requires version 2 of the metadata
service, that the pipeline's role trusts exactly one repository's `main`, and what each role's
policy names; renders the first-boot script and the deploy document's script and puts both
through `shellcheck`; runs `shellcheck` on the scripts and `actionlint` on the workflows; and
last runs two test files with `uv`: `tests/test_deploy_files.py`, which runs the deploy script
against a stand-in for Docker, and `tests/test_deploy_command.py`, which runs the server's
deploy command against stand-ins for Docker, the AWS CLI and the lock, and the first-boot
script's writing of the environment file twice over. It is the Infrastructure workflow in CI.
What it does not prove: the tests read a plan made against a
mock and the checks read text, so AWS has parsed none of the policies, and nothing has been
created. The first-boot script's steps that need the machine (the data volume, the mounts,
Docker, the swap file) have not been run anywhere.

`make smoke-deploy` builds both images, pushes them to a registry container on the developer's
machine under made-up commit SHAs, and runs the server's own deploy command, as Terraform
would render it, against local Docker: so `deploy/deploy.sh` is taken out of the image by the
lines that take it out on the server. Six cases: a first deploy from nothing comes up
healthy; a second tag whose Caddyfile differs replaces the first, and the proxy is replaced
while Postgres and Redis are not; the first tag deployed again creates no image and serves its
own Caddyfile; a tag whose worker reads healthy and then dies fails with status 30 and leaves
the recorded commit alone; and a deploy of the commit already recorded keeps the previous one's
images, while a failed migration ends with status 20 and leaves every running container the one
it was; and a failed migration from an image whose Caddyfile differs leaves the running commit's
Caddyfile on disk, so that the proxy, restarted, still serves it. What it does not prove:
anything of AWS; the command under Systems Manager, as root, with the server's shell, lock and
registry login (stand-ins answer for the last two); what Docker does with the Caddyfile's mount
at a restart on a Linux server, since Docker Desktop does not mount files as Linux does; or what
Compose does when an image cannot be pulled (that path is tested against a stand-in for Docker
in `tests/test_deploy_files.py`). It needs Docker, uses the ports
of `make smoke-prod`, so the two do not run together, and is not run in CI because it builds
both images and the minutes are counted.

## Starting a deploy

```bash
gh workflow run deploy.yml --ref main -f sha=<the commit's full SHA> -f confirm=<the site's name>
```

Or from the Actions tab: "Deploy", "Run workflow", on `main`. Nothing deploys by itself. The
workflow refuses unless the commit is on `main` and the `CI` workflow succeeded for it as the tip
of a push to `main`. The fast suite and the Infrastructure workflow are asked for in the same
way: where either ran for the commit, it must have succeeded, so a deploy straight after a merge
waits for the fast suite. It builds and pushes the two images only if the registry does not already
have them for that commit, so a rollback, which is the same workflow with an earlier SHA, builds
nothing while the registry still holds the images. A tag is never pushed twice, and the registry
would refuse it: an image that is bad under its SHA is answered by a new commit, or by an
administrator deleting the tag (the runbook has the command). It then sends the deploy document to the
server, waits for it, prints the end of its output, and asks the site three questions from
outside. A deploy interrupts the site for the seconds the services take to restart.

## Who can deploy, and what the pipeline's role can do

The workflow holds no AWS key. GitHub signs a token for the run and AWS exchanges it for the
pipeline role's short-lived credentials. The role's trust accepts one token subject, compared
whole: a run in this repository from the branch `main`. A run from another branch, a tag or a
pull request is refused, whatever its copy of the workflow says. For the same reason the deploy
job declares no GitHub environment (a job with one presents a different subject), and the five
identifiers the workflow reads are repository variables, none of them a secret. The approval is
the hand start from `main` and the site's name typed as a confirmation.

The role can push to and read the two image repositories, send the one deploy document to the
one machine, and read a command's result. It cannot open a session, run any other document or
any other command, or touch any other resource. The document takes one parameter, which must be
forty lowercase hexadecimal digits.

What that amounts to, stated plainly. The deploy document takes `deploy/deploy.sh` out of the
app image for the commit it is given and runs it as root. So whoever can push an app image and
send the command has root on the server: its environment file and the secrets in it, its
database, and the server's own role. The pipeline's role can do both. The role's trust tells
`main` from every other ref, and does not tell the deploy workflow from any other workflow file
on `main`; GitHub enforces no branch protection on this repository. So anyone who can push to
`main` has, in effect, root on the server. That is the intended boundary for a private
repository with one owner. A test holds that the deploy workflow is the only one that asks
GitHub for an identity token and that it is started by hand only, which guards against an
accident and not against a person who can push. If a second person is ever given the right to
push, the boundary has to move first: enforced branch protection on `main`, and the deploy job
in a GitHub environment with required reviewers, with the role's trust changed to that
environment's subject. AWS offers no narrower form of the permission to read a
command's result, so the role can read the output of any run-command in the account and
region whose id it is given. The account is shared with another project.

## Secrets

The database password and the session secret are generated on the server at its first boot, with
`openssl rand -hex 32`, into an environment file on the data volume that only root can read. They
never leave the machine: they are not in Terraform's state, in GitHub, in an image or on a
developer's machine, and the first-boot script prints the names it wrote and never the values.
The deploy document takes every value of that file that is eight characters or longer out of a
deploy's output before the output is kept or returned. API keys are issued by hand in an
interactive session on the server, where a key appears once in the operator's terminal; the
runbook says why a run-command must never be used for it. Terraform's state does hold the
account's number and the budget's e-mail address, which is why its bucket is private.

## What it costs

Approximate, on-demand prices in `eu-west-1`, running all the time. These are
estimates; nothing has been billed yet.

| Item | US dollars a month |
|---|---|
| `c7i-flex.large` | 66 |
| 50 GB of disk | 4.40 |
| Public IPv4 address | 3.65 |
| Snapshots, image storage, the state bucket | under 1.50 |
| Total | about 76 |

Stopped, the machine costs its disks and its address: about 9. It is a `c7i-flex.large` and not
the cheaper `t3.medium` first chosen because the AWS account is on the Free plan, which launches
only the instance types it lists as free-tier eligible; this one has the same 4 GB. While the
account is on that plan the charge is taken from its credits. The budget's warning, at 80% of
100 US dollars, counts the whole AWS account, not this project alone.

## Operating it

`docs/runbook.md` has eleven procedures as commands that can be pasted: the first-time setup,
deploying and rolling back, issuing a key, looking at the server, restoring the data volume from
a snapshot, rotating the two secrets, changing a rate limit, stopping and starting the machine,
replacing the machine, tearing everything down, and what to do when a machine's first boot did
not finish. It also lists what exists and where each
thing is defined, and where to start when something is wrong.
