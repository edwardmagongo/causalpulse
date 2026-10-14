# Runbook: the CausalPulse server

How to create, deploy to, look at, repair and remove the one small AWS server that runs the
demo. The design is `docs/specs/2026-10-08-hosting-phase3-server-and-deploy-design.md`; this
document is the commands.

> **Nothing here has been carried out yet.** At the time of writing the server does not exist:
> Terraform has not been applied and the workflow has never run. Every procedure below was
> written from the configuration and the scripts, and each carries a note, "Not yet carried
> out." A note is removed when its procedure has been done on the real server and corrected
> where it was wrong. Until then, read a command before pasting it.

## How to read this

- **Where a command runs.** A block is run on the development Mac, at the root of the
  repository, unless the text above it says "in a session", which means typed into an
  interactive Systems Manager session on the server (procedure 3 shows how one is opened).
  There is no SSH: no port 22 and no key pair.
- **Values.** Each procedure sets the shell variables it needs at its top, from Terraform's
  outputs where there is one. A value in angle brackets, `<like this>`, is yours to supply.
  No account number, instance id or address is written in this document, and none should be
  added to it.
- **The region** is `eu-west-1`, written in `infra/versions.tf` and in the workflow. The
  commands name it explicitly, so they do not depend on the profile's default.
- **The AWS profile** is the one Terraform is applied with. `aws sts get-caller-identity` says
  who you are before anything is changed.
- **A run-command is never used for anything secret.** See procedure 3.
- **"Unverified until carried out."** These words mark a statement that rests on how a tool is
  understood to behave and that no documentation was found to settle. A statement without them
  was read from the repository's own files or from the tool's published documentation or help
  text. Neither kind has been tried on the server: that is what "Not yet carried out." says of
  a whole procedure.

## What exists and where

Everything in AWS is tagged `Project = causalpulse` where the resource takes tags.

| Resource | What it is for | Where it is defined |
|---|---|---|
| State bucket (S3) | Holds Terraform's state for `infra/`. Versioned, encrypted, private, TLS only; a replaced version is removed after 90 days. | `infra/bootstrap/main.tf` (its own state is a local file, ignored by git) |
| The server (EC2, `c7i-flex.large`, Ubuntu 24.04) | Runs the stack with Docker Compose. | `infra/machine.tf` |
| Root disk, 30 GB | The system and Docker's images and containers. Lost when the machine is replaced. | `infra/machine.tf` |
| Data volume, 20 GB, mounted at `/srv/causalpulse` | Docker's named volumes (the database, the proxy's certificates), the environment file, the release files, the deploy records and logs. Kept when the machine is replaced; Terraform refuses to destroy it. | `infra/machine.tf` |
| Fixed address (Elastic IP) | What the site's A record names. Kept across a stop, a start and a replacement. | `infra/network.tf` |
| Security group | Ports 80 and 443 in over IPv4, everything out, nothing else. | `infra/network.tf` |
| Server role and instance profile | Lets the machine be managed by Systems Manager and pull the two images. Denied the two Parameter Store reads. | `infra/machine.tf` |
| Recovery alarm (CloudWatch) | Moves the machine to healthy hardware when its host fails the system status check. | `infra/machine.tf` |
| Two image repositories (ECR), `causalpulse-app` and `causalpulse-web` | The images, tagged by commit SHA. A tag is written once: the registry refuses a second push of a tag it has. Five images are kept in each. | `infra/images.tf` |
| Snapshot policy (Data Lifecycle Manager) and its role | A snapshot of the data volume each night, started within the hour after 03:00 UTC; seven are kept. | `infra/backups.tf` |
| Pipeline role, and GitHub's identity provider | What the deploy workflow assumes, from `main` only: push the two images, send the one deploy command to the one machine, read a command's result. Because that command runs a script from the image as root, the role amounts to root on the server ("Who is root on the server", below). | `infra/pipeline.tf` |
| Deploy document (Systems Manager), `causalpulse-deploy` | The one command the pipeline can run on the server. Its one parameter is a 40-character SHA. It runs as root. | `infra/deploy-command.tf` |
| Budget, `causalpulse-monthly` | E-mails at 80% of 100 US dollars a month, actual and forecast. It counts the whole AWS account, not this project alone. | `infra/budget.tf` |
| First-boot script | Run once by cloud-init: the data volume, Docker, swap, the AWS CLI, the environment file, automatic upgrades. Its last act writes `/var/lib/causalpulse/first-boot-done` on the root disk, without which a deploy is refused. | `deploy/first-boot.sh.tftpl` |
| Deploy script, compose override | What a deploy runs on the server, taken out of the app image. | `deploy/deploy.sh`, `deploy/compose.server.yml` |
| Deploy workflow | Started by hand; builds, pushes, sends the command, asks the site from outside. | `.github/workflows/deploy.yml` |
| Infrastructure workflow | Runs `make check-infra` on a pull request or a push to `main` that touches the Terraform, the deploy files, a workflow or the few other files the gate reads. It reaches no AWS account. | `.github/workflows/infra.yml` |
| The site's A record | Points the site's name at the fixed address. Added by hand at the DNS host; Terraform does not manage DNS. | nowhere in the repository |
| Five repository variables on GitHub | The identifiers the workflow reads. Set by hand from Terraform's outputs. | nowhere in the repository (procedure 1) |

On the server, under `/srv/causalpulse` (the data volume):

| Path | What it is |
|---|---|
| `.env` | The environment file: the site's name, the database password, the session secret. Root only, mode 600. Written once at first boot and never replaced by a deploy. |
| `release/` | The files a deploy took out of its image: `docker-compose.prod.yml` and `deploy/` (the override, the Caddyfile, the deploy script). They are those of the commit that is running, except in the three cases listed under "After a failed deploy" in procedure 2. |
| `release.old/`, `release.new/` | There only while a deploy runs: the directory it set aside, and the one it is filling. One left behind by a run that was cut off is dealt with by the next run. |
| `current-sha` | The commit recorded as running: written only after a deploy found every service well. |
| `previous-sha` | The commit recorded before that one. |
| `deploy.env` | The three settings the compose override needs (`CP_IMAGE_REGISTRY`, `CP_IMAGE_TAG`, `CP_CADDYFILE_DIGEST`), as they were for the last deploy that ended well. It holds no secret. It is for running Compose by hand (procedure 6); the deploy script writes it and never reads it. |
| `deploy.log` | Every deploy's output, newest last, each under a line `==== <time> <sha>, exit status <n>`. Kept to about a megabyte. Root only. |
| `deploy-run.log` | The whole output of the last deploy alone. Root only. |
| `deploy.lock` | The lock that keeps two deploys apart. |
| `docker-volumes/` | Docker's named volumes, bound onto `/var/lib/docker/volumes`. |

On the server's root disk, and so gone when the machine is replaced:

| Path | What it is |
|---|---|
| `/var/lib/causalpulse/first-boot-done` | Written as the last act of the first-boot script. The deploy document refuses, with status 76, a machine that does not have it (procedure 11). |

The compose project is `causalpulse-prod`. Its services are `postgres`, `redis`, `migrate` (runs
once and exits), `api`, `worker`, `web` and `proxy`.

### Who is root on the server

The deploy document takes `deploy/deploy.sh` out of the app image for the commit it is given and
runs it as root. So whoever can push an app image under a new tag and send the command has root
on the server: its environment file and the secrets in it, its database, and the server's own
role. The pipeline's role can do both, and AWS gives that role to any job that runs on `main`
and asks GitHub for an identity token, whichever workflow file the job is in. The repository
has no enforced branch protection, so anyone who can push to `main` is, in effect, root on the
server. That is the intended boundary while the repository is private and has one owner. A
test (`tests/test_deploy_files.py`) holds that only the deploy workflow asks for the token and
that it is started by hand only; that guards against doing it by accident, not against a person
who can push. The same holds on AWS's side: any principal in the account with
`ssm:SendCommand` or `ssm:StartSession` on the instance, or with rights over IAM, or with EC2 rights over the machine or its volumes, is root on
the server or can read its secrets too, and the sessions this runbook opens are one of those. Before a second person is
given the right to push, or access to the account, read §7 of the design for what has to
change.

## When something is wrong

| Symptom | Start with |
|---|---|
| The site does not answer | Procedure 4, "From the Mac": is the machine running, and does Systems Manager see it? If it is stopped, procedure 8. If it runs, open a session and look at the services' state. If the name does not resolve to the fixed address, the `dig` checks of procedure 1. |
| A certificate error in the browser | The `dig` checks of procedure 1 (an A record to the fixed address, and no AAAA record), then the proxy's log (procedure 4). The proxy can get a certificate only when the name already points at the server and ports 80 and 443 are reachable. |
| A deploy failed with status N | The table of statuses in procedure 2, then `deploy-run.log` on the server (procedure 4). After status 30, read "After a failed deploy" in procedure 2 before anything restarts the machine. |
| A deploy failed with status 76 | Procedure 11: the machine's first boot did not finish, and the deploy touched nothing. |
| The worker restarts in a loop | In a session: `sudo docker compose -p causalpulse-prod logs --tail 80 worker`. A worker that says a setting is a published demo value has a wrong environment file. One that is killed at its 2 GB limit shows `true` for `sudo docker inspect -f '{{.State.OOMKilled}}' "$(sudo docker compose -p causalpulse-prod ps -a -q worker)"`. |
| The disk is full | A failed deploy's output ends with how full the two disks are. In a session: `df -h / /srv/causalpulse` and `sudo docker system df`. On the root disk the usual cause is images: a failed deploy leaves its images until the next deploy that succeeds. `sudo docker image ls` lists them. |
| The budget e-mail arrived | The budget counts the whole account, which another project shares, so first see whose spending it is: the Billing console's Cost Explorer, grouped by service. If it is this project, procedure 8 (stop the machine) or procedure 10 (remove everything). |
| The workflow says the command's result could not be read, or had not ended, or its summary says the command was sent and its result was not read | The deploy may still be running: stopping a run does not stop it. Look at `deploy.log` on the server (procedure 4) before starting another. |

## 1. First-time setup

Use this once, to create everything from nothing. It needs an AWS profile that can create the
resources in the table above, the repository checked out at a commit of `main` that holds
`infra/` and `.github/workflows/deploy.yml`, and access to the DNS host of the site's domain.
It creates resources that are billed from the moment they exist (about 42 US dollars a month,
see the README).

> **Not yet carried out.**

**The tools.** Terraform 1.15 or a later 1.x (`infra/versions.tf` asks for `~> 1.15`), the AWS
CLI version 2, the Session Manager plugin, the GitHub CLI and `dig`.

```bash
terraform version
aws --version
gh auth status
```

The Session Manager plugin is what `aws ssm start-session` needs on the Mac. With Homebrew,
whose cask of that name is "Session Manager Plugin for the AWS CLI":

```bash
brew install --cask session-manager-plugin
session-manager-plugin --version
```

Check who the AWS CLI is, and set the region, which every later command names:

```bash
REGION=eu-west-1
aws sts get-caller-identity --query Arn --output text
```

**The state bucket.** `infra/bootstrap/` makes the bucket that holds the state of `infra/`. The
bucket's name is its one variable, with no default: choose a name unused in all of AWS. The
plan should show six resources to add and nothing else.

```bash
STATE_BUCKET=<a bucket name unused in all of AWS>
terraform -chdir=infra/bootstrap init
terraform -chdir=infra/bootstrap plan -var "state_bucket=$STATE_BUCKET" -out bootstrap.tfplan
```

```bash
terraform -chdir=infra/bootstrap apply bootstrap.tfplan
```

The bootstrap's own state is the file `infra/bootstrap/terraform.tfstate`, which git ignores and
nothing backs up. Keep it: it is the only record that Terraform made the bucket.

**The backend.** The bucket is passed when the state is first set up; the key, the region and
the lock file are written in `infra/versions.tf`.

```bash
STATE_BUCKET=$(terraform -chdir=infra/bootstrap output -raw state_bucket)
terraform -chdir=infra init -backend-config="bucket=$STATE_BUCKET"
```

**The values.** Copy the example and fill it in. `terraform.tfvars` is ignored by git.

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
```

| Variable | What to put | How to find it (each command only reads) |
|---|---|---|
| `site_name` | The site's public name, lower case, with no scheme and no path. | It is yours to choose. |
| `repository` | The GitHub repository whose `main` may deploy, as `owner/name`. | `gh repo view --json nameWithOwner --jq .nameWithOwner` |
| `budget_email` | Where the budget's two warnings go. | It is yours to choose. |
| `ami_id` | The current Ubuntu 24.04 LTS image for x86. It is written down and not looked up at each apply, so a newer image never replaces the machine by itself. | The first command below. |
| `subnet_id` | One of the default VPC's public subnets. Choose one and keep it: the data volume is created in its zone and cannot leave it. | The second command below, which lists each with its zone. |
| `create_github_oidc_provider` | `true` if the account has no identity provider for GitHub Actions, `false` if it has one. An account holds one. | The third command below: `true` if nothing it prints ends `oidc-provider/token.actions.githubusercontent.com`. |

```bash
aws ssm get-parameter --region "$REGION" --query Parameter.Value --output text \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
```

```bash
aws ec2 describe-subnets --region "$REGION" \
  --filters Name=default-for-az,Values=true \
  --query 'Subnets[?MapPublicIpOnLaunch].[SubnetId,AvailabilityZone]' --output text
```

```bash
aws iam list-open-id-connect-providers --query 'OpenIDConnectProviderList[].Arn' --output text
```

With `true`, a later `terraform destroy` here removes the provider from the account, from under
anything else that has come to use it (procedure 10).

**The plan.** Save it, and read it before applying.

```bash
terraform -chdir=infra plan -out tfplan
```

What to look for:

- Only additions: nothing to change and nothing to destroy. Counted from the configuration,
  that is 30 resources, or 31 when the identity provider is created here. If the plan's number
  differs, find out why before applying.
- `aws_instance.server` is a `c7i-flex.large` with no `key_name`, and `aws_security_group.server`
  has two inbound rules, ports 80 and 443.
- `aws_iam_openid_connect_provider.github[0]` is there only if you answered `true`.
- The plan stops with a sentence of its own if the subnet does not give a public IPv4 address
  at launch or assigns IPv6 addresses.

The saved plan holds the account's number and the budget's address. Git ignores it under
`infra/`; do not copy it elsewhere.

**The apply.** Apply the plan that was read, not a new one.

```bash
terraform -chdir=infra apply tfplan
rm infra/tfplan
```

How to tell it worked: a second plan proposes nothing.

```bash
terraform -chdir=infra plan
```

The machine then runs its first-boot script, which takes some minutes: how many has not been
measured, and the script itself may wait several minutes for the package lists. **Check that it
finished before the first deploy.** A deploy sent to a machine whose first boot did not finish
is refused with status 76 and touches nothing. It is done when Systems Manager reports the
machine `Online` and, in a session (procedure 3 shows how to open one), cloud-init reports
`done` and the marker file is there:

```bash
INSTANCE_ID=$(terraform -chdir=infra output -raw instance_id)
aws ssm describe-instance-information --region "$REGION" \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].PingStatus' --output text
```

In a session:

```bash
cloud-init status --long
sudo tail -n 30 /var/log/cloud-init-output.log
test -e /var/lib/causalpulse/first-boot-done && echo "the first boot finished"
sudo test -s /srv/causalpulse/.env && echo "the environment file is there"
```

The last line of the script's output is `first-boot: done`. It prints the names of the settings
it wrote and never their values. If cloud-init reports an error, the marker is missing, or the
machine never comes `Online`, go to procedure 11 and not to the first deploy.

**The budget's e-mail.** The budget sends its warnings straight to the address in
`terraform.tfvars`. Check the budget exists, and if AWS sends that address a message asking it
to confirm, confirm it. A review of the configuration noted that a budget's e-mail subscriber
needs no confirmation, unlike a notification topic's; that has not been checked against AWS
here. A warning can also land in the spam folder: look there once.

```bash
aws budgets describe-budget --budget-name causalpulse-monthly \
  --account-id "$(aws sts get-caller-identity --query Account --output text)" \
  --query 'Budget.[BudgetName,BudgetLimit.Amount,BudgetLimit.Unit]' --output text
```

**The DNS record.** At the DNS host of the site's domain, add one A record for the site's name
pointing at the address this prints. Add no AAAA record: the proxy is published on IPv4 only.

```bash
terraform -chdir=infra output -raw site_address
```

Check it from the Mac. The first line must print `the A record is right`, and the second must
print nothing at all.

```bash
SITE_NAME=$(terraform -chdir=infra output -raw site_name)
[ "$(dig +short A "$SITE_NAME")" = "$(terraform -chdir=infra output -raw site_address)" ] && echo "the A record is right"
dig +short AAAA "$SITE_NAME"
```

A resolver may hold an old answer for a while. To ask the domain's own name servers, list them
and name one:

```bash
dig +short NS <the domain the site's name belongs to>
dig +short A "$SITE_NAME" @<one of the name servers just listed>
```

Do this before the first deploy. The proxy asks for the site's certificate as soon as it
starts, and it can only get one when the name already points at the server.

**The repository's variables.** The workflow reads five, by these exact names. They are
identifiers and not secrets, and each comes from one of Terraform's outputs.

```bash
gh variable set AWS_ROLE_ARN --body "$(terraform -chdir=infra output -raw pipeline_role_arn)"
gh variable set INSTANCE_ID --body "$(terraform -chdir=infra output -raw instance_id)"
gh variable set DEPLOY_DOCUMENT --body "$(terraform -chdir=infra output -raw deploy_document_name)"
gh variable set REGISTRY --body "$(terraform -chdir=infra output -raw registry)"
gh variable set SITE_NAME --body "$(terraform -chdir=infra output -raw site_name)"
```

```bash
gh variable list
```

They are the repository's variables and not an environment's, on purpose: the deploy job
declares no GitHub environment, because the pipeline role's trust accepts only the token
subject of a run from `main`, and a job with an environment presents a different one.

**The first deploy** is procedure 2, and **the first key** is procedure 3.

## 2. Deploying and rolling back

Use this to put a commit of `main` on the server, or to go back to an earlier one. It needs the
five repository variables set (procedure 1), a machine whose first boot finished (procedure 1,
and procedure 11 if it did not), and the commit's gates to have passed.

> **Not yet carried out.**

**Which commit.** The workflow deploys a commit only if it is on `main` and the `CI` workflow
(`ci.yml`) succeeded for it as the tip of a push to `main`. A commit that reached `main` inside
a merged pull request, and was not the last commit of that push, never has such a run: deploy
the merge commit, which has. A run of CI on a pull request, or on another branch that held the
same commit, does not count.

Two more workflows are looked at in the same exact way. The fast Python suite (`fast.yml`)
runs on every push to `main`, as four shards that each take a while; the infrastructure gate
(`infra.yml`) runs only for a push that touches what it checks. Where either has a run for the
commit, one must have succeeded, and a run still in progress has not: after a merge, a deploy
has to wait for the fast suite. Where the infrastructure gate has no run, the workflow says so
and goes on, which is usual. Where the fast suite has none, it warns and goes on, and that is
not expected: find out why. A run that was waiting on `main` when a later push took its place
is cancelled by GitHub and counts as not succeeded; run it again from its page.

To see where the commit stands before starting:

```bash
git fetch origin
SHA=$(git rev-parse origin/main)
gh run list --commit "$SHA" --json workflowName,event,headBranch,status,conclusion
```

**Starting it.** Two inputs: the commit's full SHA, and the site's name typed out as the
confirmation. Start it from `main`: a run started from another branch or a tag is refused the
role by AWS.

```bash
SITE_NAME=$(terraform -chdir=infra output -raw site_name)
gh workflow run deploy.yml --ref main -f sha="$SHA" -f confirm="$SITE_NAME"
```

From the web page: the repository's Actions tab, "Deploy" in the list of workflows, "Run
workflow", the branch `main`, and the same two fields.

To follow it:

```bash
RUN_ID=$(gh run list --workflow deploy.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

**What it does, in order.**

1. Refuses unless it was started from `main`, the SHA is forty lowercase hexadecimal digits, the
   five variables are set, and the confirmation equals `SITE_NAME`.
2. Checks the commit out and refuses unless it is an ancestor of `origin/main`.
3. Refuses unless CI succeeded for it, and the fast suite and the infrastructure gate where
   they ran for it, as above.
4. Assumes the pipeline's role, for two hours, and asks the registry which of the two images it
   already has for the SHA. It builds and pushes only what is missing: a second deploy of a
   commit, or a rollback to one whose images are still in the registry, builds nothing. A run
   that pushed one image and failed on the other is started again and pushes the other.
5. Sends the deploy document to the server with the SHA and asks every ten seconds how it went.
6. Asks the site three questions from outside, for up to two minutes: `https://<site>/study`
   answers 200 with a `Strict-Transport-Security` header; `https://<site>/` answers 307 to
   `/sign-in`; `http://<site>/` answers 301 or 308 to `https://<site>/`.

A summary of the run, with the commit and the outcome of each part, is on the run's page. The
list of runs does not show which commit each deployed (the workflow sets no run name): the
summary does, and so do the two lists under "Rolling back".

**How long.** Not measured: the builds have never run on a runner. The limits are known. The
job is stopped after 75 minutes, which is inside the two hours the role's credentials last. The
server is given a minute to take the command and the
document 30 minutes to run it; the deploy script waits up to 300 seconds for the services to be
healthy and then looks again 25 seconds later. One deploy runs at a time: a second waits, and a
third started while one runs and one waits replaces the waiting one.

**Stopping a run does not stop a deploy the server has been sent.** The role cannot cancel a
command, and the server finishes it. `deploy.log` on the server has how it went. The summary of
a run that was stopped, or timed out, while it waited says the command was sent and its result
was not read.

**Reading a failure.** `gh run view "$RUN_ID" --log-failed` shows the step that failed. A
failure before the step "Deploy on the server" means the server was not touched. That step
prints the end of the deploy's own output, with the values of the server's settings already
taken out of it, and the exit status. The statuses of `deploy/deploy.sh`:

| Status | Meaning |
|---|---|
| 0 | every service is well and `current-sha` holds the SHA |
| 1 | a command failed where the script did not expect one to (`set -e`) |
| 2 | the arguments or the deadline are not what the script's header describes |
| 10 | the release directory cannot be entered; a file of the release or the environment file is missing; `current-sha`, `previous-sha` or `deploy.env` is there and is not a file; the Caddyfile's digest could not be computed; or an image could not be pulled or is not on the machine afterwards |
| 20 | the migration failed, or Postgres was not there for it to run against |
| 30 | the stack did not come up, or was not well by the deadline |
| 40 | the stack is well on the SHA and `previous-sha`, `deploy.env` or `current-sha` could not be written |

The script does not roll back. After 2, 10 or 20 no application service has been replaced.
After 30 or 40 the stack is as the attempt left it, and `current-sha` still holds what it held
before.

The document around the script has statuses of its own:

| Status | Meaning |
|---|---|
| 75 | another deploy was running on the server; this one did nothing |
| 76 | the machine's first boot did not finish: `/var/lib/causalpulse/first-boot-done` is missing. Nothing was touched, the lock included. Procedure 11. |
| the failed command's own | the registry login, the pull of the app image, or the copy of the files out of it failed before the script was reached. The status is that command's and is not one of the script's own: it can be 1, which the script also uses, so read the output, which says what failed. |
| -1, as the workflow reports it | Systems Manager has no exit status: the command was not run on the server, or was stopped there. |

After any status of the script but 0 the output ends with how full the two disks are (`df -h` for `/` and
`/srv/causalpulse`). If the document cannot search its own output for the environment file's
values, or the environment file is missing or cannot be read, it keeps and shows none of the
run, says which of the two it was, and still returns the status.

**After a failed deploy.** The proxy's Caddyfile is mounted from `/srv/causalpulse/release` by
its path, so the files there should be those of the commit that is running. The document
copies the new commit's files to `release.new`, sets the directory in place aside as
`release.old`, and when the deploy script has returned decides which stays. After 0, 30 and 40
it is the new one. After anything else (1, 2, 10, 20) it puts back the one that was there:
by the script's contract nothing was replaced after 2, 10 or 20, and 1, which the script does
not expect to give, is treated as not deployed. A failed registry login, pull or copy
ends the run before any directory is renamed. So after a failure with status 10 or 20 the
release directory matches what is running, and nothing has to be put back by hand.

Three cases are left where the directory does not match what is running, and a fourth is
possible if a status the script did not choose (1 after the stack was replaced, or 137 from a
kill) puts the old directory back over a stack that is new; the remedy is the same, to deploy
the commit that should be running:

- **After status 30.** The directory holds the new commit's files, and the stack is as the
  attempt left it: possibly some services on the new commit and some on the old.
  `current-sha` and `deploy.env` still name the commit before. If the proxy's container is
  restarted in this state, by a reboot for one, it starts on the new commit's Caddyfile. Before
  anything reboots the machine, either fix the fault and deploy again, or go back by running
  the workflow with the commit named in `current-sha`:

  ```bash
  gh workflow run deploy.yml --ref main -f sha="<the SHA in current-sha>" -f confirm="$SITE_NAME"
  ```

- **After a run that was cut off** while the deploy script ran (a reboot, or the document's
  30-minute limit). It leaves both `release` and `release.old`, and the next run puts
  `release.old` back before it does anything else, on the reasoning that a run is cut off in the
  pulls or the migration more easily than after them. If that run had in fact replaced the
  stack, the directory then holds the commit before the one running. In either case the answer
  is the same: deploy the commit that should be running, and the files of the directory and
  the stack agree again when that deploy ends with 0. If the run was cut off by a reboot, the
  proxy was restarted while `release` held the new commit's files, and a deploy of the commit
  that was running does not recreate it, because Compose sees the label the proxy was created
  with. So when that deploy has ended with 0, restart the proxy so that it reads the Caddyfile
  now in place:

  ```bash
  sudo docker restart "$(sudo docker compose -p causalpulse-prod ps -q proxy)"
  ```
  The same happens when a deploy ended with 0, 30 or 40 but its rename of `release.old` out of
  the way failed (the log says "could not be renamed"): `release.old` is still there, and the
  next run puts it back, which sets `release` to the commit before the one running. If that
  is not wanted, remove it by hand from a session (`sudo rm -rf /srv/causalpulse/release.old`) before the next run.
- **After a first deploy that failed.** There was no directory to put back, so the new files
  stay whatever the status.

After status 40 the stack is well on the new commit and only a record is missing. Find why the
file could not be written (a full data volume is the likely cause), then run the workflow again
with the new SHA: `previous-sha` and `current-sha` are both written again by that run.

That Docker looks the Caddyfile's path up again at each start of the proxy's container on a
Linux server, which is what all of this rests on, is unverified until carried out: the gate
that tries it runs on Docker Desktop, which does not mount files as Linux does.

**A bad image.** A tag is written once. The registry refuses a second push of a tag it has, and
the pipeline's role may neither overwrite a tag nor delete an image. So an image that is bad
under its SHA cannot be replaced by running the workflow again. The usual answer is a new
commit, which has a new SHA. To build the same commit again, someone with an administrator's
rights in the account deletes the tag, and the next deploy of that SHA finds it missing and
builds it. The build reads the layer cache (`cache-from: type=gha`), so a bad layer that was
cached comes back the same: when the image itself was the fault, delete the cache first with
`gh cache delete --all`.

```bash
REGION=eu-west-1
aws ecr batch-delete-image --region "$REGION" --repository-name causalpulse-app --image-ids "imageTag=$SHA"
```

The dashboard's repository is `causalpulse-web`. Deleting an image the server is running does
not stop the server, which has its own copy; it does mean that a machine replaced before the
image is built again has nothing to pull.

**Rolling back.** Run the same workflow with the earlier SHA. The deploy requires the
continuous integration run of that commit, and its fast suite and infrastructure gate where
they ran, to have succeeded; a run still in progress or queued is refused. A run of
either that was cancelled can be run again for 30 days only, so a commit older than that whose
run was cancelled cannot be a rollback target. To find it, on the server
`previous-sha` holds the commit recorded before the current one (in a session:
`sudo cat /srv/causalpulse/previous-sha`), and from the Mac the deploy commands that were sent
are listed with the SHA in each one's comment. Systems Manager keeps that list for a limited
time.

```bash
REGION=eu-west-1
aws ssm list-commands --region "$REGION" \
  --filters "key=DocumentName,value=$(terraform -chdir=infra output -raw deploy_document_name)" \
  --query 'Commands[].[RequestedDateTime,Status,Comment]' --output text
```

The server keeps the images of the current commit and the one before it, and each repository
keeps five images. A rollback further back than the registry holds is built again from the
commit.

**If a migration ran in between.** A deploy runs `cp-api-migrate`, which upgrades the database
to the newest revision the image holds, and nothing in a deploy ever downgrades it. To tell
whether the two commits differ in their migrations (nothing printed means they do not):

```bash
git diff --stat <the earlier SHA> <the later SHA> -- packages/cp-api/src/cp_api/migrations/versions
```

The database's own revision, in a session:

```bash
sudo docker compose -p causalpulse-prod exec -T postgres \
  psql -U causalpulse -d causalpulse -c 'select version_num from alembic_version'
```

If they differ, the earlier image's migration step meets a database stamped with a revision it
does not have. What is expected, from how Alembic works and not from a trial, is that the step
fails, the deploy ends with status 20, and no service is replaced: the rollback does not
happen, and the document puts the release directory back as it was ("After a failed deploy",
above). Unverified until carried out. The choices are then a person's:

- Fix forward: a new commit on `main` that repairs the fault, deployed in the usual way. This
  is the one that loses nothing.
- Restore the data volume from a snapshot taken before the migration (procedure 5), and then
  deploy the earlier commit. Everything written since the snapshot is lost.
- Downgrade the schema by hand. The migrations have `downgrade` functions and
  `cp_api.migrate.downgrade` calls them, but no command in the image runs it and it has never
  been run against this database. Take a snapshot first if you try it.

## 3. Creating a user and issuing a key

Use this to give someone access, or to take it away. It needs the Session Manager plugin
(procedure 1) and a stack that is running.

> **Not yet carried out.**

**Never use a run-command (`aws ssm send-command`) to issue a key. A run-command's output is
kept by AWS and can be read back later by anyone allowed to read command results in the
account, the pipeline's role among them. A key is shown once and only its hash is stored, so
it must appear in your terminal and nowhere else: issue it in an interactive session.**

A session can itself be recorded, if the account's Session Manager preferences send sessions to
S3 or CloudWatch. This configuration sets no such preference, but the account is shared, so
look once. An error saying the document does not exist means the defaults, which record
nothing; if it prints a bucket's or a log group's name that is not empty, do not issue a key
until that is understood. The document's name is from AWS's guide to Session Manager's
preferences; that its absence means nothing is recorded is unverified until carried out.

```bash
REGION=eu-west-1
aws ssm get-document --region "$REGION" --name SSM-SessionManagerRunShell --query Content --output text
```

Open the session:

```bash
INSTANCE_ID=$(terraform -chdir=infra output -raw instance_id)
aws ssm start-session --region "$REGION" --target "$INSTANCE_ID"
```

In the session, create the user and issue a key. The first command prints `created user` and
the name; the second prints the key, once. A session is the agent's own user, `ssm-user`, which
AWS's guide says the agent creates with `sudo` rights. Compose is given the project's name and
no file: its reference shows `ps` and `logs` used that way, and that `exec` works the same way
is unverified until carried out.

```bash
cd /
sudo docker compose -p causalpulse-prod exec api cp-api-admin create-user <name>
sudo docker compose -p causalpulse-prod exec api cp-api-admin issue-key <name>
```

Copy the key to where it is going, then clear the terminal and leave:

```bash
clear
exit
```

How to tell it worked: the key signs in at `https://<site>/sign-in`.

**Revoking a key.** The key id is the 12 characters after `cpk_` in the key. In a session:

```bash
sudo docker compose -p causalpulse-prod exec api cp-api-admin revoke-key <key id>
```

It prints `revoked key` and the id, and the key is refused from then on. `cp-api-admin` has
these three commands and no other: it cannot list users or keys, so a key whose text is lost
cannot be looked up by this command.

## 4. Looking at the server

Use this to see what the server is doing. Nothing here changes anything.

> **Not yet carried out.**

**From the Mac**, without a session: whether the machine is running, whether its two status
checks pass, and whether Systems Manager can reach it.

```bash
REGION=eu-west-1
INSTANCE_ID=$(terraform -chdir=infra output -raw instance_id)
aws ec2 describe-instance-status --region "$REGION" --instance-ids "$INSTANCE_ID" --include-all-instances \
  --query 'InstanceStatuses[0].[InstanceState.Name,SystemStatus.Status,InstanceStatus.Status]' --output text
aws ssm describe-instance-information --region "$REGION" \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].[PingStatus,LastPingDateTime]' --output text
```

And the site, as a visitor sees it:

```bash
SITE_NAME=$(terraform -chdir=infra output -raw site_name)
curl -sS -o /dev/null -w '%{http_code}\n' "https://$SITE_NAME/study"
```

**In a session** (`aws ssm start-session --region "$REGION" --target "$INSTANCE_ID"`). Type
`cd /` first: Compose is given the project's name and no file, and must not find one where it
stands.

The services' state. Every long-running service should read `running` and `healthy`, and
`migrate` should have exited with 0.

```bash
sudo docker compose -p causalpulse-prod ps -a
```

A service's log. Each is kept to three files of 10 MB.

```bash
sudo docker compose -p causalpulse-prod logs --tail 100 api
sudo docker compose -p causalpulse-prod logs --tail 100 proxy
```

The commit recorded as running, and the one before it; then the settings of the last deploy
that ended well, whose `CP_IMAGE_TAG` is the same commit:

```bash
sudo cat /srv/causalpulse/current-sha /srv/causalpulse/previous-sha
sudo cat /srv/causalpulse/deploy.env
```

Whether the last deploy run ended well, and then whether the release directory's Caddyfile is the
one that was deployed. Each prints its sentence when it holds; see "After a failed deploy" in
procedure 2 when one does not. The first is the one that tells a status 30 from a deploy that
ended well: after a 30 the Caddyfile's digest, like the two SHAs above, still names the commit
before, and the directory holds the new commit's files.

```bash
[ "$(sudo grep '^==== ' /srv/causalpulse/deploy.log | tail -n 1 | sed 's/^==== [^ ]* //')" = "$(sudo cat /srv/causalpulse/current-sha), exit status 0" ] && echo "the last deploy ended with status 0 for the running commit"
[ "$(sudo sha256sum /srv/causalpulse/release/deploy/Caddyfile | cut -d' ' -f1)" = "$(sudo sed -n 's/^CP_CADDYFILE_DIGEST=//p' /srv/causalpulse/deploy.env)" ] && echo "the release's Caddyfile is the one deployed"
ls -d /srv/causalpulse/release*
```

The last line should list `release` alone, unless a deploy is running.

The deploys: one line for each, then the whole of the last one.

```bash
sudo grep '^====' /srv/causalpulse/deploy.log
sudo cat /srv/causalpulse/deploy-run.log
```

Disk and memory. The machine has 4 GB and a 2 GB swap file; the worker is limited to 2 GB.

```bash
df -h / /srv/causalpulse
free -m
sudo docker system df
sudo docker stats --no-stream
```

After a first boot or a replacement, what the first-boot script did, and whether it finished
(procedure 11 if it did not):

```bash
cloud-init status --long
sudo tail -n 60 /var/log/cloud-init-output.log
ls -l /var/lib/causalpulse/first-boot-done
```

## 5. Restoring the data volume from a snapshot

Use this when the database is lost or damaged and last night's copy is wanted back. It needs
the Mac, Terraform's state, and about a quarter of an hour of the site being down. Everything
written to the server since the snapshot is lost: the database, and with it users, keys,
datasets and analyses.

> **Not yet carried out.**

This procedure is awkward, and it is the least bad one the configuration allows. The data
volume is a Terraform resource that Terraform refuses to destroy, and a volume cannot be
rewound to a snapshot in place: a new volume is made from the snapshot. So the new volume is
swapped in by hand and Terraform's state is then told about it, by removing two resources from
the state and importing two. Between those two steps the state does not describe what exists:
**run no `terraform apply` until the last step here has shown a clean plan.**

A second thing to know before starting. `current-sha`, `previous-sha` and the release files are
on the data volume, so they go back to the snapshot's time too, while the containers, which are
on the root disk, stay on the commit that was running. The last step deploys that commit again
to bring the two back together, and to run the migration against the restored database.

Set the values, and note the commit that is running now:

```bash
REGION=eu-west-1
INSTANCE_ID=$(terraform -chdir=infra output -raw instance_id)
SITE_NAME=$(terraform -chdir=infra output -raw site_name)
SHA=<the commit running now: current-sha on the server, or the last successful deploy in procedure 2's list>
OLD_VOLUME=$(terraform -chdir=infra output -raw data_volume_id)
ZONE=$(aws ec2 describe-volumes --region "$REGION" --volume-ids "$OLD_VOLUME" \
  --query 'Volumes[0].AvailabilityZone' --output text)
echo "$OLD_VOLUME $ZONE"
```

Find the snapshot. They are listed oldest first; choose one that is `completed`.

```bash
aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
  --filters "Name=volume-id,Values=$OLD_VOLUME" \
  --query 'sort_by(Snapshots,&StartTime)[].[SnapshotId,StartTime,State]' --output text
```

```bash
SNAPSHOT=<the snapshot's id>
```

Stop the machine, which stops the stack with it. Do not stop the services with Compose first: a
service stopped by hand does not come back by itself when the machine starts.

```bash
aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$INSTANCE_ID"
```

Make the new volume from the snapshot, in the machine's zone, with the tags the old one
carries. The `Backup` tag is what the snapshot policy looks for, and it finds volumes by that
tag and by nothing else, so a volume without it is not snapshotted. The tags are also
Terraform's: once the new volume has been imported below, an apply would put them on a volume
that lacked them (unverified until carried out). They are given here all the same, so that the
volume is never without the `Backup` tag. A volume made from an encrypted snapshot is expected
to be encrypted (unverified until carried out); the last command must print `True`.

```bash
NEW_VOLUME=$(aws ec2 create-volume --region "$REGION" --snapshot-id "$SNAPSHOT" \
  --availability-zone "$ZONE" --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=causalpulse-data},{Key=Backup,Value=causalpulse-data},{Key=Project,Value=causalpulse}]' \
  --query VolumeId --output text)
aws ec2 wait volume-available --region "$REGION" --volume-ids "$NEW_VOLUME"
aws ec2 describe-volumes --region "$REGION" --volume-ids "$NEW_VOLUME" --query 'Volumes[0].Encrypted' --output text
```

Swap them. The old volume is kept, detached, until the restore is known to be good; its
`Backup` tag is taken off so that the policy stops snapshotting it, and it is renamed so the
two can be told apart.

```bash
aws ec2 detach-volume --region "$REGION" --volume-id "$OLD_VOLUME"
aws ec2 wait volume-available --region "$REGION" --volume-ids "$OLD_VOLUME"
aws ec2 attach-volume --region "$REGION" --volume-id "$NEW_VOLUME" --instance-id "$INSTANCE_ID" --device /dev/sdf
aws ec2 wait volume-in-use --region "$REGION" --volume-ids "$NEW_VOLUME"
aws ec2 delete-tags --region "$REGION" --resources "$OLD_VOLUME" --tags Key=Backup
aws ec2 create-tags --region "$REGION" --resources "$OLD_VOLUME" --tags Key=Name,Value=causalpulse-data-replaced
```

Start the machine. It mounts the volume by its file system's identifier, which the snapshot
carries, and the services come back by their restart policy (unverified until carried out). The
first-boot marker is on the root disk, which was not touched, so a deploy is not refused.

```bash
aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
```

Tell Terraform. Remove the old volume and its attachment from the state (this deletes nothing
in AWS), import the new ones, and plan. The attachment's import id has the form the AWS
provider's documentation gives, `DEVICE_NAME:VOLUME_ID:INSTANCE_ID`.

```bash
terraform -chdir=infra state rm aws_volume_attachment.data aws_ebs_volume.data
terraform -chdir=infra import aws_ebs_volume.data "$NEW_VOLUME"
terraform -chdir=infra import aws_volume_attachment.data "/dev/sdf:$NEW_VOLUME:$INSTANCE_ID"
terraform -chdir=infra plan
```

The plan should propose nothing, or only changes in place (to the volume's tags, if any was
left off above). That a volume made from a snapshot imports with no replacement planned is
unverified until carried out: the provider computes `snapshot_id` when the configuration does
not give it, which is why none is expected. If the plan proposes to replace or destroy
`aws_ebs_volume.data`, do not apply: Terraform would refuse in any case, and the difference it
names has to be understood first.

Deploy the commit that was running, which puts the release files and the two records back and
runs the migration against the restored database:

```bash
gh workflow run deploy.yml --ref main -f sha="$SHA" -f confirm="$SITE_NAME"
```

How to tell it worked: the deploy succeeds; the site shows the data as it was at the snapshot;
`terraform -chdir=infra plan` proposes nothing; `terraform -chdir=infra output -raw
data_volume_id` prints the new volume's id; and the next morning the listing above, given
`$NEW_VOLUME`, shows a new snapshot.

When you are satisfied, the old volume is still there and still billed, and Terraform no longer
knows it. Deleting it cannot be undone:

```bash
aws ec2 delete-volume --region "$REGION" --volume-id "$OLD_VOLUME"
```

## 6. Rotating the session secret, and the database password

Use this when either may have been seen by someone who should not have it. Both are in
`/srv/causalpulse/.env` on the server and nowhere else. It needs a session, and a server whose
last deploy succeeded.

> **Not yet carried out.**

**Set-up, in a session, for this procedure and the next.** A Compose command that creates a
container needs the stack's configuration, and the override in it requires three values. The
deploy script writes them to `/srv/causalpulse/deploy.env` each time a deploy ends well, so
Compose is given two environment files: the server's own, then that one. (`docker compose
--help` gives `--env-file` as a list.) This defines `stack` as the deploy script's own Compose
command with the second file added, and `set_env`, which changes one line of the environment
file without printing any of it.

```bash
sudo -i
cd /srv/causalpulse/release
stack() {
    docker compose -p causalpulse-prod --env-file /srv/causalpulse/.env \
        --env-file /srv/causalpulse/deploy.env \
        -f docker-compose.prod.yml -f deploy/compose.server.yml "$@"
}
set_env() {
    (umask 077 && VALUE="$2" awk -v name="$1" '
        index($0, name "=") == 1 { print name "=" ENVIRON["VALUE"]; found = 1; next }
        { print }
        END { if (!found) print name "=" ENVIRON["VALUE"] }
    ' /srv/causalpulse/.env >/srv/causalpulse/.env.new) &&
        mv /srv/causalpulse/.env.new /srv/causalpulse/.env
}
```

Before going on, check that the release directory's files are those of the commit that is
running. First the history, and that no run was cut off: the last deploy run (every run past
the lock writes one line) must have ended with status 0 for the commit in `current-sha`, and
`release.old` must not exist. After a status 30, or a 40 from `previous-sha` or the record of
`deploy.env`, both files still name the commit before, so the three lines after this one
agree whether the last deploy succeeded or not; the history line tells those apart. A run that
was cut off writes no line, so the history line alone cannot tell it: it leaves `release.old`
beside `release`, and the first test in the line below sees that. If the line does not print
its sentence, stop and see "After a failed deploy" in procedure 2; a cut-off run is a case to
stop on.

```bash
[ ! -e /srv/causalpulse/release.old ] && [ "$(grep '^==== ' /srv/causalpulse/deploy.log | tail -n 1 | sed 's/^==== [^ ]* //')" = "$(cat /srv/causalpulse/current-sha), exit status 0" ] && echo "the last deploy ended with status 0 for the running commit, and no run was cut off"
```

Then the files. The first two lines printed must name the same SHA, and the third must be the
sentence. If `deploy.env` is missing, no deploy has ended well on this server yet.

```bash
cat /srv/causalpulse/current-sha
sed -n 's/^CP_IMAGE_TAG=//p' /srv/causalpulse/deploy.env
[ "$(sha256sum <deploy/Caddyfile | cut -d' ' -f1)" = "$(sed -n 's/^CP_CADDYFILE_DIGEST=//p' /srv/causalpulse/deploy.env)" ] && echo "the release's Caddyfile is the one deployed"
```

Do not do any of this while a deploy is running.

**The session secret.** The dashboard seals its session cookie with it, so changing it signs
everyone out. Only `web` reads it.

```bash
set_env CP_SESSION_SECRET "$(openssl rand -hex 32)"
stack up -d --no-build --no-deps web
```

How to tell it worked: `stack ps web` reads `healthy` within a minute, the site's sign-in page
answers, and a browser that was signed in is asked to sign in again. That the proxy finds the
new `web` container without being restarted itself is unverified until carried out.

**The database password.** It is in two places: inside Postgres, as the role's password, and in
the environment file, from which the API, the worker and the migration build their connection.
Postgres took it from the file only when its volume was first created; changing the file alone
does not change it in the database. The password must be letters and digits only, because the
compose file places it in a URL as it is: `openssl rand -hex 32` gives one.

The order matters: the database first, then the file, then the services.

```bash
NEW=$(openssl rand -hex 32)
printf "ALTER ROLE causalpulse PASSWORD '%s';\n" "$NEW" |
    docker compose -p causalpulse-prod exec -T postgres psql -U causalpulse -d causalpulse -v ON_ERROR_STOP=1
```

That must print `ALTER ROLE`. Only then:

```bash
set_env POSTGRES_PASSWORD "$NEW"
unset NEW
stack up -d --no-build --no-deps api worker
```

Recreating the worker ends an analysis that is running; it is then marked failed and has to be
submitted again.

How to tell it worked: `stack ps` reads `healthy` for `api` and `worker` within a minute, and
the site lists a signed-in user's analyses.

What breaks in the wrong order:

- **The file and the services first, the database not yet.** The API and the worker start with
  a password Postgres does not accept. The API's health check asks Postgres, so the API never
  reads healthy and the site cannot sign anyone in until the `ALTER ROLE` is run.
- **The database, and then nothing more.** The services go on for a while on the connections
  they already hold, and each new connection is refused, so the failures arrive piecemeal; a
  restart of either service, or a reboot, then fails outright. A deploy in this state ends with
  status 20, because the migration cannot connect.
- **The new value lost between the two steps** (the session closed, say). The database has a
  password nobody holds. Run the `ALTER ROLE` again with a fresh one: `psql` inside the
  Postgres container does not need the password, because the image sets up `trust`
  authentication for connections from inside the container, as its documentation says.

Afterwards the Postgres container's own settings differ from the file's, since it was created
with the old value. The next deploy is expected to replace that container for this reason
(unverified until carried out); the data is on the volume and is kept, and the password in the
database is not changed by it.

## 7. Changing a rate limit

Use this to raise or lower one of the four limits. It needs a session and no deploy.

> **Not yet carried out.**

The names, from `.env.example`. Each is a whole number greater than zero, and a name that is
absent or empty takes the default.

| Name | Default | What it limits |
|---|---|---|
| `CP_LIMIT_SIGNIN_FAILURES` | 10 | requests answered 401 from one client address in 15 minutes |
| `CP_LIMIT_UPLOADS_PER_DAY` | 20 | uploads a user may send in a day |
| `CP_LIMIT_ANALYSES_PER_DAY` | 30 | analyses a user may start in a day |
| `CP_LIMIT_ANALYSES_ACTIVE` | 3 | analyses a user may have waiting or running at once |

Only the API reads them, so only `api` is recreated. In a session, run the set-up of
procedure 6 and its check, then:

```bash
set_env CP_LIMIT_UPLOADS_PER_DAY 40
stack up -d --no-build --no-deps api
```

How to tell it worked: the container holds the new number, and reads `healthy` within a minute.

```bash
stack exec api printenv CP_LIMIT_UPLOADS_PER_DAY
stack ps api
```

The API stops at start-up on a limit that is not a whole number greater than zero, and Docker
then restarts it in a loop: if `api` does not become healthy, read its log, correct the line and
run the `up` again. The counts already taken are in Redis and are not reset by this.

The change is in the environment file on the data volume, so it outlasts deploys and a
replacement of the machine.

## 8. Stopping the machine to save money, and starting it again

Use this when the demo is not needed for a while. It needs only the Mac.

> **Not yet carried out.**

What goes on costing while it is stopped: the two disks, the fixed address, the snapshots, the
images and the state bucket, about 9 US dollars a month by the spec's estimate. The nightly
snapshot policy stays on. The fixed address is expected to stay the machine's, so that the DNS
record is not touched (unverified until carried out; the check below shows it).

```bash
REGION=eu-west-1
INSTANCE_ID=$(terraform -chdir=infra output -raw instance_id)
aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$INSTANCE_ID"
```

Do not stop the services with Compose first. Every long-running service has the restart policy
`unless-stopped`, and Docker's documentation says a container with it that was stopped by hand
is not started again when Docker next starts. So a stack stopped with `docker compose stop` or
`down` does not come back. That one which was simply running when the machine stopped does
come back is unverified until carried out.

To start it again:

```bash
aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
```

What to check after, allowing a minute or two for the services:

```bash
SITE_NAME=$(terraform -chdir=infra output -raw site_name)
[ "$(dig +short A "$SITE_NAME")" = "$(terraform -chdir=infra output -raw site_address)" ] && echo "the A record is right"
curl -sS -o /dev/null -w '%{http_code}\n' "https://$SITE_NAME/study"
```

The second line must print `200`. If it does not, look at the services' state (procedure 4).
What Redis held is gone, as after any restart: the day's counts start again, and an analysis
that was queued is marked failed when it is next read.

## 9. Applying a change to the first-boot script or the machine image

Use this when `deploy/first-boot.sh.tftpl` has changed, or a newer Ubuntu image is wanted. It
needs the Mac, and the site is down from the moment the old machine stops until the deploy at
the end succeeds.

> **Not yet carried out.**

`infra/machine.tf` tells Terraform to ignore changes to the machine's `user_data` (the rendered
first-boot script) and its `ami`, so a plain plan proposes nothing for either: a running machine
is never replaced by itself. The script runs once, at a machine's first boot, so the way to
apply a change is to replace the machine on purpose. The new machine is expected to be created
from the script as it stands in the working tree and from `ami_id` as it stands in
`terraform.tfvars`: Terraform's documentation says `ignore_changes` is for planning updates to
an object that exists, and `terraform plan -help` says `-replace` plans a replacement where an
update or nothing would have been planned. Unverified until carried out.

What a replacement keeps: the data volume, and with it the database, the certificates, the
environment file, `current-sha`, `previous-sha`, the release files and the deploy logs; the
fixed address; the images in the registry. The first-boot script on the new machine finds the
file system and the environment file and keeps both.

What it loses: the root disk, and with it Docker's images and containers, the containers'
logs, what Redis held, and the first-boot marker, so the new machine refuses a deploy (status
76) until its own first boot has finished. The machine's id changes, so the repository variable
that names it must be set again.

Note the commit that is running (procedure 4, or the list in procedure 2), and check out the
commit of `main` whose script you mean to apply. For a newer image, put its id in
`terraform.tfvars` (procedure 1 has the command that finds it).

```bash
REGION=eu-west-1
SITE_NAME=$(terraform -chdir=infra output -raw site_name)
SHA=<the commit running now>
terraform -chdir=infra plan -replace=aws_instance.server -out tfplan
```

Read the plan. It should replace `aws_instance.server` and what hangs on its id (the volume's
attachment, the address's association, the network interface's two tags), and change in place
what names the machine (the recovery alarm, the pipeline role's policy). It must not name
`aws_ebs_volume.data` or `aws_eip.server`. Then:

```bash
terraform -chdir=infra apply tfplan
rm infra/tfplan
gh variable set INSTANCE_ID --body "$(terraform -chdir=infra output -raw instance_id)"
```

Wait for the first boot to finish, as in procedure 1 (`Online`, then `first-boot: done` and the
marker file); procedure 11 if it does not. The new machine has no images and no containers, so
deploy. The registry must still hold both images for that commit, or the workflow builds them
again:

```bash
gh workflow run deploy.yml --ref main -f sha="$SHA" -f confirm="$SITE_NAME"
```

How to tell it worked: the deploy succeeds, the site answers, and a signed-in user still sees
their analyses.

## 10. Tearing everything down

Use this to remove the server and everything made for it. It destroys the database and every
snapshot of it, and none of it can be undone. It needs the Mac and access to the DNS host.

> **Not yet carried out.**

Terraform refuses two things: the data volume and the state bucket, each marked
`prevent_destroy`. A `terraform destroy` that includes either stops before it destroys
anything. It also cannot delete an image repository that still holds images. The steps below
go round each, in an order that leaves nothing pointing at something already gone.

```bash
REGION=eu-west-1
INSTANCE_ID=$(terraform -chdir=infra output -raw instance_id)
STATE_BUCKET=$(terraform -chdir=infra/bootstrap output -raw state_bucket)
DATA_VOLUME=$(terraform -chdir=infra output -raw data_volume_id)
SITE_NAME=$(terraform -chdir=infra output -raw site_name)
echo "$DATA_VOLUME $STATE_BUCKET $SITE_NAME"
```

Set these first: the outputs are read from the state, and the steps below take things out of
it.

**The DNS record.** At the DNS host, delete the site's A record first, so that the name does not
go on pointing at an address that will be released to someone else.

**The repository's variables.**

```bash
for name in AWS_ROLE_ARN INSTANCE_ID DEPLOY_DOCUMENT REGISTRY SITE_NAME; do gh variable delete "$name"; done
```

**The images.** Empty both repositories. A repository that lists nothing makes the second
command fail, which is harmless.

```bash
for repository in causalpulse-app causalpulse-web; do
  list=$(mktemp)
  aws ecr list-images --region "$REGION" --repository-name "$repository" --query 'imageIds[*]' --output json >"$list"
  aws ecr batch-delete-image --region "$REGION" --repository-name "$repository" --image-ids "file://$list"
  rm "$list"
done
```

**GitHub's identity provider.** If `create_github_oidc_provider` is `true`, the destroy removes
the provider from the account. If anything else in the account has come to use it, take it out
of the state first, which leaves it in place:

```bash
terraform -chdir=infra state rm 'aws_iam_openid_connect_provider.github[0]'
```

**Everything Terraform will destroy.** Take the data volume out of the state, so the destroy
leaves it alone, then plan the destroy, read it, and apply it. The plan should list every
resource of `infra/` but the volume. That Terraform plans a destroy without objecting to a
volume its configuration still declares and its state no longer holds is unverified until
carried out; the paragraph after the commands is for when it does object.

```bash
terraform -chdir=infra state rm aws_ebs_volume.data
terraform -chdir=infra plan -destroy -out tfplan
```

```bash
terraform -chdir=infra apply tfplan
rm infra/tfplan
```

If Terraform still stops over the volume, the other way is to take the `prevent_destroy` line
out of `aws_ebs_volume.data` in a working copy for the length of the destroy, and not commit
it: `infra/tests/inventory.sh` fails without that line.

**The data volume.** It is now detached and unmanaged. This is the database:

```bash
aws ec2 delete-volume --region "$REGION" --volume-id "$DATA_VOLUME"
```

**The snapshots.** Removing the snapshot policy does not remove the snapshots it made. List
them by the tag the policy gave them, and delete each. Look also for any snapshot or volume made
by hand during a restore (procedure 5).

```bash
aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
  --filters Name=tag:Name,Values=causalpulse-data-nightly \
  --query 'Snapshots[].[SnapshotId,StartTime]' --output text
```

```bash
for snapshot in $(aws ec2 describe-snapshots --region "$REGION" --owner-ids self \
  --filters Name=tag:Name,Values=causalpulse-data-nightly --query 'Snapshots[].SnapshotId' --output text); do
  aws ec2 delete-snapshot --region "$REGION" --snapshot-id "$snapshot"
done
```

**The state bucket**, last, when nothing is left for the state to describe. The bucket is
versioned, so every version and every delete marker has to go before the bucket can. The
console's "Empty" button on the bucket does this; from the command line, run this pair until
both listings are empty (a listing with nothing in it makes its delete fail, which is
harmless):

```bash
list=$(mktemp)
aws s3api list-object-versions --bucket "$STATE_BUCKET" --output json \
  --query '{Objects: Versions[].{Key: Key, VersionId: VersionId}}' >"$list"
aws s3api delete-objects --bucket "$STATE_BUCKET" --delete "file://$list"
aws s3api list-object-versions --bucket "$STATE_BUCKET" --output json \
  --query '{Objects: DeleteMarkers[].{Key: Key, VersionId: VersionId}}' >"$list"
aws s3api delete-objects --bucket "$STATE_BUCKET" --delete "file://$list"
rm "$list"
```

```bash
aws s3api delete-bucket --bucket "$STATE_BUCKET" --region "$REGION"
rm infra/bootstrap/terraform.tfstate infra/bootstrap/terraform.tfstate.backup
```

**How to tell it worked.** Nothing in the region still carries the project's tag, and the name
no longer resolves:

```bash
aws resourcegroupstaggingapi get-resources --region "$REGION" \
  --tag-filters Key=Project,Values=causalpulse \
  --query 'ResourceTagMappingList[].ResourceARN' --output text
dig +short A "$SITE_NAME"
```

Both should print nothing. The first does not see everything (the budget and the roles are not
in the region's list), so also check that `aws budgets describe-budget` for
`causalpulse-monthly` (procedure 1) now fails, and look at the next bill.

## 11. The first boot did not finish

Use this when a deploy ends with status 76, or when, after an apply or a replacement of the
machine, cloud-init reports an error, the marker file is missing, or the machine never shows
as `Online`. It needs the Mac, and a session if the machine can be reached.

> **Not yet carried out.**

The first-boot script runs once, when cloud-init first starts a new machine, and cloud-init
does not run it again. Its last act writes `/var/lib/causalpulse/first-boot-done`. The file is
on the root disk, so a replaced machine has none until its own first boot has finished. The
deploy document looks for it before anything else and, without it, ends with status 76 having
touched nothing: not the lock, not the release directory, not the stack. So a 76 never needs
anything undone; it needs the first boot finished.

**Find how far it got.** From the Mac, whether Systems Manager can reach the machine:

```bash
REGION=eu-west-1
INSTANCE_ID=$(terraform -chdir=infra output -raw instance_id)
aws ssm describe-instance-information --region "$REGION" \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].PingStatus' --output text
```

If it prints `Online`, open a session (procedure 3) and look:

```bash
cloud-init status --long
sudo grep -n 'first-boot:' /var/log/cloud-init-output.log | tail -n 20
sudo tail -n 60 /var/log/cloud-init-output.log
ls -l /var/lib/causalpulse/first-boot-done
```

A status of `running` means it has not ended yet: the script may wait up to ten minutes for the
data volume and several more for the package lists. Wait, and look again. A script that stopped
itself says why in a line beginning `first-boot:` at the end of the log; one stopped by a
command that failed ends with that command's own message.

If it does not print `Online`, there is no way in (the machine has no SSH). The console's
output, which holds what cloud-init printed, can be read from the Mac:

```bash
aws ec2 get-console-output --region "$REGION" --instance-id "$INSTANCE_ID" --latest --output text | tail -n 80
```

How much of the boot that output holds, and how soon after it, is unverified until carried
out. A machine that cannot be reached and whose first boot failed is replaced: procedure 9,
which gives a new machine a first boot of its own and keeps the data volume.

**Finish it.** The script may be run again by hand, as root: every step looks at what is there
before it acts, it formats the data volume only when two probes agree that the volume holds
nothing, and it never replaces an environment file. First mend what stopped it, if the log
names something that will not mend itself (a volume that holds something unexpected stops the
script on purpose, and is a person's decision). Then, in a session, find the script as
cloud-init kept it and run it. Where cloud-init keeps it is unverified until carried out; the
first command shows what is there, and the script is the file that begins with a comment
saying it is the server's first boot.

```bash
sudo ls -l /var/lib/cloud/instance/scripts/
sudo head -n 3 /var/lib/cloud/instance/scripts/<the file listed>
```

```bash
sudo bash /var/lib/cloud/instance/scripts/<the file listed>
```

If no copy can be found on the machine, replace the machine (procedure 9) and do not paste the
script in by hand: the one on the machine has the site's name and the data volume's id written
into it by Terraform.

How to tell it worked: the script's last line is `first-boot: done`, and these three print
their sentences.

```bash
test -e /var/lib/causalpulse/first-boot-done && echo "the first boot finished"
sudo test -s /srv/causalpulse/.env && echo "the environment file is there"
sudo docker compose version >/dev/null && echo "Docker and Compose are installed"
```

Then deploy (procedure 2). Do not make the marker file by hand to get a deploy through: it is
the only thing that says Docker, the mounts and the environment file are all in place.
