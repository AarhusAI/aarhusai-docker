# Docker setup for Open WebUI

Docker deployment wrapper for a customized fork of [Open WebUI](https://docs.openwebui.com/),
tailored for Aarhus Kommune. The fork lives at
[AarhusAI/open-webui](https://github.com/AarhusAI/open-webui) and is cloned into `./open-webui/`
during install.

This repo does **not** contain the Open WebUI source — it manages Docker orchestration,
configuration, and a patch system applied on top of tagged upstream releases.

## Install

```shell
task install
```

`task install` clones the required repositories on first run (when their
directories don't already exist) and then applies patches and starts the
stack. On subsequent runs the existing checkouts are reused.

The repositories cloned are:

- [open-webui](https://github.com/AarhusAI/open-webui) — into `open-webui/`, checked out at `OPEN_WEBUI_VERSION`
- [ingestion-service](https://github.com/AarhusAI/ingestion-service) — into `ingestion-service/`, checked out at `INGESTION_SERVICE_VERSION`
- [retrieval-agent](https://github.com/AarhusAI/retrieval-agent) — into `retrieval-agent/`, checked out at `RETRIEVAL_AGENT_VERSION`
- [search-agent](https://github.com/AarhusAI/search-agent) — into `search-agent/`, checked out at `SEARCH_AGENT_VERSION`

The individual clone steps are also available as standalone tasks:

```shell
task git:clone              # open-webui
task git:clone:ingestion    # ingestion-service
task git:clone:retrieval    # retrieval-agent
task git:clone:search       # search-agent
```

### Task install

### ARM hosts (Apple Silicon, ARM Linux)

`task compose` auto-detects the host architecture. On `arm64` it transparently layers
`docker-compose.arm.yml` on top of `docker-compose.yml`, which pins the openwebui image build
to `linux/arm64`. No flags or environment tweaks required.

### Storage

The `openwebui` MinIO bucket is created automatically by the `minio-init` service on first
boot — no manual step required.

## Usage

```shell
task open                     # open the app in your browser
task compose -- up --detach   # bring services up
task compose -- logs -f openwebui
```

## Configuring connections and models

The `litellm_config.yaml` file should have the proper API keys. If these were not set from
the get-go, restart the container with the new values:

```shell
task compose -- restart litellm
```

Then go to `/admin/settings/connections` and configure the connection: OpenAI API base URL
`http://litellm:4000/v1`, API key `sk-1234` (see `docker-compose.yml`). Models then appear
under `/admin/settings/models`.

## OIDC

An (incomplete) OIDC dev setup with a mock server is included in `docker-compose.oidc.yml`.
Currently broken with:

```text
Unhandled exception. System.IO.FileLoadException: Could not load file or assembly 'OpenIdConnectServerMock, Version=0.10.1.0, Culture=neutral, PublicKeyToken=null'.
```

## Guardrails

Custom LiteLLM guardrails live in `guardrails/` and are bind-mounted into the `litellm`
container. Run the test suite with:

```shell
task guardrails:test
```

## Patching

This project needs changes the upstream project may not accept. To stay in sync with
upstream while tracking local modifications, we manage patches as GitHub PR diffs on the
fork, applied via `git apply`.

There are **two patch sets** defined in `Taskfile.yml`:

- `PATCHES` — generic features (pinned banner, TTS auth, extra permissions, OAuth fixes,
  model search, role/group claims, RAG template).
- `PATCHES_AARHUS` — org-specific (external RAG service, admin-only API keys,
  branding/UI, citation modal, agentic search).

Each patch corresponds to a PR on the fork repo with a `feature/*` branch. See the open
PRs at [AarhusAI/open-webui/pulls](https://github.com/AarhusAI/open-webui/pulls). Labels
mark patches, approval state, and pending-upstream status.

Apply patches:

```shell
task patch         # base + Aarhus
task patch:base    # base only
task patch:aarhus  # Aarhus only
```

### Updating to a new upstream release

When a new upstream release is published, the patches need to be rebased onto the new tag.

1. Sync `dev` and `main` branches with upstream (do this on GitHub).
2. Sync tags (GitHub doesn't propagate them):

   ```shell
   task git:sync:tags
   ```

3. Create a new base branch from the new upstream tag:

   ```shell
   task git:checkout:dev
   git checkout -b upstream/<release tag>
   git push origin -u upstream/<release tag>
   ```

4. Update each PR's base branch to the new `upstream/<release tag>` on GitHub.
5. Update `OPEN_WEBUI_VERSION` and `OPEN_WEBUI_PREV_VERSION` in `Taskfile.yml`.
6. Rebase all patch branches automatically:

   ```shell
   task patches:rebase
   ```

7. If conflicts arise, resolve them locally per branch (`git add .` + `git rebase --continue`).
8. Force-push the rebased branches:

   ```shell
   task patches:force
   ```

If you need to discard local patch-branch state and reset back to what's on GitHub:

```shell
task patches:reset
```

### Rules

1. Prefix all commits with a ticket number, so a single change can be isolated or removed
   when resolving merge conflicts.
2. Always first try to open a PR against
   [open-webui/open-webui](https://github.com/open-webui/open-webui). If declined, open a
   PR against the corresponding `feature/*` branch on the fork.
3. Wrap patches in comments clearly stating that it is a patch, with a short note on
   what and why.

## Production

Production builds are multi-platform (`linux/amd64` + `linux/arm64`) and push to a
container registry. The image tag comes from `PROD_OPEN_WEBUI_VERSION` in `Taskfile.yml` —
update it before building.

Two registries are supported:

```shell
task prod:build            # itkdev/openwebui on Docker Hub (full patch set: base + Aarhus)
task prod:build:aarhusai   # ghcr.io/os2ai/open-webui (base patches only)
```
