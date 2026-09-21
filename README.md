# AarhusAI developer setup

Docker/Taskfile orchestration for Aarhus Kommune's fork of [Open WebUI](https://github.com/AarhusAI/open-webui). It
clones the fork plus sibling agent/RAG repos, applies a patch system on top of a tagged upstream release, and runs the
stack locally or builds production images; models are served by an external LiteLLM gateway on the GPU servers, with
MinIO/Garage for object storage, Postgres, Redis, Qdrant and (on the gateway) custom LiteLLM guardrails.

It contains **no application source** — the Open WebUI code lives in the `AarhusAI/open-webui` fork and is checked out
into `open-webui/` at build time. This repo only orchestrates: cloning, patching, composing, backups and image builds.

## Architecture

### Services

Two compose stacks exist. `docker-compose.yml` is the **local dev** stack (builds `open-webui/` from source, Garage for
S3). `docker-compose.server.yml` is the **production/server** stack (pulls the prebuilt `itkdev/openwebui` image, MinIO
for S3, SearXNG). Overlays add ARM support and agents. The LiteLLM model gateway is **not** part of these compose
files — it runs on the GPU servers and Open WebUI reaches it over the network
(see [Connecting Open WebUI to LiteLLM](#connecting-open-webui-to-litellm)).

| Service                 | Stack  | Role                                                                                      |
|-------------------------|--------|-------------------------------------------------------------------------------------------|
| `openwebui`             | both   | Open WebUI backend + frontend; routed via Traefik on `COMPOSE_DOMAIN`, internal port 8080 |
| `postgres`              | both   | Open WebUI application database (Postgres 17)                                             |
| `redis`                 | both   | Websocket manager + cache                                                                 |
| `qdrant`                | both   | Vector store for RAG (ports 6333/6334)                                                    |
| `kreuzberg`             | both   | Document text extraction service (port 8000)                                              |
| `gotenberg`             | both   | Office/HTML → PDF conversion (port 3000)                                                  |
| `retrieval`             | both   | External RAG retrieval agent (`RAG/retrieval-agent`, port 8000)                           |
| `ingestion`             | both   | External RAG ingestion service (`RAG/ingestion-service`, port 8000)                       |
| `node`                  | dev    | One-shot Node container for the Vite frontend build                                       |
| `garage`                | dev    | S3-compatible object storage (Garage, API 3900 / RPC 3901)                                |
| `garage-init`           | dev    | One-shot: creates the S3 bucket and access keys in Garage                                 |
| `minio`                 | server | S3-compatible object storage (API 9000 / console 9001)                                    |
| `minio-init`            | server | One-shot: creates the `openwebui` bucket in MinIO                                         |
| `searxng`               | server | Metasearch backend for Open WebUI web search (port 8080)                                  |
| `search-agent`          | agents | Web-search MCP tool server (port 8001)                                                    |
| `eventdatabase-agent`   | agents | Event-database MCP tool server (port 8000)                                                |
| `retsinformation-agent` | agents | Retsinformation-API MCP tool server (port 8000)                                           |
| `eu-funding-agent`      | agents | EU Funding & Tenders Portal MCP tool server (port 8000)                                   |

All containers join the internal `app` network; `openwebui` also joins the external `frontend` network (Traefik,
provided by `itkdev-docker-compose`). Services publish container ports only — there are no fixed host port mappings;
reach the UI through Traefik at `http://${COMPOSE_DOMAIN}`.

### Compose layering

`task compose` runs `docker-compose.yml` on amd64 and `docker-compose.yml -f docker-compose.arm.yml` on arm64. Other
overlays are added with extra `-f` flags (or via `include:`):

- **base** — `docker-compose.yml`: dev stack, builds `openwebui` from `open-webui/`.
- **arm** — `docker-compose.arm.yml`: overrides `openwebui.build.platforms` to `linux/arm64`. Auto-applied by
  `task compose` on arm64.
- **server** — `docker-compose.server.yml`: standalone production stack (prebuilt image, MinIO, SearXNG). Not layered on
  base.
- **agents** — `docker-compose.agents.yml` (dev, builds from `agents/*`) / `docker-compose.server.agents.yml` (prod,
  prebuilt images). Both override `openwebui.TOOL_SERVER_CONNECTIONS` to register the MCP tool servers.

```mermaid
graph TD
    Traefik[Traefik / frontend net] --> OW[openwebui]
    OW --> PG[(postgres)]
    OW --> REDIS[(redis)]
    OW --> QD[(qdrant)]
    OW --> KRZ[kreuzberg]
    OW --> GOT[gotenberg]
    OW --> RET[retrieval]
    OW --> ING[ingestion]
    OW -->|S3| S3[garage / minio]
    OW -->|OpenAI API| LLM[litellm gateway - GPU servers, external]
    OW -->|web search| SX[searxng]
    OW -->|MCP tools| SA[search-agent]
    OW -->|MCP tools| EV[eventdatabase-agent]
    OW -->|MCP tools| RI[retsinformation-agent]
    OW -->|MCP tools| EU[eu-funding-agent]
    RET --> QD
    ING --> QD
    ING -->|S3| S3
```

## Prerequisites

- **Docker** with the Compose plugin (`docker compose`).
- **Task** ([go-task](https://taskfile.dev)) — the task runner.
- **Git**.
- **`itkdev-docker-compose`** — the default compose wrapper (`DOCKER_COMPOSE` var). It provides Traefik and the external
  `frontend` network. To use plain Compose instead, set `TASK_DOCKER_COMPOSE="docker compose"` (you must then supply the
  `frontend` network and routing yourself).
- **curl** — patch tasks pipe PR `.diff` URLs into `git apply`.
- Access to the `AarhusAI/*` GitHub repos (fork + agents + RAG services).

**ARM / Apple Silicon:** `task compose` auto-adds `docker-compose.arm.yml` on arm64 hosts, which builds `openwebui` for
`linux/arm64`. The `db:*` and `s3:*` tasks apply the same overlay automatically on arm64.

## Quick start

```bash
# 1. Clone this repo and enter it
git clone <this-repo> openwebui-docker && cd openwebui-docker

# 2. Create .env from the template
task copy:config          # copies .env.default -> .env if missing

# 3. Replace placeholders in .env (CHANGE_ME_NOW / XXXX) with real values (see Configuration)

# 4. Clone the fork, reset to the pinned tag, apply patches,
#    clone RAG services, pull images, start, build the frontend
task install
```

`task install` runs: `git:clone` (if needed) → `git:reset` → `patch:aarhus` → `rag:clone` → `copy:config` →
`compose -- pull` → `compose -- up --detach --force-recreate` → `wui:frontend:build`.

`task copy:config` only creates `.env` if missing; it does not overwrite. `.env.default` is a full template — every
variable is present, secrets redacted as `XXXX`/`sk-XXXX` and `WEBUI_SECRET_KEY=CHANGE_ME_NOW`. You don't add variables;
you **replace the redacted placeholders** with real values (local-dev set in 1Password, personalized API keys from
devops). The admin rows ship working defaults (`noreply@itkdev.dk` / `admin`) — change them for anything but throwaway
local use. The placeholders that block a working stack:

- `WEBUI_SECRET_KEY` (ships `CHANGE_ME_NOW`)
- `OPENAI_API_BASE_URLS` / `OPENAI_API_KEYS` — LiteLLM gateway
- `RETRIEVAL_API_KEY` (used as `RAG_EXTERNAL_RETRIEVAL_API_KEY`) and the other `*_API_KEY`s for the RAG/embedding
  features you use
- the agent `*_SERVICE_API_KEY`s when running an agents overlay

Then open the UI:

```bash
task open   # opens http://${COMPOSE_DOMAIN} (default webui.local.itkdev.dk)
```

## Configuration

### `.env`

`.env.default` (copied to `.env` by `task copy:config`) mirrors the 1Password developer note with secrets redacted.
Variables:

| Variable                                                                              | Purpose                                                                       | Default                                                | Required             |
|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------|----------------------|
| `COMPOSE_PROJECT_NAME`                                                                | Compose project name / Traefik router prefix                                  | `openwebui`                                            | yes                  |
| `COMPOSE_DOMAIN`                                                                      | Host Traefik routes the UI on; base of `BASE_URL`                             | `webui.local.itkdev.dk`                                | yes                  |
| `WEBUI_SECRET_KEY`                                                                    | Open WebUI session/JWT signing key                                            | `CHANGE_ME_NOW`                                        | yes                  |
| `OAUTH_CLIENT_ID`                                                                     | OIDC client id                                                                | `XXXXX`                                                | yes (for OIDC login) |
| `OAUTH_CLIENT_SECRET`                                                                 | OIDC client secret                                                            | `XXXX`                                                 | yes (for OIDC login) |
| `OPENID_PROVIDER_URL`                                                                 | OIDC discovery URL (Azure B2C)                                                | `https://aarhuskommunetest.b2clogin.com/...`           | yes (for OIDC login) |
| `OAUTH_PROVIDER_NAME`                                                                 | Display name of the OAuth provider                                            | `Aarhus Kommune`                                       | no                   |
| `OAUTH_SCOPES`                                                                        | OAuth scopes requested                                                        | `openid email`                                         | no                   |
| `OAUTH_EMAIL_CLAIM`                                                                   | Claim used as email                                                           | `upn`                                                  | no                   |
| `OAUTH_ROLES_CLAIM`                                                                   | Claim used for roles                                                          | `role`                                                 | no                   |
| `OAUTH_ADMIN_ROLES`                                                                   | Roles mapped to admin                                                         | `admin`                                                | no                   |
| `OAUTH_ALLOWED_ROLES`                                                                 | Roles allowed to log in                                                       | `admin,end-user,local-admin,builder`                   | no                   |
| `ENABLE_OAUTH_ROLE_MANAGEMENT`                                                        | Manage roles from OAuth claims                                                | `true`                                                 | no                   |
| `AAK_OAUTH_DEBUG_FORCE_ROLE`                                                          | Force a role for debugging (commented)                                        | —                                                      | no                   |
| `ENABLE_LOGIN_FORM`                                                                   | Show the local login form                                                     | `TRUE`                                                 | no                   |
| `ENABLE_SIGNUP`                                                                       | Allow local signup                                                            | `TRUE`                                                 | no                   |
| `OPENAI_API_BASE_URLS`                                                                | LiteLLM gateway base URL(s), `;`-separated                                    | `https://stgxxxx.itkdev.dk/v1;https://xxxx.itkdev.dk`  | yes                  |
| `OPENAI_API_KEYS`                                                                     | Gateway key(s), `;`-separated                                                 | `sk-XXXX;sk-XXXX`                                      | yes                  |
| `RAG_OPENAI_API_KEY`                                                                  | Key for the embedding endpoint (`RAG_OPENAI_API_BASE_URL`)                    | `XXXX`                                                 | yes (for RAG)        |
| `TTS_API_KEY`                                                                         | Key for the TTS endpoint                                                      | `XXXX`                                                 | no                   |
| `STT_API_KEY`                                                                         | Key for the STT endpoint                                                      | `XXXX`                                                 | no                   |
| `GLOBAL_LOG_LEVEL`                                                                    | Open WebUI log level                                                          | `DEBUG`                                                | no                   |
| `ENABLE_PERSISTENT_CONFIG`                                                            | Persist config in DB vs. env-driven                                           | `false`                                                | no                   |
| `ENABLE_OTEL` / `ENABLE_OTEL_METRICS`                                                 | OpenTelemetry export toggles                                                  | `false`                                                | no                   |
| `WEBUI_ADMIN_EMAIL`                                                                   | Bootstrap admin email                                                         | `noreply@itkdev.dk`                                    | yes                  |
| `WEBUI_ADMIN_PASSWORD`                                                                | Bootstrap admin password                                                      | `admin`                                                | yes                  |
| `SEARCH_AGENT_*`                                                                      | Web-search agent: provider, keys, LLM base/key/model, debug flags             | see file (`staan`, `AarhusAI-default-v2`, keys `XXXX`) | for search-agent     |
| `INGESTION_API_KEY`                                                                   | Auth key Open WebUI uses to call the ingestion service                        | `XXXX`                                                 | for ingestion        |
| `INGESTION_DEBUG` / `INGESTION_LOG_LEVEL*`                                            | Ingestion service logging                                                     | `false` / `INFO`                                       | no                   |
| `EXTERNAL_INGESTION_ENGINE`                                                           | Enable external ingestion engine                                              | `external`                                             | for ingestion        |
| `INGESTION_EMBEDDING_API_KEY`                                                         | Ingestion embedding key                                                       | `XXXX`                                                 | for ingestion        |
| `INGESTION_VISION_LLM_*`                                                              | Ingestion vision-LLM base/key/model                                           | see file (`AarhusAI-default-v2`)                       | for ingestion        |
| `RETRIEVAL_API_KEY`                                                                   | Auth key Open WebUI uses to call retrieval (`RAG_EXTERNAL_RETRIEVAL_API_KEY`) | `XXXX`                                                 | yes                  |
| `RETRIEVAL_LOG_LEVEL*`                                                                | Retrieval agent logging                                                       | `INFO`                                                 | no                   |
| `RETRIEVAL_AGENT_*`                                                                   | Retrieval agent LLM base/key/model                                            | see file (`AarhusAI-default-v2`)                       | for RAG              |
| `RETRIEVAL_EMBEDDING_*`                                                               | Retrieval embedding base/key/model                                            | see file (`multilingual-e5-large`)                     | for RAG              |
| `RETRIEVAL_ENABLE_RERANKING`                                                          | Enable reranking                                                              | `true`                                                 | no                   |
| `RETRIEVAL_RERANKER_*`                                                                | Reranker model/base/key                                                       | see file (`bge-reranker-v2-m3`)                        | for RAG              |
| `EVENT_DATABASE_SERVICE_API_KEY`                                                      | Bearer key Open WebUI uses for the event-database MCP tool                    | `XXXX`                                                 | for agents           |
| `EVENT_DATABASE_AGENT_API_KEY` / `EVENT_DATABASE_BASE_URL` / `EVENT_DATABASE_API_KEY` | Event-database agent's own LLM key, upstream API base + key                   | see file                                               | for event agent      |
| `RETSINFORMATION_SERVICE_API_KEY`                                                     | Bearer key for the retsinformation MCP tool                                   | `XXXX`                                                 | for agents           |
| `RETSINFORMATION_LLM_API_KEY`                                                         | Retsinformation agent LLM key                                                 | `sk-XXXX`                                              | for retsinfo agent   |
| `EU_FOUNDING_SERVICE_API_KEY`                                                         | Bearer key for the EU-funding MCP tool                                        | `XXXX`                                                 | for agents           |
| `EU_FOUNDING_AGENT_API_KEY`                                                           | EU-funding agent LLM key                                                      | `sk-XXXX`                                              | for EU-funding agent |

Values shown as `XXXX` / `sk-XXXX` / `CHANGE_ME_NOW` are redacted placeholders — get the real ones from 1Password /
devops. "Required" is relative to the feature: the core stack needs the admin/secret/gateway/retrieval rows; agent and
RAG rows only matter when those overlays/features are enabled.

A ready-made `.env` for local development (working OIDC on the local itkdev domain, gateway keys, etc.) is in
1Password — secure note **"Open-webui local developer (.env)"**. Personalized API keys are obtained from devops. Keep
real secrets out of git; `.env` is git-ignored and `.env.default` is the only committed template.

### `litellm_config.yaml`

`litellm_config.yaml` (and its committed template `litellm_config.example.yaml`) configures the LiteLLM gateway —
`model_list` and `guardrails`. The gateway now runs on the GPU servers, not from these compose files; how this file and
`guardrails/` reach that gateway is not defined in this
repo. <!-- TODO: verify — confirm how litellm_config.yaml / guardrails/ are deployed/synced to the GPU-server LiteLLM. -->
It has two blocks:

- `model_list:` — each entry maps a friendly `model_name` to an upstream `litellm_params.model` + `api_base` + `api_key`
  (and optional `api_version`). Attach guardrails per model with a `guardrails:` list.
- `guardrails:` — registers `MessageTrimmingGuardrail` (`message_overflow.MessageTrimmingGuardrail`, `mode: pre_call`)
  and its `default_config`. See [`guardrails/README.md`](guardrails/README.md).

### Connecting Open WebUI to LiteLLM

Open WebUI talks to LiteLLM over the OpenAI-compatible API (`ENABLE_OPENAI_API: true`):

- `OPENAI_API_BASE_URLS` — LiteLLM gateway base URL (s) on the GPU servers, `;`-separated (`.env.default` ships a
  staging + prod pair, redacted). Set from the 1Password `.env`.
- `OPENAI_API_KEYS` — matching gateway key (s). Personalized keys come from devops.

Semicolon-separate the lists to configure multiple gateways.

## Tasks reference

Run `task` (or `task --list-all`) to list everything. Tasks that reset git, empty databases, overwrite storage or push
images prompt for confirmation.

### compose / lifecycle

| Task                | Description                                                                                                          |
|---------------------|----------------------------------------------------------------------------------------------------------------------|
| `compose -- <args>` | Run `docker compose` with the right base/arm file. Example: `task compose -- up --detach`                            |
| `install`           | Full local install: clone fork, reset to tag, apply Aarhus patches, clone RAG, copy config, pull, up, build frontend |
| `open`              | Open the UI in a browser (`http://${BASE_URL}`)                                                                      |
| `copy:config`       | Create `.env` from `.env.default` if missing                                                                         |

### git (fork checkout in `open-webui/`)

| Task               | Description                                                                           |
|--------------------|---------------------------------------------------------------------------------------|
| `git:clone`        | Clone `AarhusAI/open-webui` and check out `OPEN_WEBUI_VERSION`                        |
| `git:reset`        | Hard-reset the checkout and clean, then check out `OPEN_WEBUI_VERSION`                |
| `git:checkout:dev` | Reset and check out the `dev` branch                                                  |
| `git:sync:tags`    | Add the `open-webui/open-webui` upstream remote, fetch + push tags, remove the remote |

### agents / rag (sibling repos)

| Task            | Description                                                              |
|-----------------|--------------------------------------------------------------------------|
| `agents:clone`  | Clone every agent repo in `AGENTS` into `agents/*` (skips existing)      |
| `agents:update` | `git pull` each cloned agent                                             |
| `rag:clone`     | Clone the RAG services in `RAG` (`ingestion-service`, `retrieval-agent`) |
| `rag:update`    | `git pull` each cloned RAG service                                       |

### patch (apply patch sets to the checkout)

| Task                   | Description                                                                    |
|------------------------|--------------------------------------------------------------------------------|
| `patch:base`           | Apply the `PATCHES` (general/base) set via `git apply`                         |
| `patch:aarhus`         | Apply `patch:base` then the `PATCHES_AARHUS` set                               |
| `patch:os2ai`          | Apply `patch:base` then the `PATCHES_OS2` set                                  |
| `patches:download`     | Download all patch diffs to `patches/${OPEN_WEBUI_VERSION}/{base,aarhus,os2}/` |
| `patches:download:one` | (internal) Download one patch diff into a target dir                           |

### patches (branch maintenance — danger zone)

| Task             | Description                                                                                           |
|------------------|-------------------------------------------------------------------------------------------------------|
| `patches:rebase` | Rebase `PATCHES` + `PATCHES_AARHUS` branches from `OPEN_WEBUI_PREV_VERSION` onto `OPEN_WEBUI_VERSION` |
| `patches:reset`  | Reset `PATCHES` + `PATCHES_AARHUS` branches to their `origin/<branch>`                                |
| `patches:force`  | Force-push all `PATCHES` + `PATCHES_AARHUS` branches                                                  |

### wui (frontend/backend build)

| Task                     | Description                                                                                                  |
|--------------------------|--------------------------------------------------------------------------------------------------------------|
| `wui:npmrc`              | (internal) Raise the Node heap in `open-webui/.npmrc` for the Vite build (`NODE_BUILD_MEMORY`, default 8192) |
| `wui:backend:build`      | Bump npmrc, then `compose build --pull`                                                                      |
| `wui:frontend:build`     | Bump npmrc, `npm ci --force` + `npm run build` in the `node` container                                       |
| `wui:frontend:dev:build` | Same, with `ENV=dev`                                                                                         |

### db (Open WebUI Postgres)

| Task              | Description                                                                   |
|-------------------|-------------------------------------------------------------------------------|
| `db:dump`         | `pg_dump` the Open WebUI database (default `dump.sql`)                        |
| `db:import`       | Empty the database and import a dump (default `dump.sql`)                     |
| `db:users:export` | Export users (name, email, role, last_active_at) to CSV (default `users.csv`) |

### s3 (Open WebUI storage)

| Task        | Description                                                         |
|-------------|---------------------------------------------------------------------|
| `s3:dump`   | Mirror the `openwebui` bucket to a local folder (default `s3-dump`) |
| `s3:import` | Restore a dump folder into the `openwebui` bucket (overwrites)      |

### prod (image builds)

| Task                  | Description                                                                                              |
|-----------------------|----------------------------------------------------------------------------------------------------------|
| `prod:prepare`        | Reset git, apply patches (`PATCH_TASK`, default `patch:base`), bump npmrc                                |
| `prod:build:shared`   | (internal) Prepare, build `openwebui`, tag + push `IMAGE_NAME` at `PROD_OPEN_WEBUI_VERSION` and `latest` |
| `prod:build:aarhusai` | Build + push `itkdev/openwebui` with `patch:aarhus`                                                      |
| `prod:build:os2ai`    | Build + push `ghcr.io/os2ai/open-webui` with `patch:os2ai`                                               |

## Patch system

Open WebUI is a tagged upstream checkout in `open-webui/`; Aarhus/OS2 changes are applied on top as patches rather than
committed into the tree. Each patch is a GitHub PR `.diff` on a fork, fetched with `curl` and applied with `git apply`.
Versions are pinned in `Taskfile.yml`: `OPEN_WEBUI_VERSION` (`v0.11.3`), `OPEN_WEBUI_PREV_VERSION` (`v0.11.2`),
`PROD_OPEN_WEBUI_VERSION` (`v0.11.3-2`).

### Patch sets

- **`PATCHES` (base)** — general fixes/features, from `os2ai/open-webui` PRs #1–6. Branches: `patch/pinned-info-banner`,
  `patch/tts-token`, `patch/extra-permissions`, `patch/oauth_token_expire`, `feature/agentic-search`,
  `feature/rag-query-template`.
- **`PATCHES_AARHUS`** — Aarhus-specific, from `AarhusAI/open-webui` PRs #33, #34, #55, #41, #42, #44 (OIDC
  roles/groups, keep role names, external RAG, admin-only API keys, front-end login banner/logo, citation-modal URL).
- **`PATCHES_OS2`** — OS2 additions, from `AarhusAI/open-webui` PR #56
  (`lasseborly:feature/add-enable-prompt-suggestions`).

`patch:aarhus` = base + Aarhus; `patch:os2ai` = base + OS2. Each patch maps to a `feature/*` or `patch/*` branch on the
fork and a PR whose `.diff` is the applied artifact.

### Applying patches

```bash
task git:reset        # clean checkout at OPEN_WEBUI_VERSION
task patch:aarhus     # or patch:os2ai / patch:base
```

Downloaded snapshots for offline reference / review live under `patches/<version>/{base,aarhus,os2}/` (populate with
`task patches:download`).

### Rebasing onto a new upstream release

1. Bump `OPEN_WEBUI_VERSION` / `OPEN_WEBUI_PREV_VERSION` (and `PROD_OPEN_WEBUI_VERSION`) in `Taskfile.yml`.
2. Sync tags into the fork: `task git:sync:tags` (assumes `main`/`dev` already synced with upstream on GitHub).
3. Ensure the fork has an `upstream` remote with the new/old tags fetched.
4. `task patches:rebase` — for each branch: `git rebase --onto upstream/<new> upstream/<prev>`. Resolve conflicts per
   branch.
5. `task patches:force` to publish the rebased branches (force-push).
6. `task patches:download` to refresh the offline snapshots.
7. Reinstall / rebuild and verify.

### Contribution rules

- **Upstream-first.** Every change is a PR on the fork (`AarhusAI/open-webui` or `os2ai/open-webui`); the applied
  artifact is that PR's `.diff`. Add a new patch by adding its PR to `PATCHES`, `PATCHES_AARHUS` or `PATCHES_OS2` with
  its branch name.
- **Ticket-prefixed / referenced commits.** Patch changes carry a reference to their PR and ticket, e.g.
  `# PATCH (AarhusAI/open-webui#41, ticket 5511): …`.
- **Comment-wrapped patches.** Wrap each change in identifying comments so it survives rebases and stays greppable, e.g.
  `<!-- PATCH ADD BANNERS TO CHAT INPUT -->` … `<!-- /PATCH ADD BANNERS TO CHAT INPUT -->` in Svelte, or `# PATCH (...)`
  blocks in Python.

## Optional components

### Agents

Agent MCP tool servers are cloned into `agents/*` (`task agents:clone`, from the `AGENTS` list) and run via an overlay:

```bash
task compose -- -f docker-compose.agents.yml up --detach          # dev, builds from agents/*
# server: -f docker-compose.server.agents.yml (prebuilt images)
```

The overlay overrides `openwebui.TOOL_SERVER_CONNECTIONS` to register `search-agent` (websearch, no auth),
`eventdatabase-agent`, `retsinformation-agent` and `eu-funding-agent` (bearer-auth, keys from `*_SERVICE_API_KEY`). The
`office-agent` repo is in the `AGENTS` clone list but has no compose service. RAG services (`retrieval`, `ingestion`)
are part of the base/server stacks, not this overlay.

### OAuth / OIDC

Open WebUI authenticates against an external OIDC provider (Azure B2C in production) via the `OAUTH_*` /
`OPENID_PROVIDER_URL` variables on the `openwebui` service. Set `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`,
`OPENID_PROVIDER_URL`, `OAUTH_PROVIDER_NAME` and the claim/role mappings in `.env`.

OIDC login works on the local itkdev domain using the ready-made `.env` (see [`.env`](#env)); production uses Azure B2C.
The former `docker-compose.oidc.yml` mock identity provider has been removed.

### Guardrails

`guardrails/` holds LiteLLM custom guardrails for the LiteLLM gateway. It currently ships `MessageTrimmingGuardrail`
(`message_overflow.py`), which trims oversized histories to the target model's context window and repairs
tool-call/tool-response pairings. Full detail: [`guardrails/README.md`](guardrails/README.md).

The gateway runs on the GPU servers, not from these compose files; how `guardrails/` reaches it is not defined in this
repo. <!-- TODO: verify — confirm the guardrails deployment path to the GPU-server LiteLLM. --> There is no longer a
`guardrails:test` task — it was removed with `docker-compose.extra.yml`. To test locally, run `pytest guardrails/` in a
container with the `litellm` package installed.

## Production builds

Production images build the `openwebui` service from `docker-compose.yml`
(`COMPOSE_BAKE=true docker compose --file docker-compose.yml build --no-cache --pull openwebui`), then tag and push at
`PROD_OPEN_WEBUI_VERSION` and `latest`. Each build first runs `prod:prepare` (git reset → apply patch set → bump npmrc):

| Task                  | Image                      | Patch set                      |
|-----------------------|----------------------------|--------------------------------|
| `prod:build:aarhusai` | `itkdev/openwebui`         | `patch:aarhus` (base + Aarhus) |
| `prod:build:os2ai`    | `ghcr.io/os2ai/open-webui` | `patch:os2ai` (base + OS2)     |

The server stack (`docker-compose.server.yml`) consumes `itkdev/openwebui:${OPENWEBUI_VERSION:-latest}`.

Build for `linux/arm64` by adding `-f docker-compose.arm.yml` (auto-applied by `task compose` on arm64 hosts), which
sets `openwebui.build.platforms`.

<!-- TODO: verify — the prod build tasks build for the host architecture only; there is no buildx multi-arch/manifest step. Confirm whether amd64 + arm64 manifests are expected. -->

## Troubleshooting

- **Ports / Traefik conflicts.** Services publish container ports only; the UI is reached through Traefik (from
  `itkdev-docker-compose`) on `${COMPOSE_DOMAIN}`. If startup fails on 80/443, another Traefik/webserver holds them —
  stop it, or run with `TASK_DOCKER_COMPOSE="docker compose"` and your own routing. `postgres`/`minio`/`redis` conflicts
  mean a leftover stack: `task compose -- down`.
- **MinIO/Garage bucket missing.** The bucket is created by `minio-init` (server) / `garage-init` (dev). If uploads 500
  with a missing-bucket error, re-run init: `task compose -- up minio-init` (or `garage-init`), or recreate manually
  with `mc mb myminio/openwebui`. The bucket name is `openwebui`.
- **Patch conflicts.** `git apply` fails when the checkout drifts from the patch's base. Run `task git:reset` first so
  the tree is clean at `OPEN_WEBUI_VERSION`, then re-apply. After an upstream bump, rebase the branches
  (`task patches:rebase`) before re-applying.
- **ARM builds.** On Apple Silicon, `task compose` adds `docker-compose.arm.yml` automatically. If you invoke
  `docker compose` directly, add `-f docker-compose.arm.yml` yourself or the `openwebui` build targets the wrong
  platform. Frontend build OOM → raise `NODE_BUILD_MEMORY` (default 8192, applied by `wui:npmrc`).
- **LiteLLM keys not picked up.** Open WebUI reads `OPENAI_API_BASE_URLS` + `OPENAI_API_KEYS` from the environment.
  After editing `.env`, recreate the container (`task compose -- up --detach --force-recreate openwebui`) — a plain
  restart keeps the old env. Ensure the key matches the gateway (personalized keys from devops; the local-dev `.env` in
  1Password is preconfigured).
- **Guardrail changes.** `guardrails/` targets the LiteLLM gateway on the GPU servers, not these compose files. Watch
  traces with `debug: true` in `litellm_config.yaml` and the gateway's logs. There is no `guardrails:test` task anymore
  (removed with `docker-compose.extra.yml`).

## Related repositories

- Fork: [AarhusAI/open-webui](https://github.com/AarhusAI/open-webui) (OS2 PRs
  against [os2ai/open-webui](https://github.com/os2ai/open-webui))
-
RAG: [AarhusAI/ingestion-service](https://github.com/AarhusAI/ingestion-service), [AarhusAI/retrieval-agent](https://github.com/AarhusAI/retrieval-agent)
-
Agents: [AarhusAI/search-agent](https://github.com/AarhusAI/search-agent), [AarhusAI/eventdatabasen-agent](https://github.com/AarhusAI/eventdatabasen-agent), [AarhusAI/retsinformation-api-agent](https://github.com/AarhusAI/retsinformation-api-agent), [AarhusAI/office-agent](https://github.com/AarhusAI/office-agent), [AarhusAI/eu-funding-tenders-portal-agent](https://github.com/AarhusAI/eu-funding-tenders-portal-agent)
