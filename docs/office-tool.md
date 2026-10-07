# Office tool

[AarhusAI/office-tool](https://github.com/AarhusAI/office-tool) generates and inspects docx/xlsx/pptx files from chat.
It runs from `docker-compose.office.yml`:

- `office-sandbox` — build-only; produces the sandbox image on the host daemon, then exits.
- `office-mcp` — MCP server, registered in Open WebUI as tool id `office` at `http://office-mcp:8000/mcp`.
- `office-sandbox-runner` — mounts `/var/run/docker.sock` and starts one sandbox container per chat on the host daemon.
- `soffice-svc` — LibreOffice/poppler conversion.

## Setup

Clone the agents (office-tool lands in `agents/office-tool`, branch `develop`):

```bash
task agents:clone
```

Set in `.env` (see the "Office tool" block in `.env.default`):

| Variable                           | Purpose                                                                                                                  |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `OFFICE_SERVICE_API_KEY`           | Bearer key Open WebUI uses against `office-mcp`.                                                                         |
| `OFFICE_TOOLS=true`                | Enables the tool in Open WebUI (default `false`).                                                                        |
| `ENABLE_FORWARD_USER_INFO_HEADERS=true` | Needed by `import_file`. Caveat: every tool server in the stack then receives `X-OpenWebUI-User-*` headers.          |
| `OFFICE_OPENWEBUI_API_KEY`         | Admin API key `office-mcp` uses to fetch user uploads from Open WebUI.                                                   |
| `OFFICE_TEMPLATES_DIR`             | Absolute **host** path to house templates, mounted read-only at `/templates`. Empty disables templates.                  |
| `OFFICE_SANDBOX_IMAGE` (optional)  | Sandbox image; default `office-sandbox:dev`.                                                                             |
| `DOCKER_GID`                       | Group id of the docker socket, added to `office-sandbox-runner` (default `999`).                                         |

`docker-compose.office.yml` reads `DOCKER_GID` (`group_add: ${DOCKER_GID:-999}`), not `OFFICE_DOCKER_GID` as named in
`.env.default`. On Linux:

```bash
echo "DOCKER_GID=$(getent group docker | cut -d: -f3)" >> .env
```

Start — the agents overlay does not include the office file, so pass it explicitly:

```bash
task compose -- -f docker-compose.agents.yml -f docker-compose.office.yml up --detach
```

Check readiness:

```bash
task compose -- -f docker-compose.agents.yml -f docker-compose.office.yml exec office-sandbox-runner curl -s localhost:8000/health/ready
```

`{"status":"ok"}` means ready. On HTTP 503 the `detail` field says why:

| `detail`        | Cause                                                          |
|-----------------|----------------------------------------------------------------|
| `docker socket` | Runner cannot reach `/var/run/docker.sock` — check `DOCKER_GID`. |
| `runtime`       | `SANDBOX_RUNTIME` is not registered in `docker info`.          |
| `image`         | Sandbox image missing on the host daemon.                      |

## Runtime / gVisor

`SANDBOX_RUNTIME` controls both the sandbox containers (`--runtime`) and `soffice-svc` (`runtime:`). This repo's compose
defaults it to `runc`; upstream office-tool treats gVisor (`runsc`) as a hard requirement.

On Linux hosts with gVisor installed, set `SANDBOX_RUNTIME=runsc` in `.env` and verify the runtime is registered:

```bash
docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}'
```

## macOS (no runsc)

Docker Desktop cannot run gVisor. To run the tool anyway:

- Set `SANDBOX_RUNTIME` to `SANDBOX_RUNTIME=runc`.
- If `/health/ready` reports `docker socket`: Docker Desktop exposes the socket inside containers as `root` (gid 0), so
  set `DOCKER_GID=0`.
- `OFFICE_TEMPLATES_DIR` must be a path shared with Docker Desktop (under `/Users/...`), since the runner passes it to
  the host daemon.
- Apple Silicon: images build natively (arm64) from source; nothing extra needed.

> [!WARNING]
> `runc` gives no kernel isolation for model-generated Python and untrusted documents. Use it for local development
> only, never in production.
