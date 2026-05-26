# Docker setup for open WebUI

https://docs.openwebui.com/

## Install

Use the task file to install it.

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

Installs open-webui from this fork of [open-webui](https://github.com/itk-dev/open-webui).

### Storage

Log into minio and create the bucket "openwebui" before uploading files.

## Usage

```shell
task open
```

## OIDC

An (incomplete) OIDC dev setup with a mock server is included. Currently, this fails with

```text
Unhandled exception. System.IO.FileLoadException: Could not load file or assembly 'OpenIdConnectServerMock, Version=0.10.1.0, Culture=neutral, PublicKeyToken=null'.
```

## Configuring connections and models and stuff

The `litellm_config.yaml` file should have the proper api keys. If these were not set from the get-go, you should
restart the container with the new values.

Go to `/admin/settings/connections` and configure the connections, OpenAI API should have the Api Base Url
`http://litellm:4000/v1` and api key `sk-1234` (see `docker.compose.yml`). Then the models will appear in the models
tab: `/admin/settings/models`.

## Patching

This project requires us to make changes that the upstream project may not appreciate. We need to be able to keep
this project in sync with the upstream and at the same time track local changes/patches to the codebase.

We use branches and local pull request against our own fork to track changes and go-task to manage the patches. The pull requests can be
found [in this repository](https://github.com/itk-dev/open-webui/pulls). Notice the use of labels to mark pull requests as patches,
mark when pull requests are approved and to mark which pull requests are pending upstream.

### Update pull requests

When a new upstream release is published, we need to update the patches to the new release.

First we need to update `dev` and `main` branches with upstream. This is easily done on GitHub. But as GitHub does not
synchronize the tags, the following task can be used to do that:

```shell
task git:sync:tags
```

Then we can update the patches. This requires some manual steps to ensure that the patches can be applied cleanly.
First create a new branch based on the upstream tag, named `upstream/<release tag>`, which then is used as the new base
branch for the patches.

```shell
task git:checkout:dev
git checkout -b upstream/<release tag>
git push origin -u upstream/<release tag>
```

Then go to [https://github.com/itk-dev/open-webui/pulls](https://github.com/itk-dev/open-webui/pulls) and changes each
PR's base branch to the new branch.

Lastly, checkout each branch locally and rebase it on the new base branch.

```shell
git checkout feature/<PR branch>
git rebase --onto upstream/<new tag (e.g. v0.6.27)> upstream/<old tag (e.g. v0.6.26)> 
```

Resolve any conflicts (if any, using `add` and `rebase --continue`).

```shell
git add .
git rebase --continue
```

Push the branch to the remote.
```shell
git push origin feature/<PR branch>
```

### Branches

* `upstream/<release tag>`

### Rules

So the following rules should help with this goal.

1. Prefix all commits with a ticket number, this is to be able to remove a single change or isolate a change for easier
   merge conflict handling.
2. Always first try to make at pull request back
   to [https://github.com/open-webui/open-webui/](https://github.com/open-webui/open-webui/), if the request is
   declined, make a new pull request to the patche branch above based on what you change are.
3. Wrap patches in a comment clearly stating it is a patch, and some text on what or why.

## Production

To build image for production first edit the `Taskfile.yaml` and update the `PROD_OPEN_WEBUI_VERSION` variable to the
desired version. The run:

```shell
task prod:build
```
