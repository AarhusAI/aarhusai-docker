# `guardrails/` — LiteLLM custom guardrails

Pre-call guardrails the [LiteLLM](https://github.com/BerriAI/litellm) proxy applies to inbound chat requests before forwarding them to the upstream model. The directory is bind-mounted at `/app/custom_guardrails/` inside the `litellm` service (see [`docker-compose.extra.yml`](../docker-compose.extra.yml)) and registered through the proxy's `guardrails:` config block.

Currently ships one guardrail:

| Guardrail | File | What it does |
|---|---|---|
| `MessageTrimmingGuardrail` | [`message_overflow.py`](message_overflow.py) | Trims oversized message histories to fit the target model's context window, then sanitizes tool-call/tool-response pairings so the trimmed (or otherwise broken) history doesn't crash strict chat templates. |

---

## Quick start

### Enable in `litellm_config.yaml`

```yaml
model_list:
  - model_name: my-model
    litellm_params:
      model: openai/some-deployed-model
      api_base: https://...
      api_key: ""
    guardrails:
      - message_trimming      # attach the guardrail to this model

guardrails:
  - guardrail_name: message_trimming
    litellm_params:
      guardrail: message_overflow.MessageTrimmingGuardrail
      mode: pre_call
      default_on: true
    default_config:
      trim_ratio: 0.75
      max_output_tokens: 2000
      safety_buffer: 500
      debug: true
      default_max_context_tokens: 8192
      max_context_tokens_by_model:
        openai/some-deployed-model: 32768
      pop_trailing_tool_messages: false
```

A canonical example lives at [`litellm_config.example.yaml`](../litellm_config.example.yaml).

### Run tests

```shell
task guardrails:test
```

This spins up a one-shot `litellm` container with `pytest` installed and runs everything under `guardrails/` (see the `guardrails:test` target in [`Taskfile.yml`](../Taskfile.yml)). Edits to `.py` files are picked up on the next run — no rebuild needed.

### Pick up code edits in a running litellm

```shell
task compose -- restart litellm
```

The directory is bind-mounted, so a restart is enough; rebuilding the image is not required.

---

## How `MessageTrimmingGuardrail` works

`async_pre_call_hook` runs on every chat completion request. The flow:

1. **Resolve context window** for the target model (`_resolve_max_context_tokens`):
   per-model override map → `litellm.get_max_tokens` → global default. Logs a warning if it falls through to the global default.
2. **Compute a safe completion budget** (`_calculate_safe_completion_tokens`) — leaves room for input + safety buffer + a 25% headroom factor for tokens LiteLLM/the provider may add later.
3. **Update `max_tokens` / `max_completion_tokens`** in the request so the model can't be asked for more than fits.
4. **Trim input messages** (`litellm.trim_messages`) if `current_input_tokens > max_input_tokens`, dropping older messages from the head until it fits.
5. **Sanitize** (`_sanitize_messages`):
   - `_repair_tool_call_pairings` — strip orphan `role: tool` messages and orphan `tool_calls` entries that the trimmer may have created.
   - (Optional, opt-in via `pop_trailing_tool_messages`) pop trailing `role: tool` messages and re-run the repair, then append `"Please continue"` if the new terminus is an assistant message.
6. **Recount and re-budget** completion tokens once more, since sanitize may have grown or shrunk the message list.

### Why `_repair_tool_call_pairings` exists

LiteLLM's `trim_messages` has **no tool-call awareness** — it drops messages by token count from the head and freely produces:
- Orphan `role: tool` messages (no surviving `assistant.tool_calls` advertised them).
- Orphan `tool_calls` entries on assistant messages (no surviving `role: tool` answered them).

Both shapes are rejected by strict chat templates (Mistral, vLLM, OpenAI strict mode). The repair pass enforces the invariant: every surviving `tool_calls[].id` has a later matching `role: tool` message, and every surviving `role: tool` was advertised by an earlier surviving `assistant.tool_calls` entry. See `_repair_tool_call_pairings` in [`message_overflow.py`](message_overflow.py).

### Why the trailing-tool pop is opt-in

The "normal" agent-loop shape ends on a `role: tool` message:

```
User → Assistant{tool_calls} → Tool{result} → ← model is asked to continue here
```

Most providers (OpenAI, Anthropic, Google, Mistral via the official APIs) **accept** this shape — that's how tool calling works. Popping the tool message and substituting `"Please continue"` deprives the model of the result it was supposed to reason from, so the default is **off**.

Set `pop_trailing_tool_messages: true` only for upstream chat templates that explicitly reject `role: tool` messages — notably the strict HuggingFace `Mistral-7B-Instruct-v0.3` template that raises `"Only user and assistant roles are supported!"`. The per-model override map lets you flip it for one model in a fleet without affecting the others.

### Why both repairs run when pop is enabled

The order is `repair → pop → repair → maybe-append-continue`:

- The first repair cleans up orphans created by `trim_messages`.
- The pop may break a previously-valid `[Assistant{tool_calls=[X]}, Tool X]` pair, leaving the assistant holding orphan `tool_calls`.
- The second repair restores the invariant — strips the now-orphan `tool_calls`, drops content-empty assistants entirely.
- *Then* we decide whether to append `"Please continue"`, after seeing the post-repair terminus. (Appending before would risk leaving a stale "user-continue" line after a now-deleted assistant.)

---

## Configuration reference

Read from `default_config` of the guardrail entry in `litellm_config.yaml`. All keys optional.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `trim_ratio` | float | `0.75` | Forwarded to `litellm.trim_messages`. Fraction of `max_tokens` that trimming aims for, leaving headroom for additions later in the pipeline. |
| `max_output_tokens` | int | `2000` | Default completion budget when the request specifies neither `max_tokens` nor `max_completion_tokens`. |
| `safety_buffer` | int | `500` | Reserved tokens carved out of the context window before computing input/output budgets — covers system prompts, function schemas, and other tokens added downstream. |
| `debug` | bool | `false` | When `true`, the guardrail prints `[GUARDRAIL]`-prefixed traces to stdout. Show up in `task compose -- logs -f litellm`. |
| `default_max_context_tokens` | int | `8192` | Fallback context-window size when neither `max_context_tokens_by_model` nor `litellm.get_max_tokens` resolves the model. **Bump this if your fleet's smallest model is bigger than 8k.** |
| `max_context_tokens_by_model` | dict | `{}` | Per-model overrides keyed by the upstream `model:` value LiteLLM forwards (NOT the friendly `model_name`). Wins over `litellm.get_max_tokens`. Use this for vLLM, Bedrock variants, custom deployments — anything not in [`litellm/model_prices_and_context_window.json`](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). |
| `pop_trailing_tool_messages` | bool | `false` | Strip trailing `role: tool` messages before forwarding. **Leave `false` unless the upstream chat template rejects them** — popping loses tool-call results the model needs to reason from. |
| `pop_trailing_tool_messages_by_model` | dict | `{}` | Per-model override of the flag above, same key shape as `max_context_tokens_by_model`. |

### Resolution order, illustrated

```
context window:
  max_context_tokens_by_model[model]   ─┐
  ↓ (miss)                              │
  litellm.get_max_tokens(model)         ├─ first hit wins
  ↓ (raises / 0)                        │
  default_max_context_tokens           ─┘

pop trailing tools:
  pop_trailing_tool_messages_by_model[model]  ─┐
  ↓ (miss)                                     ├─ first hit wins
  pop_trailing_tool_messages                  ─┘
```

---

## File map

| File | Purpose |
|---|---|
| [`message_overflow.py`](message_overflow.py) | The guardrail implementation. |
| [`test_message_overflow.py`](test_message_overflow.py) | Unit tests, run via `task guardrails:test`. |
| [`pytest.ini`](pytest.ini) | Pinning pytest's rootdir to this directory so it doesn't pick up `/app/pyproject.toml` inside the litellm container (which expects `pytest-asyncio`, a dep we deliberately skip — tests use `asyncio.run()` directly). |

---

## Adding a new guardrail

1. Drop a new `<name>.py` in this directory exporting a class that subclasses `litellm.integrations.custom_guardrail.CustomGuardrail`. Override `async_pre_call_hook` (or `async_post_call_*` for output-side hooks).
2. Read configuration from `default_config` via the same `_load_config` / `_get_default_config` pattern as `MessageTrimmingGuardrail`, so it can be tweaked from `litellm_config.yaml` without code changes.
3. Add it to `guardrails:` in `litellm_config.yaml` and attach it to the relevant models via their `guardrails:` list.
4. Drop a `test_<name>.py` next to it. The shared test runner (`task guardrails:test`) will pick it up automatically.
5. Update this README's guardrail table at the top.

---

## References

- [LiteLLM custom guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail)
