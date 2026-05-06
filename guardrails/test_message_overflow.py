import asyncio

import pytest

import message_overflow
from message_overflow import MessageTrimmingGuardrail


def _run(coro):
    """Tiny helper so we don't need pytest-asyncio for `async_pre_call_hook` tests."""
    return asyncio.run(coro)


def _mock_litellm(
    monkeypatch,
    *,
    max_tokens=32768,
    token_count=None,
    trim_returns=None,
):
    """Pin the three litellm helpers the guardrail imports so integration tests
    don't depend on the live model registry or the network.

    - `max_tokens` -> what `get_max_tokens` returns.
    - `token_count` -> if int, what `token_counter` returns; if `None`, falls
      back to a deterministic word-count estimate.
    - `trim_returns` -> if a list, `trim_messages` returns that list verbatim;
      if `None`, it's a passthrough (returns input unchanged).

    Returns a dict accumulating call records: keys `"trim_calls"` and
    `"counter_calls"`.
    """
    calls = {"trim_calls": [], "counter_calls": []}

    monkeypatch.setattr(message_overflow, "get_max_tokens", lambda m: max_tokens)

    if isinstance(token_count, int):
        def _counter(model, messages):
            calls["counter_calls"].append(len(messages))
            return token_count
    else:
        def _counter(model, messages):
            calls["counter_calls"].append(len(messages))
            return sum(
                len(str(m.get("content", "")).split()) for m in messages
            )
    monkeypatch.setattr(message_overflow, "token_counter", _counter)

    def _trim(messages, model, max_tokens, trim_ratio):
        calls["trim_calls"].append(
            {"max_tokens": max_tokens, "trim_ratio": trim_ratio, "n_in": len(messages)}
        )
        return list(trim_returns) if trim_returns is not None else messages

    monkeypatch.setattr(message_overflow, "trim_messages", _trim)
    return calls


def _make_guardrail(monkeypatch, default_config=None):
    monkeypatch.setattr(MessageTrimmingGuardrail, "_load_config", lambda self: {})
    monkeypatch.setattr(
        MessageTrimmingGuardrail,
        "_get_default_config",
        lambda self, name: default_config or {},
    )
    return MessageTrimmingGuardrail(guardrail_name="test")


@pytest.fixture
def guardrail(monkeypatch):
    """Default-config guardrail: pop_trailing_tool_messages=False."""
    return _make_guardrail(monkeypatch)


@pytest.fixture
def guardrail_with_pop(monkeypatch):
    """Guardrail with pop_trailing_tool_messages=True (legacy/strict-template behaviour)."""
    return _make_guardrail(monkeypatch, {"pop_trailing_tool_messages": True})


# --- _ensure_last_is_user ---


def test_assistant_last_appends_user_continue(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = guardrail._ensure_last_is_user(messages)
    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "Please continue"},
    ]


def test_user_last_unchanged(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
    ]
    assert guardrail._ensure_last_is_user(messages) == messages


def test_trailing_tool_popped_then_user_appended_if_assistant_exposed(guardrail_with_pop):
    """When pop_trailing_tool_messages=True, the trailing tool is popped so the
    new terminus is the assistant; we then append a user-continue. (Note:
    `_ensure_last_is_user` alone leaves an orphan tool_call here — the
    `_sanitize_messages` chain re-runs repair to clean it up; see
    `test_sanitize_does_not_orphan_assistant_tool_calls_when_popping_terminal_tool`.)"""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "r"},
    ]
    result = guardrail_with_pop._ensure_last_is_user(messages, pop_trailing_tools=True)
    assert result[-1] == {"role": "user", "content": "Please continue"}
    assert result[-2]["role"] == "assistant"
    assert len(result) == 3  # tool popped, user appended


def test_empty_list_returns_empty(guardrail):
    assert guardrail._ensure_last_is_user([]) == []


def test_system_only_unchanged(guardrail):
    messages = [{"role": "system", "content": "you are helpful"}]
    assert guardrail._ensure_last_is_user(messages) == messages


# --- _repair_tool_call_pairings ---


def test_orphan_tool_stripped(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "x", "content": "stale"},
    ]
    assert guardrail._repair_tool_call_pairings(messages) == [
        {"role": "user", "content": "hi"}
    ]


def test_orphan_assistant_tool_calls_stripped_content_preserved(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "tool_calls": [{"id": "x", "type": "function"}],
        },
    ]
    result = guardrail._repair_tool_call_pairings(messages)
    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_orphan_assistant_no_content_dropped(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "x", "type": "function"}],
        },
    ]
    assert guardrail._repair_tool_call_pairings(messages) == [
        {"role": "user", "content": "hi"}
    ]


def test_paired_tool_calls_preserved(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "x", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "x", "content": "result"},
    ]
    assert guardrail._repair_tool_call_pairings(messages) == messages


def test_partial_tool_calls_filtered(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "x", "type": "function"},
                {"id": "y", "type": "function"},
            ],
        },
        {"role": "tool", "tool_call_id": "x", "content": "result"},
    ]
    result = guardrail._repair_tool_call_pairings(messages)
    assert result == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "x", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "x", "content": "result"},
    ]


def test_tool_without_tool_call_id_dropped(guardrail):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "oops"},
    ]
    assert guardrail._repair_tool_call_pairings(messages) == [
        {"role": "user", "content": "hi"}
    ]


def test_system_messages_preserved_through_repair(guardrail):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert guardrail._repair_tool_call_pairings(messages) == messages


def test_tool_after_dropped_assistant_also_dropped(guardrail):
    # Assistant advertises tool_call "y" but there is no tool response for "y",
    # so the assistant (content-empty) is dropped. A separate tool message with
    # id "x" IS satisfied by its own presence (satisfied_ids={"x"}), but since
    # no surviving assistant ever advertised "x", advertised_ids stays empty
    # and the "x" tool is dropped too. This tests the advertised_ids vs
    # satisfied_ids distinction.
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "y", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "x", "content": "unbound"},
    ]
    result = guardrail._repair_tool_call_pairings(messages)
    assert result == [{"role": "user", "content": "hi"}]


# --- _sanitize_messages (composition) ---


def test_sanitize_empty_returns_empty(guardrail):
    assert guardrail._sanitize_messages([]) == []


def test_sanitize_repairs_orphan_tool_and_fixes_assistant_last(guardrail):
    messages = [
        {"role": "tool", "tool_call_id": "stale", "content": "stale"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert guardrail._sanitize_messages(messages) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "Please continue"},
    ]


def test_sanitize_system_only(guardrail):
    messages = [{"role": "system", "content": "sys"}]
    assert guardrail._sanitize_messages(messages) == messages


# --- BUG 1: orphan tool_calls after popping trailing tool message ---


def _assert_no_orphan_tool_calls(messages):
    """Every assistant.tool_calls entry must have a matching later role: tool."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc_ids = {tc["id"] for tc in m["tool_calls"]}
            later_ids = {
                x.get("tool_call_id")
                for x in messages[i + 1 :]
                if x.get("role") == "tool"
            }
            orphans = tc_ids - later_ids
            assert not orphans, f"orphan tool_calls left after sanitize: {orphans}"


def test_sanitize_does_not_orphan_assistant_tool_calls_when_popping_terminal_tool(
    guardrail_with_pop,
):
    """With pop enabled, sanitize must re-repair so that popping the terminal
    tool message does not leave the assistant holding orphan tool_calls."""
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
    ]
    result = guardrail_with_pop._sanitize_messages(messages)
    _assert_no_orphan_tool_calls(result)
    # Empty-content assistant whose only tool_call became orphan should be dropped.
    assert all(
        not (m.get("role") == "assistant" and m.get("tool_calls")) for m in result
    )


def test_sanitize_does_not_orphan_assistant_tool_calls_with_multi_tool_terminus(
    guardrail_with_pop,
):
    """Two parallel tool_calls, two tool responses, pop both — the assistant's
    advertisement of *both* IDs must not survive as orphans."""
    messages = [
        {"role": "user", "content": "fetch two things"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function"},
                {"id": "t2", "type": "function"},
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "a"},
        {"role": "tool", "tool_call_id": "t2", "content": "b"},
    ]
    result = guardrail_with_pop._sanitize_messages(messages)
    _assert_no_orphan_tool_calls(result)


def test_sanitize_terminal_assistant_with_content_strips_orphan_tool_calls(
    guardrail_with_pop,
):
    """Content-bearing assistant: content survives, orphan tool_calls get stripped."""
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "Looking it up...",
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    result = guardrail_with_pop._sanitize_messages(messages)
    _assert_no_orphan_tool_calls(result)
    assistants = [m for m in result if m.get("role") == "assistant"]
    assert assistants, "content-bearing assistant must survive"
    assert assistants[-1].get("content") == "Looking it up..."
    assert "tool_calls" not in assistants[-1] or not assistants[-1].get("tool_calls")


def test_sanitize_terminal_assistant_no_content_dropped_after_pop(guardrail_with_pop):
    """Empty-content assistant whose tool_calls all become orphan after pop is
    dropped, leaving the prior user as terminus (no spurious 'Please continue')."""
    messages = [
        {"role": "user", "content": "tell me"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    result = guardrail_with_pop._sanitize_messages(messages)
    _assert_no_orphan_tool_calls(result)
    assert result == [{"role": "user", "content": "tell me"}]


# --- BUG 5: default behaviour preserves trailing tool messages ---


def test_sanitize_default_keeps_trailing_tool_message(guardrail):
    """Default config (pop_trailing_tool_messages=False) must preserve the
    tool message at the terminus — that's where the model is expected to
    reason from in a normal agent loop."""
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
    ]
    result = guardrail._sanitize_messages(messages)
    assert result[-1]["role"] == "tool"
    assert result[-1]["tool_call_id"] == "t1"


def test_sanitize_default_does_not_append_user_continue_on_tool_terminus(guardrail):
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
    ]
    result = guardrail._sanitize_messages(messages)
    assert not any(
        m.get("role") == "user" and m.get("content") == "Please continue"
        for m in result
    )


# --- BUG 2: max-context-tokens resolution ---


def test_resolve_max_context_tokens_uses_per_model_map_first(monkeypatch):
    # Even if litellm would return something else, the per-model map wins.
    monkeypatch.setattr(message_overflow, "get_max_tokens", lambda m: 4096)
    g = _make_guardrail(
        monkeypatch,
        {
            "default_max_context_tokens": 8192,
            "max_context_tokens_by_model": {"my-model": 128000},
        },
    )
    assert g._resolve_max_context_tokens("my-model") == 128000


def test_resolve_max_context_tokens_falls_back_to_litellm(monkeypatch):
    """No per-model override -> defer to litellm.get_max_tokens (not the global default)."""
    monkeypatch.setattr(message_overflow, "get_max_tokens", lambda m: 16384)
    g = _make_guardrail(monkeypatch, {"default_max_context_tokens": 4096})
    assert g._resolve_max_context_tokens("some-model") == 16384


def test_resolve_max_context_tokens_falls_back_to_default_when_litellm_raises(
    monkeypatch,
):
    """litellm raises (model not in price map) -> use default_max_context_tokens."""
    def _boom(model):
        raise Exception("Model isn't mapped yet")

    monkeypatch.setattr(message_overflow, "get_max_tokens", _boom)
    g = _make_guardrail(monkeypatch, {"default_max_context_tokens": 32768})
    assert g._resolve_max_context_tokens("openai/totally-not-a-real-model-xyz") == 32768


# --- _resolve_pop_trailing_tools ---


def test_resolve_pop_trailing_tools_per_model_overrides_global(monkeypatch):
    g = _make_guardrail(
        monkeypatch,
        {
            "pop_trailing_tool_messages": False,
            "pop_trailing_tool_messages_by_model": {"strict-model": True},
        },
    )
    assert g._resolve_pop_trailing_tools("strict-model") is True
    assert g._resolve_pop_trailing_tools("other-model") is False


def test_resolve_pop_trailing_tools_falls_back_to_global(monkeypatch):
    g = _make_guardrail(monkeypatch, {"pop_trailing_tool_messages": True})
    assert g._resolve_pop_trailing_tools("any-model") is True


def test_resolve_pop_trailing_tools_none_model_uses_global(monkeypatch):
    g = _make_guardrail(
        monkeypatch,
        {
            "pop_trailing_tool_messages": True,
            "pop_trailing_tool_messages_by_model": {"strict-model": False},
        },
    )
    # `None` model can't be in the map, so falls through to the global flag.
    assert g._resolve_pop_trailing_tools(None) is True


# --- _calculate_safe_completion_tokens ---


def test_calculate_safe_completion_happy_path(guardrail):
    """Ample budget: returns the requested completion."""
    # available = 32768 - 1000 - 500 = 31268
    # max(256, int(31268 * 0.75)) = 23451 -> min(2000, 23451) = 2000
    assert guardrail._calculate_safe_completion_tokens(32768, 1000, 2000) == 2000


def test_calculate_safe_completion_tight_budget_returns_available_share(guardrail):
    """Cramped budget: returns the 75% share of remaining headroom."""
    # available = 8192 - 7000 - 500 = 692
    # max(256, int(692 * 0.75)) = 519 -> min(2000, 519) = 519
    assert guardrail._calculate_safe_completion_tokens(8192, 7000, 2000) == 519


def test_calculate_safe_completion_overflow_clamps_to_floor(guardrail):
    """Input exceeds context entirely: clamps to the 256 floor."""
    # available negative; max(256, negative) = 256; min(2000, 256) = 256
    assert guardrail._calculate_safe_completion_tokens(4096, 5000, 2000) == 256


def test_calculate_safe_completion_requested_below_floor_returns_requested(guardrail):
    """If the caller requested less than 256, honor that — don't inflate."""
    # min(50, max(256, 24126)) = min(50, 24126) = 50
    assert guardrail._calculate_safe_completion_tokens(32768, 100, 50) == 50


# --- _update_completion_tokens ---


def test_update_completion_tokens_max_tokens_branch(guardrail):
    data = {"max_tokens": 9999}
    guardrail._update_completion_tokens(
        data, 1000, has_max_tokens=True, has_max_completion=False
    )
    assert data["max_tokens"] == 1000
    assert "max_completion_tokens" not in data


def test_update_completion_tokens_max_completion_branch(guardrail):
    data = {"max_completion_tokens": 9999}
    guardrail._update_completion_tokens(
        data, 1000, has_max_tokens=False, has_max_completion=True
    )
    assert data["max_completion_tokens"] == 1000
    assert "max_tokens" not in data


def test_update_completion_tokens_neither_set_adds_max_tokens(guardrail):
    data = {}
    guardrail._update_completion_tokens(
        data, 1000, has_max_tokens=False, has_max_completion=False
    )
    assert data == {"max_tokens": 1000}


# --- idempotency ---


def test_sanitize_is_idempotent_default(guardrail):
    """Default config: running sanitize twice yields the same shape."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    once = guardrail._sanitize_messages(messages)
    twice = guardrail._sanitize_messages(once)
    assert once == twice


def test_sanitize_is_idempotent_with_pop(guardrail_with_pop):
    """Pop config: same idempotency requirement."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Looking up...",
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    once = guardrail_with_pop._sanitize_messages(messages)
    twice = guardrail_with_pop._sanitize_messages(once)
    assert once == twice


# --- default-off multi-tool terminus (mirror of pop=True multi-tool case) ---


def test_sanitize_default_keeps_multi_tool_terminus(guardrail):
    """Default (pop=False): both tool messages of a parallel tool_call must
    survive — we want the model to see all results."""
    messages = [
        {"role": "user", "content": "fetch two things"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function"},
                {"id": "t2", "type": "function"},
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "a"},
        {"role": "tool", "tool_call_id": "t2", "content": "b"},
    ]
    result = guardrail._sanitize_messages(messages)
    tool_ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
    assert tool_ids == ["t1", "t2"]
    # Assistant's tool_calls advertisement preserved intact.
    assistants = [m for m in result if m.get("role") == "assistant"]
    assert assistants[-1]["tool_calls"] == [
        {"id": "t1", "type": "function"},
        {"id": "t2", "type": "function"},
    ]


# --- multi-turn agent loop ---


def test_sanitize_multi_turn_agent_loop_default_preserves_all(guardrail):
    """Default config: a fully-paired multi-turn agent loop passes through
    untouched (no orphans, no popping, ends on tool which is the natural
    place for the model to be asked to continue)."""
    messages = [
        {"role": "user", "content": "Q1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "r1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t2", "content": "r2"},
    ]
    result = guardrail._sanitize_messages(messages)
    assert result == messages


def test_sanitize_multi_turn_agent_loop_with_pop_drops_last_tool_round(
    guardrail_with_pop,
):
    """Pop config: the last tool round (`Asst{tcs=[t2]}` + `Tool t2`) must
    collapse together — popping `Tool t2` would leave the assistant holding
    orphan `tool_calls=[t2]`, so the second repair drops the assistant.
    The earlier `t1` round and the content-bearing `Asst A1` survive."""
    messages = [
        {"role": "user", "content": "Q1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "r1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "t2", "content": "r2"},
    ]
    result = guardrail_with_pop._sanitize_messages(messages)
    _assert_no_orphan_tool_calls(result)
    # Earlier round preserved
    assert any(m.get("tool_call_id") == "t1" for m in result)
    # Last round (both Asst{tcs=[t2]} and Tool t2) gone
    assert not any(m.get("tool_call_id") == "t2" for m in result)
    assert not any(
        m.get("role") == "assistant"
        and any(tc.get("id") == "t2" for tc in (m.get("tool_calls") or []))
        for m in result
    )
    # Terminus is the surviving Q2 user message — no spurious "Please continue"
    assert result[-1] == {"role": "user", "content": "Q2"}


# --- async_pre_call_hook integration ---


def test_async_pre_call_hook_no_messages_key_returns_unchanged(guardrail, monkeypatch):
    _mock_litellm(monkeypatch)
    data = {"model": "m"}
    result = _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert result == {"model": "m"}


def test_async_pre_call_hook_empty_messages_returns_unchanged(guardrail, monkeypatch):
    _mock_litellm(monkeypatch)
    data = {"model": "m", "messages": []}
    result = _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert result == {"model": "m", "messages": []}


def test_async_pre_call_hook_small_conversation_does_not_trim(guardrail, monkeypatch):
    """token_counter returns small number → trim_messages must not be invoked."""
    calls = _mock_litellm(monkeypatch, max_tokens=32768, token_count=100)
    data = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "ping"},
        ],
    }
    result = _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert calls["trim_calls"] == []
    # max_tokens added because the request had no completion limit originally
    assert "max_tokens" in result
    # message list shape preserved (last is user; sanitize is a no-op here)
    assert len(result["messages"]) == 3


def test_async_pre_call_hook_oversized_conversation_invokes_trim(
    guardrail, monkeypatch
):
    """current > max_input → trim_messages called with the configured trim_ratio."""
    calls = _mock_litellm(monkeypatch, max_tokens=8192, token_count=7000)
    data = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ],
    }
    _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert len(calls["trim_calls"]) == 1
    # Default trim_ratio from `_make_guardrail` config is the guardrail default 0.75.
    assert calls["trim_calls"][0]["trim_ratio"] == 0.75


def test_async_pre_call_hook_existing_max_tokens_is_clamped(guardrail, monkeypatch):
    """Caller-provided `max_tokens` is updated (clamped down to safe budget),
    `max_completion_tokens` is NOT introduced."""
    _mock_litellm(monkeypatch, max_tokens=32768, token_count=100)
    data = {
        "model": "m",
        "max_tokens": 999_999,
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert result["max_tokens"] < 999_999
    assert "max_completion_tokens" not in result


def test_async_pre_call_hook_existing_max_completion_tokens_is_clamped(
    guardrail, monkeypatch
):
    """Symmetric: `max_completion_tokens` updated, `max_tokens` not introduced."""
    _mock_litellm(monkeypatch, max_tokens=32768, token_count=100)
    data = {
        "model": "m",
        "max_completion_tokens": 999_999,
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert result["max_completion_tokens"] < 999_999
    assert "max_tokens" not in result


def test_async_pre_call_hook_no_completion_limit_adds_max_tokens(guardrail, monkeypatch):
    """No completion limit set → `max_tokens` is added (so the model isn't
    asked for an unbounded completion that could overflow context)."""
    _mock_litellm(monkeypatch, max_tokens=32768, token_count=100)
    data = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert "max_tokens" in result
    assert "max_completion_tokens" not in result


def test_async_pre_call_hook_token_counter_raises_uses_estimate(guardrail, monkeypatch):
    """Counter throwing must not crash the hook; the word-count fallback runs."""
    monkeypatch.setattr(message_overflow, "get_max_tokens", lambda m: 32768)

    def _boom(model, messages):
        raise Exception("counter failed")

    monkeypatch.setattr(message_overflow, "token_counter", _boom)
    monkeypatch.setattr(
        message_overflow,
        "trim_messages",
        lambda messages, model, max_tokens, trim_ratio: messages,
    )
    data = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi there how are you"}],
    }
    # Must not raise.
    result = _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert "messages" in result


def test_async_pre_call_hook_recounts_tokens_after_sanitize(guardrail, monkeypatch):
    """token_counter is invoked at least twice: once before trim, once after
    sanitize — so the post-sanitize re-budget step actually runs."""
    calls = _mock_litellm(monkeypatch, max_tokens=32768, token_count=100)
    data = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    _run(guardrail.async_pre_call_hook(None, None, data, "completion"))
    assert len(calls["counter_calls"]) >= 2
