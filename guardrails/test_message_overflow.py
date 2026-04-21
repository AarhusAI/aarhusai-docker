import pytest

from message_overflow import MessageTrimmingGuardrail


@pytest.fixture
def guardrail(monkeypatch):
    monkeypatch.setattr(
        MessageTrimmingGuardrail, "_load_config", lambda self: {}
    )
    return MessageTrimmingGuardrail(guardrail_name="test")


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


def test_trailing_tool_popped_then_user_appended_if_assistant_exposed(guardrail):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "r"},
    ]
    result = guardrail._ensure_last_is_user(messages)
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
