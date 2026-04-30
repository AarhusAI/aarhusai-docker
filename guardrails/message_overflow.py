from typing import Literal, Optional, Union
from litellm.utils import trim_messages, get_max_tokens, token_counter
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
import logging
import yaml

logger = logging.getLogger(__name__)


class MessageTrimmingGuardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

        default_config = self._get_default_config(
            self.optional_params.get("guardrail_name")
        )

        # Default Configuration parameters
        self.trim_ratio = default_config.get("trim_ratio", 0.75)
        self.max_output_tokens = default_config.get("max_output_tokens", 2000)
        self.safety_buffer = default_config.get("safety_buffer", 500)
        self.debug = default_config.get("debug", False)

    def _load_config(self):
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config

    def _get_default_config(self, guardrail_name: str):
        """Extract default_config for this guardrail from the full config."""
        try:
            config = self._load_config()
            guardrails = config.get("guardrails", [])
            for guardrail in guardrails:
                if guardrail.get("guardrail_name") == guardrail_name:
                    return guardrail.get("default_config", {})
            return {}
        except Exception as e:
            logger.warning(f"Failed to extract default_config: {e}")
            return {}

    def _log_debug(self, message: str):
        """Log debug messages if debug mode is enabled."""
        if self.debug:
            print(f"[GUARDRAIL] {message}")

    def _calculate_safe_completion_tokens(
        self,
        max_context_tokens: int,
        current_input_tokens: int,
        requested_completion: int,
    ) -> int:
        """Calculate safe completion tokens based on context and input constraints.

        Args:
            max_context_tokens: Maximum context window size for the model
            current_input_tokens: Current number of input tokens
            requested_completion: Requested number of completion tokens

        Returns:
            Safe number of completion tokens that won't exceed context window
        """
        available_for_completion = (
            max_context_tokens - int(current_input_tokens) - self.safety_buffer
        )
        safe_completion_tokens = min(
            requested_completion, max(256, int(available_for_completion * 0.75))
        )
        return safe_completion_tokens

    def _update_completion_tokens(
        self,
        data: dict,
        safe_completion_tokens: int,
        has_max_tokens: bool,
        has_max_completion: bool,
    ) -> None:
        """Update completion token limits in the request data.

        Args:
            data: The request data dictionary to update
            safe_completion_tokens: The safe number of completion tokens to set
            has_max_tokens: Whether max_tokens was originally in the data
            has_max_completion: Whether max_completion_tokens was originally in the data
        """
        if has_max_tokens:
            self._log_debug(
                f"Updating max_tokens from {data['max_tokens']} to {safe_completion_tokens}"
            )
            data["max_tokens"] = safe_completion_tokens
        elif has_max_completion:
            self._log_debug(
                f"Updating max_completion_tokens from {data['max_completion_tokens']} to {safe_completion_tokens}"
            )
            data["max_completion_tokens"] = safe_completion_tokens
        else:
            # If neither was set, we need to add one to prevent default from being too large
            self._log_debug(
                f"No completion token limit set, adding max_tokens={safe_completion_tokens}"
            )
            data["max_tokens"] = safe_completion_tokens

    def _repair_tool_call_pairings(self, messages: list) -> list:
        """Strip orphan tool messages and orphan tool_calls from assistant messages.

        After `trim_messages` truncates history, tool-call/tool-response pairings
        can end up broken. Mistral/vLLM rejects such conversations. We keep only
        tool_calls that have a matching later `role: tool` response, and keep only
        `role: tool` messages whose `tool_call_id` was advertised by a surviving
        assistant `tool_calls` entry.
        """
        satisfied_ids = {
            m.get("tool_call_id")
            for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }

        result = []
        advertised_ids = set()
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                tcs = msg.get("tool_calls") or []
                if tcs:
                    kept_tcs = [tc for tc in tcs if tc.get("id") in satisfied_ids]
                    content = msg.get("content")
                    if not kept_tcs and not (content or "").strip():
                        self._log_debug(
                            "Dropping assistant message with no content and all tool_calls orphaned"
                        )
                        continue
                    new_msg = dict(msg)
                    if kept_tcs:
                        new_msg["tool_calls"] = kept_tcs
                        advertised_ids.update(tc["id"] for tc in kept_tcs)
                    else:
                        new_msg.pop("tool_calls", None)
                        self._log_debug(
                            "Stripped all orphan tool_calls from assistant message"
                        )
                    result.append(new_msg)
                else:
                    result.append(msg)
            elif role == "tool":
                if msg.get("tool_call_id") in advertised_ids:
                    result.append(msg)
                else:
                    self._log_debug(
                        f"Dropping orphan tool message (tool_call_id={msg.get('tool_call_id')})"
                    )
            else:
                result.append(msg)
        return result

    def _ensure_last_is_user(self, messages: list) -> list:
        """Guarantee the conversation does not end on an assistant message.

        vLLM's HuggingFace chat template rejects assistant-terminal conversations
        when `add_generation_prompt=True`. Pops any trailing `role: tool` messages
        (invalid terminus after trimming), then appends a user continuation
        message if the new terminus is an assistant message.
        """
        if not messages:
            return messages
        result = list(messages)
        while result and result[-1].get("role") == "tool":
            self._log_debug("Popping trailing tool message from terminus")
            result.pop()
        if result and result[-1].get("role") == "assistant":
            self._log_debug("Appending user continue message after assistant terminus")
            result.append({"role": "user", "content": "Please continue"})
        return result

    def _sanitize_messages(self, messages: list) -> list:
        """Repair tool-call pairings and ensure the conversation does not end
        on an assistant message. Safe to call on every request."""
        if not messages:
            return messages
        repaired = self._repair_tool_call_pairings(messages)
        return self._ensure_last_is_user(repaired)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        # Trim messages if present
        if "messages" in data and data["messages"]:
            model = data.get("model")

            # Get model's context window size
            try:
                max_context_tokens = get_max_tokens(model)
            except:
                max_context_tokens = 8192  # Default fallback

            self._log_debug(f"Model: {model}")
            self._log_debug(f"Max context tokens: {max_context_tokens}")

            # Get requested completion tokens
            requested_completion = (
                data.get("max_tokens")
                or data.get("max_completion_tokens")
                or self.max_output_tokens
            )

            self._log_debug(f"Requested completion tokens: {requested_completion}")

            # Count current input tokens
            try:
                current_tokens = token_counter(model=model, messages=data["messages"])
            except Exception as e:
                # Fallback: estimate tokens
                current_tokens = sum(
                    len(str(msg.get("content", "")).split()) * 1.3
                    for msg in data["messages"]
                )
                self._log_debug(f"Token counting failed ({e}), using estimate")

            self._log_debug(f"Current input tokens: {current_tokens}")

            # Calculate safe completion tokens FIRST based on current input
            # Use a larger safety margin to account for LiteLLM adding extra tokens
            safe_completion_tokens = self._calculate_safe_completion_tokens(
                max_context_tokens, current_tokens, requested_completion
            )

            self._log_debug(f"Safe completion tokens: {safe_completion_tokens}")
            self._log_debug(
                f"Calculation: min({requested_completion}, max(512, ({max_context_tokens} - {int(current_tokens)} - {self.safety_buffer}) * 0.90))"
            )

            # Update completion tokens in the request
            has_max_tokens = "max_tokens" in data
            has_max_completion = "max_completion_tokens" in data

            self._update_completion_tokens(
                data, safe_completion_tokens, has_max_tokens, has_max_completion
            )

            # Now calculate max input tokens based on the safe completion tokens
            # Be even more conservative here
            max_input_tokens = int(
                (max_context_tokens - safe_completion_tokens - self.safety_buffer)
                * 0.85
            )  # Add 85% multiplier for extra safety

            self._log_debug(f"Max input tokens allowed: {max_input_tokens}")

            # ALWAYS trim to be safe, accounting for tokens that will be added later
            if current_tokens > max_input_tokens:
                self._log_debug(
                    f"Input tokens ({current_tokens}) exceed limit ({max_input_tokens}), trimming messages..."
                )
                data["messages"] = trim_messages(
                    data["messages"],
                    model=model,
                    max_tokens=int(max_input_tokens * 0.90),
                    trim_ratio=self.trim_ratio,
                )
            else:
                self._log_debug(
                    f"No trimming needed, but current={current_tokens}, max={max_input_tokens}"
                )

            # Always sanitize: repair tool-call pairings and ensure the conversation
            # does not end on an assistant message (rejected by vLLM chat template).
            data["messages"] = self._sanitize_messages(data["messages"])

            # Recount once after sanitize — messages may have grown (user continue
            # appended) or shrunk (orphan tool / empty assistant dropped).
            try:
                final_tokens = token_counter(model=model, messages=data["messages"])
                self._log_debug(f"After sanitize, input tokens: {final_tokens}")
                safe_completion_tokens = self._calculate_safe_completion_tokens(
                    max_context_tokens, final_tokens, requested_completion
                )
                self._update_completion_tokens(
                    data, safe_completion_tokens, has_max_tokens, has_max_completion
                )
                current_tokens = final_tokens
            except Exception as e:
                self._log_debug(f"Failed to recount tokens after sanitize: {e}")

            self._log_debug(
                f"Expected total: input ~{int(current_tokens)} + completion {safe_completion_tokens} + buffer {self.safety_buffer} = ~{int(current_tokens) + safe_completion_tokens + self.safety_buffer}/{max_context_tokens}"
            )

        return data
