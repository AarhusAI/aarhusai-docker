"""
title: Presidio PII Guard (block only)
"""

import aiohttp
from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        analyzer_url: str = "http://presidio-analyzer:3000"
        language: str = "en"
        score_threshold: float = 0.5
        # comma-separated Presidio entity types to block, or * for all
        blocked_entities: str = (
            "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,IBAN_CODE"
        )
        target_models: str = "*"

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: dict = None) -> dict:
        targets = [m.strip() for m in self.valves.target_models.split(",")]
        if "*" not in targets and body.get("model") not in targets:
            return body

        # Only check the newest user message; earlier ones were already checked
        user_msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
        if not user_msgs:
            return body
        content = user_msgs[-1].get("content")
        if isinstance(content, list):
            text = "\n".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        else:
            text = content or ""
        if not text.strip():
            return body

        payload = {
            "text": text,
            "language": self.valves.language,
            "score_threshold": self.valves.score_threshold,
        }
        if self.valves.blocked_entities.strip() != "*":
            payload["entities"] = [
                e.strip() for e in self.valves.blocked_entities.split(",") if e.strip()
            ]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.post(f"{self.valves.analyzer_url}/analyze", json=payload) as r:
                r.raise_for_status()
                results = await r.json()

        if results:
            found = sorted({r["entity_type"] for r in results})
            raise Exception(
                f"Message blocked: personal data detected ({', '.join(found)}). "
                "Please remove it and try again."
            )
        return body
