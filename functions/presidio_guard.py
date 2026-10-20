"""
title: Presidio PII Guard (block only)
"""

import asyncio
import logging

import aiohttp
from pydantic import BaseModel

log = logging.getLogger(__name__)
UNAVAILABLE = "PII check unavailable, try again later."


class Filter:
    class Valves(BaseModel):
        analyzer_url: str = "http://presidio-analyzer:3000"
        language: str = "da"
        score_threshold: float = 0.5
        # comma-separated Presidio entity types to block, or * for all
        blocked_entities: str = (
            "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,IBAN_CODE,DK_CPR,DK_ADDRESS"
        )
        target_models: str = "*"

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: dict = None) -> dict:
        targets = [m.strip() for m in self.valves.target_models.split(",")]
        if "*" not in targets and body.get("model") not in targets:
            return body

        # Check every user message: a blocked message stays in history until edited out
        texts = []
        for m in body.get("messages", []):
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str):
                if content:
                    texts.append(content)
            elif isinstance(content, list):
                texts += [p.get("text", "") for p in content if p.get("type") == "text"]
        text = "\n\n".join(texts)
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

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as s:
                async with s.post(
                    f"{self.valves.analyzer_url}/analyze", json=payload
                ) as r:
                    r.raise_for_status()
                    results = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.exception("Presidio analyzer request failed")
            raise Exception(UNAVAILABLE) from e

        if results:
            found = sorted({r["entity_type"] for r in results})
            raise Exception(
                f"Message blocked: personal data detected ({', '.join(found)}). "
                "Remove it from the conversation (edit the message or start a new chat) and try again."
            )
        return body
