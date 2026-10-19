"""
title: Presidio PII Filter (REST)
"""

import aiohttp
from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        analyzer_url: str = "http://presidio-analyzer:3000"
        anonymizer_url: str = "http://presidio-anonymizer:3000"
        language: str = "en"
        score_threshold: float = 0.4
        target_models: str = "*"  # comma-separated model IDs, or *

    def __init__(self):
        self.valves = self.Valves()

    async def _scrub(self, session: aiohttp.ClientSession, text: str) -> str:
        async with session.post(
            f"{self.valves.analyzer_url}/analyze",
            json={
                "text": text,
                "language": self.valves.language,
                "score_threshold": self.valves.score_threshold,
            },
        ) as r:
            r.raise_for_status()
            results = await r.json()
        if not results:
            return text
        async with session.post(
            f"{self.valves.anonymizer_url}/anonymize",
            json={"text": text, "analyzer_results": results},
        ) as r:
            r.raise_for_status()
            return (await r.json())["text"]

    async def inlet(self, body: dict, __user__: dict = None) -> dict:
        targets = [m.strip() for m in self.valves.target_models.split(",")]
        if "*" not in targets and body.get("model") not in targets:
            return body

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for msg in body.get("messages", []):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = await self._scrub(session, content)
                elif isinstance(content, list):  # multimodal messages
                    for part in content:
                        if part.get("type") == "text":
                            part["text"] = await self._scrub(session, part["text"])
        return body
