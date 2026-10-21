import asyncio

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from presidio_anonymize import Filter as Anonymizer
from presidio_guard import Filter as Guard

ANALYZE = "http://presidio-analyzer:3000/analyze"


def conversation(first):
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": first},
            {"role": "assistant", "content": "Hej"},
            {"role": "user", "content": "Hvad er vejret?"},
        ],
    }


def fake_analyzer(url, **kwargs):
    if "Mette" in kwargs["json"]["text"]:
        return CallbackResult(
            payload=[{"entity_type": "PERSON", "start": 0, "end": 5, "score": 0.85}]
        )
    return CallbackResult(payload=[])


def test_guard_blocks_pii_in_earlier_user_message():
    with aioresponses() as m:
        m.post(ANALYZE, callback=fake_analyzer)
        with pytest.raises(Exception, match="Message blocked"):
            asyncio.run(Guard().inlet(conversation("Jeg hedder Mette Frederiksen")))


def test_guard_passes_clean_conversation():
    body = conversation("Hvad laver kommunen?")
    with aioresponses() as m:
        m.post(ANALYZE, callback=fake_analyzer)
        assert asyncio.run(Guard().inlet(body)) is body


FAILURES = {
    "timeout": {"exception": asyncio.TimeoutError()},
    "refused": {"exception": aiohttp.ClientConnectionError()},
    "http500": {"status": 500},
}


@pytest.mark.parametrize("failure", FAILURES.values(), ids=FAILURES.keys())
@pytest.mark.parametrize("filter_cls", [Guard, Anonymizer])
def test_presidio_failure_raises_clear_message(filter_cls, failure):
    body = {"model": "m", "messages": [{"role": "user", "content": "Hej"}]}
    with aioresponses() as m:
        m.post(ANALYZE, **failure)
        with pytest.raises(Exception, match="PII check unavailable"):
            asyncio.run(filter_cls().inlet(body))
