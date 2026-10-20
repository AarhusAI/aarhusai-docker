"""Runs against a real analyzer: set PRESIDIO_ANALYZER_URL (e.g. http://presidio-analyzer:3000)."""

import json
import os
import urllib.request

import pytest

from presidio_guard import Filter as Guard

URL = os.environ.get("PRESIDIO_ANALYZER_URL")
if not URL:
    pytest.skip("PRESIDIO_ANALYZER_URL not set", allow_module_level=True)


def analyze(text):
    v = Guard.Valves()
    payload = {
        "text": text,
        "language": v.language,
        "score_threshold": v.score_threshold,
        "entities": [e.strip() for e in v.blocked_entities.split(",") if e.strip()],
    }
    req = urllib.request.Request(
        f"{URL}/analyze",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return {res["entity_type"] for res in json.load(r)}


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Hej, jeg hedder Mette Frederiksen og bor på Rådhuspladsen 2, 8000 Aarhus C.",
            {"PERSON", "DK_ADDRESS"},
        ),
        ("Lars Løkke Rasmussen mødtes med Søren Pind.", {"PERSON"}),
        ("Adressen er Søren Frichs Vej 42, 8230 Åbyhøj", {"DK_ADDRESS"}),
        ("Jeg bor på Nørregade 12, 2. th.", {"DK_ADDRESS"}),
        ("Ring til Jens Hansen på 20 30 40 50", {"PERSON", "PHONE_NUMBER"}),
        ("tlf 20304050", {"PHONE_NUMBER"}),
        ("Mobil: 2030 4050", {"PHONE_NUMBER"}),
        ("Mit nummer er +45 87 65 43 21", {"PHONE_NUMBER"}),
        ("Call me at +1 212 555 0100", {"PHONE_NUMBER"}),
        ("Mit CPR-nummer er 010190-1234.", {"DK_CPR"}),
        ("CPR 0101901234", {"DK_CPR"}),
        ("Borgerens personnummer: 311299-4321", {"DK_CPR"}),
        ("Send en mail til anne.jensen@aarhus.dk", {"EMAIL_ADDRESS"}),
        ("Kontonummer DK5000400440116243", {"IBAN_CODE"}),
    ],
)
def test_detects(text, expected):
    assert expected <= analyze(text)


@pytest.mark.parametrize(
    "text",
    [
        "Mødet er den 15-03-2024 kl. 10.",
        "Ordrenummer 4839201756 er afsendt.",
        "Faktura 2024-00123 skal betales.",
        "Vi sælger 32345678 enheder",
        "CVR 12345678",
        "Der er plads 10 personer",
        "Hvad er åbningstiderne på Dokk1 i Aarhus?",
        "Kan du opsummere lokalplan 1234 for Aarhus Kommune?",
    ],
)
def test_ignores(text):
    assert analyze(text) == set()
