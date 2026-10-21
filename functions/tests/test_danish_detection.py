"""Runs against a real analyzer: set PRESIDIO_ANALYZER_URL (e.g. http://presidio-analyzer:3000)."""

import json
import os
import urllib.request

import pytest

from long_texts import TEXTS
from presidio_anonymize import Filter as Anonymizer
from presidio_guard import Filter as Guard

URL = os.environ.get("PRESIDIO_ANALYZER_URL")
if not URL:
    pytest.skip("PRESIDIO_ANALYZER_URL not set", allow_module_level=True)


def analyze(payload):
    req = urllib.request.Request(
        f"{URL}/analyze",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return {res["entity_type"] for res in json.load(r)}


def guard(text):
    return analyze(Guard()._payload(text))


def anon(text):
    return analyze(Anonymizer()._payload(text))


GUARD_ENTITIES = {e.strip() for e in Guard.Valves().blocked_entities.split(",")}


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
        ("Ring til Jens Hansen på tlf 20 30 40 50", {"PERSON", "PHONE_NUMBER"}),
        ("tlf 20304050", {"PHONE_NUMBER"}),
        ("Mobil: 2030 4050", {"PHONE_NUMBER"}),
        ("Mit nummer er +45 87 65 43 21", {"PHONE_NUMBER"}),
        ("Call me at +1 212 555 0100", {"PHONE_NUMBER"}),
        ("Kontakt mig på (+45) 87 65 43 21", {"PHONE_NUMBER"}),
        ("Nummeret er 0045 87654321", {"PHONE_NUMBER"}),
        ("Mit CPR-nummer er 010190-1234.", {"DK_CPR"}),
        ("CPR 0101901234", {"DK_CPR"}),
        ("CPR 010190 1234", {"DK_CPR"}),
        ("Borgerens personnummer: 311299-4321", {"DK_CPR"}),
        ("Send en mail til anne.jensen@aarhus.dk", {"EMAIL_ADDRESS"}),
        ("Kontonummer DK5000400440116243", {"IBAN_CODE"}),
    ],
)
def test_detects(text, expected):
    assert expected <= anon(text)
    assert expected & GUARD_ENTITIES <= guard(text)


@pytest.mark.parametrize(
    "text",
    [
        "Mødet er den 15-03-2024 kl. 10.",
        "Ordrenummer 4839201756 er afsendt.",
        "Faktura 2024-00123 skal betales.",
        "Vi sælger 32345678 enheder",
        "CVR 55133018",
        "Der er plads 10 personer",
        "Hvad er åbningstiderne på Dokk1 i Aarhus?",
        "Kan du opsummere lokalplan 1234 for Aarhus Kommune?",
        "Sagsnummer 87654321",
        "Jeg ringer om sag 55133018",
    ],
)
def test_ignores(text):
    assert guard(text) == set()
    assert anon(text) == set()


@pytest.mark.parametrize(
    "text",
    [
        "Hvad sagde Mette Frederiksen i åbningstalen?",
        "Hvornår er der åbent på Rådhuspladsen 2?",
        "Hvornår er vejarbejdet på Skanderborgvej 100 færdigt?",
    ],
)
def test_guard_allows_public_names_and_places(text):
    assert guard(text) == set()


def test_allow_list():
    text = "Hvad sagde Mette Frederiksen i åbningstalen?"
    assert "PERSON" in anon(text)
    a = Anonymizer()
    a.valves.allow_list = "Mette Frederiksen"
    assert "PERSON" not in analyze(a._payload(text))


PII = {
    "cpr": ("Mit CPR-nummer er 010190-1234.", "DK_CPR"),
    "name": ("Jeg hedder Jens Peter Hansen.", "PERSON"),
    "address": ("Jeg bor på Søren Frichs Vej 42, 8230 Åbyhøj.", "DK_ADDRESS"),
}


def insert(text, sentence, pos):
    if pos == "start":
        return f"{sentence} {text}"
    if pos == "end":
        return f"{text} {sentence}"
    parts = text.split(". ")
    mid = len(parts) // 2
    return ". ".join(parts[:mid]) + f". {sentence} " + ". ".join(parts[mid:])


LONG_IDS = [f"long{i}" for i in range(len(TEXTS))]


@pytest.mark.parametrize("text", TEXTS, ids=LONG_IDS)
def test_long_text_clean(text):
    assert guard(text) == set()
    assert anon(text) == set()


@pytest.mark.parametrize("pos", ["start", "middle", "end"])
@pytest.mark.parametrize("kind", PII)
@pytest.mark.parametrize("text", TEXTS, ids=LONG_IDS)
def test_long_text_with_pii(text, kind, pos):
    sentence, entity = PII[kind]
    t = insert(text, sentence, pos)
    assert entity in anon(t)
    if entity in GUARD_ENTITIES:
        assert entity in guard(t)
    else:
        assert guard(t) == set()
