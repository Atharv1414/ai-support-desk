"""Training messages from templates, and a test set nobody generated.

Templates make cheap training data and flattering test data, so they are kept apart: training and
validation use different templates, and the test set is 120 messages written by hand, without
looking at the templates, in the way customers actually write.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / "data" / "test_messages.csv"

# The label words double as the language model's answers, so each must be one distinct token
INTENTS = {
    "late": "a delivery that has not arrived",
    "damaged": "an item damaged on arrival",
    "fault": "a product that has stopped working properly",
    "refund": "money not yet paid back",
    "return": "wanting to send an item back",
    "cancel": "stopping an order before it ships",
    "address": "changing where an order goes",
    "warranty": "what the guarantee covers",
    "help": "how to use a product",
    "charge": "a wrong or unexpected payment",
}
LABELS = list(INTENTS)

ITEMS = [
    "kettle",
    "toaster",
    "blender",
    "vacuum",
    "fan heater",
    "coffee machine",
    "oven",
    "iron",
    "microwave",
    "fridge",
    "air fryer",
    "steamer",
]
WHEN = [
    "on Monday",
    "last week",
    "three days ago",
    "in May",
    "a fortnight ago",
    "yesterday",
    "on the 12th",
    "before the weekend",
]
OPEN = ["", "", "", "Hi, ", "Hello, ", "Hi there, ", "Good morning. ", "hey "]
CLOSE = [
    "",
    "",
    "",
    " Thanks.",
    " Please help.",
    " Can someone look into this?",
    " Regards, Sam",
    " My order is HF-{ref}.",
]

TEMPLATES = {
    "late": [
        "My {item} has not been delivered yet",
        "I ordered a {item} {when} and it has not come",
        "Where is my {item}? It should have arrived {when}",
        "The delivery of my {item} is late",
        "My parcel was due {when} and I have not received it",
        "Tracking has not updated since {when}",
        "The courier never showed up with my {item}",
        "When will my {item} be delivered",
        "No sign of the {item} I bought {when}",
        "Delivery keeps being pushed back",
    ],
    "damaged": [
        "My {item} arrived broken",
        "The {item} was damaged when it was delivered",
        "I unpacked the {item} and it is cracked",
        "There is a dent in my new {item}",
        "The box was crushed and the {item} inside is smashed",
        "My {item} came with a scratch on the front",
        "The {item} was delivered in pieces",
        "Part of the {item} was snapped when it got here",
        "The {item} turned up with a chipped edge",
        "Received a shattered {item} {when}",
    ],
    "fault": [
        "My {item} has stopped working",
        "The {item} will not turn on",
        "My {item} is making a strange noise",
        "The {item} I bought {when} is faulty",
        "The {item} keeps cutting out",
        "My {item} shows an error code",
        "The {item} is leaking",
        "The {item} overheats after a few minutes",
        "My {item} does not heat up any more",
        "There is smoke coming from my {item}",
    ],
    "refund": [
        "When will I get my refund for the {item}",
        "I have not received my refund yet",
        "I sent the {item} back {when} and have not been refunded",
        "My refund for the {item} is missing",
        "How long do refunds take",
        "I was told I would get my money back {when}",
        "You have not paid me back for the {item}",
        "The refund amount is wrong",
        "My bank has no record of your refund",
        "Still waiting to be reimbursed",
    ],
    "return": [
        "I want to return my {item}",
        "How do I send the {item} back",
        "I would like to return the {item} I bought {when}",
        "Can I return an opened {item}",
        "I need a returns label for my {item}",
        "The {item} is not for me, can I return it",
        "What is your returns policy",
        "I changed my mind about the {item}",
        "Where can I drop off a {item} to return it",
        "Can you collect a {item} I don't want",
    ],
    "cancel": [
        "Please cancel my order for the {item}",
        "I want to cancel the {item} I ordered {when}",
        "Can I cancel my order before it ships",
        "Cancel my {item} please",
        "I ordered the {item} by mistake, please cancel it",
        "How do I cancel an order",
        "I no longer need the {item}, stop the order",
        "Please do not dispatch my {item}",
        "I placed an order {when} and need to cancel it",
        "Withdraw my order please",
    ],
    "address": [
        "I need to change the delivery address for my {item}",
        "My {item} is going to the wrong address",
        "Please update my address on the order",
        "I have moved house, can you send the {item} to my new address",
        "I entered the wrong postcode for my {item}",
        "Can my {item} go to a neighbour",
        "The delivery address is out of date",
        "Please deliver the {item} to my office",
        "I made a mistake in the street name",
        "Change the drop-off location for my parcel",
    ],
    "warranty": [
        "Is my {item} still under warranty",
        "How long is the warranty on the {item}",
        "What does the {item} guarantee cover",
        "Does the warranty cover parts",
        "How do I register my {item} for the guarantee",
        "Do I need proof of purchase for a warranty claim",
        "Can I extend the warranty on my {item}",
        "Is wear and tear covered on the {item}",
        "I bought the {item} second hand, does the warranty apply",
        "When does the cover on my {item} run out",
    ],
    "help": [
        "How do I clean my {item}",
        "How do I use the {item}",
        "What do the {item} buttons do",
        "Where can I find the {item} manual",
        "How do I descale the {item}",
        "Which programme should I use on the {item}",
        "Can the {item} parts go in a dishwasher",
        "How do I change the settings on my {item}",
        "How do I connect the {item} to wifi",
        "What is the light on my {item} for",
    ],
    "charge": [
        "I was charged twice for the {item}",
        "There is a wrong amount on my bill",
        "I have an unexpected charge from you",
        "My discount was not applied to the {item}",
        "Why was I charged for delivery",
        "I paid too much for the {item}",
        "There is a payment on my card I do not recognise",
        "I need an invoice for the {item}",
        "My voucher did not work when I paid for the {item}",
        "The price I paid for the {item} is different from the website",
    ],
}


@dataclass(frozen=True)
class Example:
    text: str
    intent: str


def _typo(text: str, rng: random.Random) -> str:
    if len(text) < 12:
        return text
    i = rng.randrange(3, len(text) - 3)
    return rng.choice(
        [
            text[:i] + text[i + 1 :],
            text[:i] + text[i + 1] + text[i] + text[i + 2 :],
            text[:i] + text[i] + text[i:],
        ]
    )


def generate(per_intent: int, held_out: bool, seed: int) -> list[Example]:
    """`held_out` picks the templates reserved for validation (the last three of each intent)."""
    rng = random.Random(seed)
    out, seen = [], set()
    for intent, templates in TEMPLATES.items():
        pool = templates[-3:] if held_out else templates[:-3]
        made = 0
        while made < per_intent:
            body = rng.choice(pool).format(item=rng.choice(ITEMS), when=rng.choice(WHEN))
            text = rng.choice(OPEN) + body + rng.choice(["", ".", "?", "!"])
            text += rng.choice(CLOSE).format(ref=rng.randint(10000, 39999))
            if rng.random() < 0.25:
                text = text.lower()
            if rng.random() < 0.2:
                text = _typo(text, rng)
            if text in seen:
                continue
            seen.add(text)
            out.append(Example(text, intent))
            made += 1
    rng.shuffle(out)
    return out


def train_set() -> list[Example]:
    return generate(150, False, seed=1)


def validation_set() -> list[Example]:
    return generate(30, True, seed=2)


def test_set(path: Path = TEST) -> list[Example]:
    with path.open(newline="") as f:
        return [Example(r["text"], r["intent"]) for r in csv.DictReader(f)]


def fingerprint(examples: list[Example]) -> str:
    """A short hash that changes if any example does, recorded with every model."""
    blob = json.dumps([(e.text, e.intent) for e in examples]).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
