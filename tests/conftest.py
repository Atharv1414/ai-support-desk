import os
from datetime import date

import pytest
import sqlalchemy as sa

from desk import store
from desk.agent.loop import Agent
from desk.rag.evaluate import DATA
from desk.rag.index import Index, load_sections

TODAY = date(2026, 11, 24)


@pytest.fixture(scope="session")
def index():
    return Index(load_sections(DATA / "articles"))


@pytest.fixture
def engine():
    e = store.connect(os.getenv("DATABASE_URL", "sqlite://"))
    if e.dialect.name == "sqlite":
        e = store.connect("sqlite:///:memory:")
    store.seed(e)
    yield e
    e.dispose()


@pytest.fixture
def agent(engine, index):
    return Agent(engine, index, lambda: TODAY)


@pytest.fixture
def ticket(engine):
    def make(body, email="ada@example.com", intent="help"):
        with engine.begin() as c:
            return c.execute(
                store.tickets.insert().values(
                    email=email,
                    body=body,
                    intent=intent,
                    confidence=0.9,
                    status="new",
                    created_at=store.now(),
                )
            ).inserted_primary_key[0]

    return make


@pytest.fixture
def trace(engine):
    def read(ticket):
        with engine.connect() as c:
            return [
                (r.kind, r.body)
                for r in c.execute(
                    sa.select(store.steps)
                    .where(store.steps.c.ticket == ticket)
                    .order_by(store.steps.c.n)
                )
            ]

    return read
