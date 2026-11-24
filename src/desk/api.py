from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.special import softmax

from . import store
from .agent.loop import Agent
from .agent.planners import ClaudePlanner, RulePlanner
from .ml.baseline import Baseline
from .ml.data import LABELS, train_set
from .ml.registry import Registry
from .rag.index import Index, load_sections

DATA = Path(__file__).resolve().parents[2] / "data"
# below this, a person reads the message before any automatic step
MIN_CONFIDENCE = 0.45


class Message(BaseModel):
    email: str = Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$", max_length=120)
    body: str = Field(min_length=3, max_length=2000)


class Decision(BaseModel):
    approve: bool
    person: str = Field(min_length=2, max_length=80)


class Triage:
    """The champion model, or the baseline where the small model's weights are not installed."""

    def __init__(self, registry: Registry | None = None):
        registry = registry or Registry()
        champion = registry.stages()["champion"] or "v1-tfidf"
        manifest = registry.manifest(champion) if champion in registry.versions() else None
        weights = os.getenv("BASE_MODEL")
        adapter = registry.root / champion / "adapter.safetensors"
        if weights and adapter.exists():
            from .ml.slm import SmallModel

            self.model = SmallModel.load(weights)
            self.model.load_adapter(adapter)
            self.name, self.t = champion, manifest["metrics"]["temperature"]
        else:
            self.model, self.name, self.t = Baseline().train(train_set()), "v1-tfidf", 1.0
            if "v1-tfidf" in registry.versions():
                self.t = registry.manifest("v1-tfidf")["metrics"]["temperature"]

    def __call__(self, text: str) -> tuple[str, float]:
        p = softmax(self.model.scores([text]) / self.t, axis=1)[0]
        return LABELS[p.argmax()], float(p.max())


def create_app(url: str | None = None, live: bool | None = None, triage=None) -> FastAPI:
    live = bool(os.getenv("LIVE_AGENT")) if live is None else live

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = store.connect(url or os.getenv("DATABASE_URL", "sqlite:///desk.db"))
        app.state.index = Index(load_sections(DATA / "articles"))
        app.state.agent = Agent(app.state.engine, app.state.index)
        app.state.triage = triage or Triage()
        app.state.paused = {}
        yield
        app.state.engine.dispose()

    app = FastAPI(title="Support desk", version="1.0.0", lifespan=lifespan)

    def view(ticket: int) -> dict:
        with app.state.engine.connect() as c:
            t = (
                c.execute(sa.select(store.tickets).where(store.tickets.c.id == ticket))
                .mappings()
                .first()
            )
            if not t:
                raise HTTPException(404, "There is no such ticket.")
            rows = c.execute(
                sa.select(store.steps)
                .where(store.steps.c.ticket == ticket)
                .order_by(store.steps.c.n)
            ).mappings()
            waiting = (
                c.execute(
                    sa.select(store.approvals).where(
                        store.approvals.c.ticket == ticket, store.approvals.c.status == "pending"
                    )
                )
                .mappings()
                .first()
            )
            return {
                **t,
                "steps": [{"kind": r["kind"], **r["body"]} for r in rows],
                "approval": dict(waiting) if waiting else None,
            }

    @app.post("/api/tickets", status_code=201)
    def create(m: Message):
        intent, confidence = app.state.triage(m.body)
        with app.state.engine.begin() as c:
            r = c.execute(
                store.tickets.insert().values(
                    email=m.email.lower(),
                    body=m.body,
                    intent=intent,
                    confidence=confidence,
                    status="new",
                    created_at=store.now(),
                )
            )
            ticket = r.inserted_primary_key[0]
            store.add_step(
                c,
                ticket,
                "triage",
                {
                    "intent": intent,
                    "confidence": confidence,
                    "model": app.state.triage.name if hasattr(app.state.triage, "name") else "test",
                },
            )
        return view(ticket)

    @app.get("/api/tickets")
    def inbox():
        with app.state.engine.connect() as c:
            return [
                dict(r)
                for r in c.execute(
                    sa.select(store.tickets).order_by(store.tickets.c.id.desc()).limit(100)
                ).mappings()
            ]

    @app.get("/api/tickets/{ticket}")
    def show(ticket: int):
        return view(ticket)

    @app.post("/api/tickets/{ticket}/run")
    def run(ticket: int):
        t = view(ticket)
        agent = app.state.agent
        if t["status"] == "new" and t["confidence"] < MIN_CONFIDENCE:
            with app.state.engine.begin() as c:
                if c.execute(
                    store.tickets.update()
                    .where(store.tickets.c.id == ticket, store.tickets.c.status == "new")
                    .values(status="waiting")
                ).rowcount:
                    agent._hand_over(c, ticket, "The triage model was not sure")
            return view(ticket)
        planner = ClaudePlanner() if live else RulePlanner(t["intent"], t["body"])
        if agent.run(ticket, planner) == "waiting":
            app.state.paused[ticket] = (planner, agent.history)
        return view(ticket)

    @app.post("/api/approvals/{approval}")
    def decide(approval: int, d: Decision):
        with app.state.engine.connect() as c:
            a = (
                c.execute(sa.select(store.approvals).where(store.approvals.c.id == approval))
                .mappings()
                .first()
            )
        if not a:
            raise HTTPException(404, "There is no such approval.")
        paused = app.state.paused.pop(a["ticket"], None)
        if a["status"] != "pending" or not paused:
            raise HTTPException(409, "That has already been decided.")
        app.state.agent.decide(approval, d.approve, d.person, *paused)
        return view(a["ticket"])

    @app.get("/api/models")
    def models():
        reg = Registry()
        return {"stages": reg.stages(), "models": [reg.manifest(v) for v in reg.versions()]}

    return app
