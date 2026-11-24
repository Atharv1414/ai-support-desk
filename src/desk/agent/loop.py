"""The agent loop: bounded, recorded, and able to stop and wait for a person."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from ..rag.index import Index
from ..store import add_step, approvals, steps, tickets
from .planners import Call, Planner, Turn
from .tools import TOOLS, Context, ToolError, check_refund

MAX_STEPS = 8
DANGER = re.compile(
    r"\b(smok\w+|spark\w*|burn\w*|fire|flames?|electric(al)? shock|shocked me|bang|"
    r"melt\w*|explod\w*)\b",
    re.I,
)
CITATION = re.compile(r"\[([a-z0-9-]+#[a-z0-9-]+)\]")


class Agent:
    def __init__(self, engine: Engine, index: Index, today: Callable[[], date] = date.today):
        self.engine, self.index, self.today = engine, index, today

    def run(self, ticket: int, planner: Planner) -> str:
        """Work on a new ticket. Returns its status."""
        with self.engine.begin() as c:
            # claiming the ticket is one conditional update, so two workers cannot both take it
            claimed = c.execute(
                tickets.update()
                .where(tickets.c.id == ticket, tickets.c.status == "new")
                .values(status="waiting")
            )
            row = c.execute(sa.select(tickets).where(tickets.c.id == ticket)).mappings().first()
            if not claimed.rowcount:
                return row["status"] if row else "missing"
            if DANGER.search(row["body"]):  # decided before any model sees the message
                add_step(c, ticket, "safety", {"matched": DANGER.search(row["body"]).group(0)})
                return self._hand_over(c, ticket, "Possible danger to a person")
        history = [
            {"role": "user", "content": f"<customer_message>\n{row['body']}\n</customer_message>"}
        ]
        return self._continue(ticket, row["email"], planner, history)

    def decide(
        self, approval: int, approve: bool, person: str, planner: Planner, history: list[dict]
    ) -> str:
        """A person's answer to a pending action. Only the first answer counts."""
        with self.engine.begin() as c:
            won = c.execute(
                approvals.update()
                .where(approvals.c.id == approval, approvals.c.status == "pending")
                .values(status="approved" if approve else "rejected", decided_by=person)
            )
            a = c.execute(sa.select(approvals).where(approvals.c.id == approval)).mappings().one()
            t = c.execute(sa.select(tickets).where(tickets.c.id == a["ticket"])).mappings().one()
            if not won.rowcount:
                return t["status"]
            ctx = Context(c, t["email"], self.index, self.today())
            add_step(c, t["id"], "decision", {"approved": approve, "by": person, "tool": a["tool"]})
            if approve:
                try:
                    data, bad = TOOLS[a["tool"]].run(ctx, a["args"], approval), False
                except ToolError as e:  # the world may have changed while it waited
                    data, bad = {"error": str(e)}, True
            else:
                data, bad = {"declined": True}, False
            add_step(c, t["id"], "result", {"tool": a["tool"], "data": data, "error": bad})
            call = history[-1]["content"][-1]
            history[-1]["content"][-1] = self._result(
                Call(call["tool_use_id"], a["tool"], a["args"]), data, bad
            )
        return self._continue(t["id"], t["email"], planner, history)

    def _continue(self, ticket: int, email: str, planner: Planner, history: list[dict]) -> str:
        self.history = history
        for _ in range(MAX_STEPS):
            try:
                turn = planner.next(history)
            except Exception as e:  # noqa: BLE001 - whatever failed, a person takes over
                with self.engine.begin() as c:
                    return self._hand_over(c, ticket, f"The planner failed: {type(e).__name__}")
            with self.engine.begin() as c:
                if turn.stopped:
                    return self._hand_over(c, ticket, f"The model stopped: {turn.stopped}")
                if not turn.calls:
                    return self._answer(c, ticket, turn, history)
                status = self._act(c, ticket, email, turn, history)
                if status:
                    return status
        with self.engine.begin() as c:
            return self._hand_over(c, ticket, f"No answer after {MAX_STEPS} steps")

    def _act(self, c, ticket: int, email: str, turn: Turn, history: list[dict]) -> str | None:
        ctx = Context(c, email, self.index, self.today())
        history.append(
            {
                "role": "assistant",
                "content": turn.raw
                or [
                    {"type": "tool_use", "id": k.id, "name": k.name, "input": k.args}
                    for k in turn.calls
                ],
            }
        )
        results: list[dict] = []
        history.append({"role": "user", "content": results})
        paused = False
        for i, call in enumerate(turn.calls):
            add_step(c, ticket, "call", {"tool": call.name, "args": call.args})
            tool = TOOLS.get(call.name)
            try:
                if tool is None:
                    raise ToolError(f"there is no tool called {call.name}")
                if tool.needs_person:
                    if i != len(turn.calls) - 1:
                        raise ToolError("ask for a refund on its own, after the other calls")
                    check_refund(ctx, call.args)  # never ask a person to approve the impossible
                    a = c.execute(
                        approvals.insert().values(ticket=ticket, tool=call.name, args=call.args)
                    )
                    add_step(
                        c,
                        ticket,
                        "approval",
                        {"id": a.inserted_primary_key[0], "tool": call.name, "args": call.args},
                    )
                    results.append(self._result(call, {"waiting_for_person": True}, False))
                    paused = True
                    break
                data, bad = tool.run(ctx, call.args), False
            except ToolError as e:
                data, bad = {"error": str(e)}, True
            add_step(c, ticket, "result", {"tool": call.name, "data": data, "error": bad})
            results.append(self._result(call, data, bad))
            if call.name == "escalate" and not bad:
                return self._hand_over(c, ticket, str(call.args.get("reason", ""))[:300])
        return "waiting" if paused else None

    @staticmethod
    def _result(call: Call, data: dict, bad: bool) -> dict:
        # name and data are for the offline planner; they are removed before anything is sent
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "is_error": bad,
            "name": call.name,
            "data": json.loads(json.dumps(data, default=str)),
            "content": json.dumps(data, default=str),
        }

    def _answer(self, c, ticket: int, turn: Turn, history: list[dict]) -> str:
        """Check a reply before it is sent: every cited section must be one that was retrieved."""
        got = {
            s["id"]
            for m in history
            for b in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(b, dict) and b.get("name") == "search_help"
            for s in b.get("data", {}).get("sections", [])
        }
        cited = CITATION.findall(turn.text)
        invented = [x for x in cited if x not in got]
        if invented or not turn.text.strip():
            return self._hand_over(
                c,
                ticket,
                "The reply cited something that was not retrieved: " + ", ".join(invented)
                if invented
                else "The reply was empty",
            )
        promised = re.search(r"\b(refunded|refund of)\b", turn.text, re.I)
        paid = c.execute(
            sa.select(steps.c.n).where(steps.c.ticket == ticket, steps.c.kind == "decision")
        ).first()
        if (
            promised
            and not paid
            and re.search(r"\b(have|has been|i've) refunded\b|has been approved", turn.text, re.I)
        ):
            return self._hand_over(c, ticket, "The reply claimed a refund that was never approved")
        add_step(c, ticket, "reply", {"text": turn.text, "cited": cited})
        c.execute(tickets.update().where(tickets.c.id == ticket).values(status="answered"))
        return "answered"

    @staticmethod
    def _hand_over(c, ticket: int, reason: str) -> str:
        add_step(c, ticket, "handover", {"reason": reason})
        c.execute(tickets.update().where(tickets.c.id == ticket).values(status="with_person"))
        return "with_person"
