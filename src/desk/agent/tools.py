"""What the agent can do, and what it can never do whatever a model asks for.

The rules live here, below the model. A tool checks its input, that the order belongs to the
customer who wrote in, and the shop's policy. Moving money always waits for a person.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from ..rag.index import Index
from ..store import orders, refunds


class ToolError(Exception):
    """Reported back to the model as a failed call. It does not stop the run."""


@dataclass
class Context:
    conn: Connection
    email: str
    index: Index
    today: date


def _order(ctx: Context, ref) -> dict:
    if not isinstance(ref, str) or len(ref) > 16:
        raise ToolError("order reference must be text like HF-20418")
    row = (
        ctx.conn.execute(
            sa.select(orders).where(
                orders.c.ref == ref.upper().strip(), orders.c.email == ctx.email
            )
        )
        .mappings()
        .first()
    )
    if not row:  # the same answer whether it does not exist or belongs to somebody else
        raise ToolError(f"this customer has no order {ref.upper().strip()}")
    return dict(row)


def list_orders(ctx: Context, args: dict) -> dict:
    rows = ctx.conn.execute(
        sa.select(orders).where(orders.c.email == ctx.email).order_by(orders.c.placed_on.desc())
    ).mappings()
    return {"orders": [dict(r) for r in rows]}


def find_order(ctx: Context, args: dict) -> dict:
    return _order(ctx, args.get("ref"))


def search_help(ctx: Context, args: dict) -> dict:
    q = args.get("query")
    if not isinstance(q, str) or not q.strip() or len(q) > 300:
        raise ToolError("query must be a short question")
    hits = ctx.index.search(q)
    if not ctx.index.answerable(hits):
        return {"found": False, "sections": []}
    return {
        "found": True,
        "sections": [
            {
                "id": h.section.id,
                "title": f"{h.section.article}: {h.section.heading}",
                "text": h.section.text,
            }
            for h in hits
        ],
    }


def cancel_order(ctx: Context, args: dict) -> dict:
    o = _order(ctx, args.get("ref"))
    if o["status"] != "placed":
        raise ToolError(f"cannot cancel: the order is already {o['status']}")
    ctx.conn.execute(
        orders.update()
        .where(orders.c.ref == o["ref"], orders.c.status == "placed")
        .values(status="cancelled")
    )
    return {"ref": o["ref"], "status": "cancelled"}


def start_return(ctx: Context, args: dict) -> dict:
    o = _order(ctx, args.get("ref"))
    if o["status"] != "delivered":
        raise ToolError(f"cannot return: the order is {o['status']}")
    days = (ctx.today - o["delivered_on"]).days
    if days > 30:
        raise ToolError(f"cannot return: delivered {days} days ago, and the limit is 30")
    return {"ref": o["ref"], "label": f"RL-{o['ref'][3:]}", "days_left": 30 - days}


def check_refund(ctx: Context, args: dict) -> tuple[dict, Decimal]:
    o = _order(ctx, args.get("ref"))
    try:
        amount = Decimal(str(args.get("amount"))).quantize(Decimal("0.01"))
    except InvalidOperation:
        raise ToolError("amount must be a number of pounds") from None
    if not amount.is_finite() or amount <= 0:
        raise ToolError("amount must be more than zero")
    paid = ctx.conn.execute(
        sa.select(sa.func.coalesce(sa.func.sum(refunds.c.amount), 0)).where(
            refunds.c.ref == o["ref"]
        )
    ).scalar_one()
    left = Decimal(str(o["price"])) - Decimal(str(paid))
    if amount > left:
        raise ToolError(f"cannot refund £{amount}: £{left:.2f} of this order has not been refunded")
    if o["status"] not in ("returned", "cancelled", "delivered"):
        raise ToolError(f"cannot refund: the order is {o['status']}")
    return o, amount


def issue_refund(ctx: Context, args: dict, approval: int) -> dict:
    o, amount = check_refund(ctx, args)
    ctx.conn.execute(refunds.insert().values(ref=o["ref"], amount=amount, approval=approval))
    return {"ref": o["ref"], "refunded": f"{amount:.2f}"}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict
    run: Callable
    needs_person: bool = False

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": list(self.properties),
                "additionalProperties": False,
            },
        }


REF = {"ref": {"type": "string", "description": "Order reference such as HF-20418"}}
TOOLS = {
    t.name: t
    for t in [
        Tool("list_orders", "List this customer's orders, newest first.", {}, list_orders),
        Tool("find_order", "Look up one of this customer's orders.", REF, find_order),
        Tool(
            "search_help",
            "Search the help centre. Policy answers must come from here.",
            {"query": {"type": "string"}},
            search_help,
        ),
        Tool("cancel_order", "Cancel an order that has not been dispatched.", REF, cancel_order),
        Tool(
            "start_return", "Create a free returns label for a delivered order.", REF, start_return
        ),
        Tool(
            "issue_refund",
            "Ask for a refund. A person must approve it before any money moves.",
            {**REF, "amount": {"type": "number"}, "reason": {"type": "string"}},
            issue_refund,
            True,
        ),
        Tool(
            "escalate",
            "Hand the conversation to a person.",
            {"reason": {"type": "string"}},
            lambda ctx, args: {"handed_over": True},
        ),
    ]
}
