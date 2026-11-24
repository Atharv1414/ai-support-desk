"""MySQL (or SQLite for a quick start): orders, tickets, agent runs, approvals, money."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import sqlalchemy as sa
from sqlalchemy.engine import Engine

meta = sa.MetaData()
Money = sa.Numeric(10, 2)

orders = sa.Table(
    "orders",
    meta,
    sa.Column("ref", sa.String(16), primary_key=True),
    sa.Column("email", sa.String(120), nullable=False, index=True),
    sa.Column("item", sa.String(80), nullable=False),
    sa.Column("price", Money, nullable=False),
    sa.Column(
        "status",
        sa.Enum("placed", "dispatched", "delivered", "returned", "cancelled", name="order_status"),
        nullable=False,
    ),
    sa.Column("placed_on", sa.Date, nullable=False),
    sa.Column("delivered_on", sa.Date),
    sa.CheckConstraint("price >= 0", name="price_not_negative"),
)
tickets = sa.Table(
    "tickets",
    meta,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("email", sa.String(120), nullable=False),
    sa.Column("body", sa.Text, nullable=False),
    sa.Column("intent", sa.String(16)),
    sa.Column("confidence", sa.Float),
    sa.Column(
        "status",
        sa.Enum("new", "waiting", "answered", "with_person", name="ticket_status"),
        nullable=False,
        default="new",
    ),
    sa.Column("created_at", sa.DateTime, nullable=False),
)
steps = sa.Table(
    "steps",
    meta,
    sa.Column("ticket", sa.ForeignKey("tickets.id"), primary_key=True),
    sa.Column("n", sa.Integer, primary_key=True),
    sa.Column("kind", sa.String(16), nullable=False),
    sa.Column("body", sa.JSON, nullable=False),
)
approvals = sa.Table(
    "approvals",
    meta,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("ticket", sa.ForeignKey("tickets.id"), nullable=False, index=True),
    sa.Column("tool", sa.String(40), nullable=False),
    sa.Column("args", sa.JSON, nullable=False),
    sa.Column(
        "status",
        sa.Enum("pending", "approved", "rejected", name="approval_status"),
        nullable=False,
        default="pending",
    ),
    sa.Column("decided_by", sa.String(80)),
)
refunds = sa.Table(
    "refunds",
    meta,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("ref", sa.ForeignKey("orders.ref"), nullable=False, index=True),
    # one refund per approval, enforced by the database as well as by the code
    sa.Column("approval", sa.ForeignKey("approvals.id"), nullable=False, unique=True),
    sa.Column("amount", Money, nullable=False),
    sa.CheckConstraint("amount > 0", name="amount_positive"),
)


def connect(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = sa.create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=sa.StaticPool if ":memory:" in url else None,
        )

        @sa.event.listens_for(engine, "connect")
        def _fk(conn, _):
            conn.execute("pragma foreign_keys=on")
    else:
        engine = sa.create_engine(url, pool_pre_ping=True)
    meta.create_all(engine)
    return engine


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


SAMPLE_ORDERS = [
    (
        "HF-20418",
        "ada@example.com",
        "Kettle, 1.7 litre",
        "49.00",
        "delivered",
        "2026-11-02",
        "2026-11-05",
    ),
    ("HF-20533", "ada@example.com", "Blender", "79.00", "placed", "2026-11-23", None),
    (
        "HF-21007",
        "ben@example.com",
        "Toaster, four slice",
        "59.00",
        "dispatched",
        "2026-11-10",
        None,
    ),
    ("HF-21090", "cal@example.com", "Fan heater", "64.00", "returned", "2026-10-01", "2026-10-04"),
    (
        "HF-21555",
        "dee@example.com",
        "Coffee machine",
        "149.00",
        "delivered",
        "2026-11-12",
        "2026-11-14",
    ),
    ("HF-22101", "eli@example.com", "Vacuum", "189.00", "delivered", "2025-08-20", "2025-08-24"),
]


def seed(engine: Engine) -> None:
    def d(v):
        return date.fromisoformat(v) if v else None

    with engine.begin() as c:
        for t in (refunds, approvals, steps, tickets, orders):
            c.execute(t.delete())
        c.execute(
            orders.insert(),
            [
                dict(ref=r, email=e, item=i, price=p, status=s, placed_on=d(a), delivered_on=d(b))
                for r, e, i, p, s, a, b in SAMPLE_ORDERS
            ],
        )


def add_step(c, ticket: int, kind: str, body: dict) -> None:
    n = c.execute(
        sa.select(sa.func.coalesce(sa.func.max(steps.c.n), 0)).where(steps.c.ticket == ticket)
    ).scalar_one()
    c.execute(
        steps.insert().values(
            ticket=ticket, n=n + 1, kind=kind, body=json.loads(json.dumps(body, default=str))
        )
    )
