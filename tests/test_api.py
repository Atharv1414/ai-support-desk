import os

import pytest
from fastapi.testclient import TestClient

from desk import store
from desk.api import create_app


class FixedTriage:
    name = "fixed"

    def __init__(self):
        self.answer = ("help", 0.9)

    def __call__(self, text):
        return self.answer


@pytest.fixture
def api(tmp_path):
    url = os.getenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    triage = FixedTriage()
    app = create_app(url, live=False, triage=triage)
    with TestClient(app) as c:
        store.seed(app.state.engine)
        c.triage = triage
        yield c


def post(api, body, email="ada@example.com"):
    r = api.post("/api/tickets", json={"email": email, "body": body})
    assert r.status_code == 201
    return r.json()["id"]


def test_message_to_reply(api):
    t = post(api, "What's the best way to descale the kettle")
    r = api.post(f"/api/tickets/{t}/run").json()
    assert r["status"] == "answered" and r["steps"][0]["kind"] == "triage"
    assert "kettle-care#descaling" in r["steps"][-1]["text"]


def test_unsure_triage_goes_to_a_person(api):
    api.triage.answer = ("help", 0.2)
    t = post(api, "hmm")
    r = api.post(f"/api/tickets/{t}/run").json()
    assert r["status"] == "with_person" and "not sure" in r["steps"][-1]["reason"]


def test_approval_flow_and_second_answer_is_refused(api):
    api.triage.answer = ("refund", 0.9)
    t = post(api, "I returned HF-21090, where is my money", "cal@example.com")
    r = api.post(f"/api/tickets/{t}/run").json()
    a = r["approval"]["id"]
    assert r["status"] == "waiting" and r["approval"]["args"]["amount"] == 64
    done = api.post(f"/api/approvals/{a}", json={"approve": True, "person": "Priya"})
    assert done.json()["status"] == "answered"
    again = api.post(f"/api/approvals/{a}", json={"approve": True, "person": "Priya"})
    assert again.status_code == 409


@pytest.mark.parametrize(
    "bad",
    [
        {"email": "nope", "body": "hello there"},
        {"email": "a@b.co", "body": "x"},
        {"email": "a@b.co", "body": "x" * 2001},
    ],
)
def test_bad_messages_are_rejected(api, bad):
    assert api.post("/api/tickets", json=bad).status_code == 422


def test_missing_things_are_404(api):
    assert api.get("/api/tickets/999").status_code == 404
    assert (
        api.post("/api/approvals/999", json={"approve": True, "person": "Priya"}).status_code == 404
    )


def test_inbox_is_newest_first(api):
    a, b = post(api, "first one"), post(api, "second one")
    assert [t["id"] for t in api.get("/api/tickets").json()][:2] == [b, a]


def test_models_endpoint(api):
    r = api.get("/api/models").json()
    assert r["stages"]["champion"] and r["models"]
