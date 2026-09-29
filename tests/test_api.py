import math
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul
import uvicorn

from shingi.decision import Calibration, DecisionEngine, MODEL_ID, choice_confidence, score_confidence
from shingi.model import CALIBRATION_SHA256, MODEL_SHA256, RUNTIME
from shingi.server import create_app


class StubReadout:
    def __init__(self, probabilities=None):
        self.probabilities = probabilities
        self.prompts = []

    def infer(self, prompt, labels):
        self.prompts.append(prompt)
        p = self.probabilities or [1 / len(labels)] * len(labels)
        assert len(p) == len(labels)
        return {"logits": [math.log(x) for x in p], "input_tokens": 123,
                "candidate_ids": list(range(len(labels))), "prefill_ms": 1}


def request(question, state="sample"):
    return {"model": "jev-latest", "state": state, "questions": {"q": question}}


def test_choice_and_null_descriptions_survive_validation():
    backend = StubReadout([0.2, 0.8])
    with TestClient(create_app(DecisionEngine(backend))) as client:
        result = client.post("/v1/systemone", json=request({"type": "choice", "instructions": "pick",
                                                            "criteria": {"a": None, "b": None}}))
    assert result.status_code == 200
    answer = result.json()["answers"]["q"]
    assert answer["choice"] == "b"
    assert answer["probabilities"] == pytest.approx({"a": .2, "b": .8})
    assert answer["confidence"] == pytest.approx(.6)
    assert "[A] a: " in backend.prompts[0]


def test_score_expected_value_structured_legend_and_confidence():
    engine = DecisionEngine(StubReadout([.1, .7, .2]))
    levels = ["low", {"severity": "medium"}, ["high"]]
    with TestClient(create_app(engine)) as client:
        answer = client.post("/v1/systemone", json=request({"type": "score", "instructions": ["rate"],
                                                             "criteria": levels})).json()["answers"]["q"]
    assert answer["score"] == pytest.approx(1.1)
    assert answer["legend"] == dict(zip(["0", "1", "2"], levels))
    assert answer["confidence"] == pytest.approx(.55)


def test_noul_uses_yes_probability_and_does_not_invent_confidence():
    engine = DecisionEngine(StubReadout([.8, .2]), Calibration(noul_temperature=2))
    with TestClient(create_app(engine)) as client:
        answer = client.post("/v1/systemone", json=request({"type": "noul", "instructions": "yes?"})).json()["answers"]["q"]
    assert answer == pytest.approx({"noul": 2 / 3, "type": "noul"})


@pytest.mark.parametrize("question", [
    {"type": "choice", "instructions": "pick", "criteria": {}},
    {"type": "choice", "instructions": "pick", "criteria": {str(i): None for i in range(256)}},
    {"type": "score", "instructions": "rate", "criteria": ["one"]},
    {"type": "score", "instructions": "rate", "criteria": [str(i) for i in range(11)]},
    {"type": "noul", "criteria": {}},
    {"type": "noul", "instructions": "yes?", "criteria": {"maybe": "yes"}},
    {"type": "noul", "instructions": 42},
    {"type": "bogus", "instructions": "test"},
])
def test_invalid_requests_fail_before_inference(question):
    backend = StubReadout()
    with TestClient(create_app(DecisionEngine(backend))) as client:
        response = client.post("/v1/systemone", json=request(question))
    assert response.status_code == 422
    assert backend.prompts == []


@pytest.mark.parametrize("count", [1, 52, 53, 255])
def test_all_candidates_retained_across_chunk_boundary(count):
    backend = StubReadout()
    with TestClient(create_app(DecisionEngine(backend))) as client:
        response = client.post("/v1/systemone", json=request({"type": "choice", "instructions": "pick",
                                                             "criteria": {str(i): None for i in range(count)}}))
    assert response.status_code == 200
    answer = response.json()["answers"]["q"]
    assert len(answer["probabilities"]) == count
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    assert all(x >= 0 and math.isfinite(x) for x in answer["probabilities"].values())
    assert len(backend.prompts) == (1 if count <= 52 else math.ceil(count / 52) + 1)


def test_confidence_semantics_differ_from_probability():
    assert choice_confidence([.25] * 4) == pytest.approx(0)
    assert choice_confidence([.8, .1, .1]) == pytest.approx(.7)
    assert score_confidence([0, 1, 0]) == 1
    assert score_confidence([.5, 0, .5]) == 0


def test_question_ids_do_not_reach_model_or_affect_other_questions():
    backend = StubReadout()
    with TestClient(create_app(DecisionEngine(backend))) as client:
        response = client.post("/v1/systemone", json={"model": "shingi", "state": {"message": "x"},
            "questions": {"SECRET_QUESTION_ID": {"type": "noul", "instructions": "true?"},
                          "SECOND_ID": {"type": "noul", "instructions": "false?"}}})
    assert response.status_code == 200
    assert len(backend.prompts) == 2
    assert all("SECRET_QUESTION_ID" not in p and "SECOND_ID" not in p for p in backend.prompts)
    assert "false?" not in backend.prompts[0]
    assert "true?" not in backend.prompts[1]


def test_explicit_image_input_is_rejected_not_silently_treated_as_text():
    with TestClient(create_app(DecisionEngine(StubReadout()))) as client:
        result = client.post("/v1/systemone", json=request({"type": "noul", "instructions": "yes?"},
                                                           state={"screenshot": "data:image/png;base64,abc"}))
    assert result.status_code == 422


def test_incomplete_logits_are_backend_failure_not_fabricated_probabilities():
    class Broken:
        def infer(self, prompt, labels):
            return {"logits": [1.0], "input_tokens": 12}
    with TestClient(create_app(DecisionEngine(Broken()))) as client:
        result = client.post("/v1/systemone", json=request({"type": "noul", "instructions": "yes?"}))
    assert result.status_code == 503


def test_canonical_choice_keys_make_map_reordering_identical():
    backend = StubReadout([.8, .2])
    engine = DecisionEngine(backend)
    answers = [engine.answer("x", {"type": "choice", "instructions": "pick", "criteria": criteria})[0]
               for criteria in ({"zebra": None, "apple": None}, {"apple": None, "zebra": None})]
    assert answers[0] == answers[1]
    assert backend.prompts[0] == backend.prompts[1]
    assert answers[0]["choice"] == "apple"


def test_canonical_choice_mode_preserves_ordinal_score_semantics():
    backend = StubReadout([.8, .2])
    engine = DecisionEngine(backend)
    engine.answer("x", {"type": "score", "instructions": "rate", "criteria": ["z low", "a high"]})
    assert backend.prompts[0].index("z low") < backend.prompts[0].index("a high")


def test_version_reports_identity_calibration_and_runtime():
    identity = {"model": MODEL_ID, "model_sha256": MODEL_SHA256, "verified": True}
    app = create_app(DecisionEngine(StubReadout(), Calibration(noul_temperature=1.25)),
                     identity=identity, calibration_sha256=CALIBRATION_SHA256)
    with TestClient(app) as client:
        version = client.get("/v1/version").json()
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/v1/systemone", json={**request({"type": "noul", "instructions": "x"}),
                                                  "model": "other"}).status_code == 422
    assert version["model"] == "shingi-27b"
    assert version["model_sha256"] == MODEL_SHA256
    assert version["calibration"] == {"temperature": 1.0, "noul_temperature": 1.25, "noul_bias": 0.0}
    assert version["calibration_sha256"] == CALIBRATION_SHA256
    assert version["runtime"] == RUNTIME
    assert version["context_tokens"] == 16384 and version["kv_cache"] == "q8_0"


def test_real_typesafe_sdk_over_http():
    # Exercise the installed SDK's serialization, route and response parsing.
    # A deterministic backend isolates protocol correctness from model quality.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(create_app(DecisionEngine(StubReadout())), log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        with TypeSafeClient(api_key="local-test", base_url=f"http://127.0.0.1:{sock.getsockname()[1]}") as client:
            assert client.models.list().models[0].name == MODEL_ID
            result = client.system_one(state={"text": "hello"}, questions={
                "route": Choice(instructions={"question": "route?"}, criteria={"x": None, "y": ["other"]}),
                "yes": Noul(instructions="yes?"),
                "rating": Score(instructions="rate", criteria=["low", {"level": "high"}]),
            })
        assert result.model == MODEL_ID
        assert result.choices["route"].probabilities == {"x": .5, "y": .5}
        assert result.nouls["yes"].noul == .5
        assert result.scores["rating"].score == .5
        assert result.scores["rating"].legend[1] == {"level": "high"}
        assert result.usage.input_tokens == 369
        assert result.usage.output_tokens == 0
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
