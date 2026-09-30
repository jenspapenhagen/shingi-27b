"""The README's curl examples must be valid requests that the server answers."""
import base64
import math
import re
from pathlib import Path

from fastapi.testclient import TestClient

from shingi.decision import DecisionEngine
from shingi.schema import Request
from shingi.server import create_app

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
BODIES = re.findall(r"-d '(\{.*\})'", README)


class Uniform:
    def infer(self, prompt, labels):
        return {"logits": [0.0] * len(labels), "input_tokens": 10}


def test_readme_examples_are_valid_requests():
    assert len(BODIES) == 2
    kinds = set()
    with TestClient(create_app(DecisionEngine(Uniform()))) as client:
        for body in BODIES:
            request = Request.model_validate_json(body)
            kinds |= {q.type for q in request.questions.values()}
            response = client.post("/v1/systemone", content=body, headers={"Content-Type": "application/json"})
            assert response.status_code == 200, response.text
            assert response.json()["model"] == "shingi-27b"
    assert kinds == {"choice", "noul"}


HEREDOCS = re.findall(r"-d @- <<JSON\n(.*?)\nJSON", README, re.S)


class Vision(Uniform):
    vision = True

    def infer(self, prompt, labels, images=None):
        return super().infer(prompt, labels)


def test_readme_image_and_decisions_examples_are_answered():
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
    assert len(HEREDOCS) == 2
    with TestClient(create_app(DecisionEngine(Vision()))) as client:
        image = client.post("/v1/systemone", content=HEREDOCS[0].replace("$IMAGE", png),
                            headers={"Content-Type": "application/json"})
        decisions = client.post("/v1/decisions", content=HEREDOCS[1], headers={"Content-Type": "application/json"})
    assert image.status_code == 200, image.text
    assert image.json()["usage"]["images"] == 1
    assert decisions.status_code == 200, decisions.text
    assert set(decisions.json()["answers"]) == {"urgent", "team"}


def test_readme_choice_example_returns_every_option():
    with TestClient(create_app(DecisionEngine(Uniform()))) as client:
        answer = client.post("/v1/systemone", content=BODIES[0],
                             headers={"Content-Type": "application/json"}).json()["answers"]["route"]
    assert set(answer["probabilities"]) == {"returns", "billing", "shipping"}
    assert math.isclose(sum(answer["probabilities"].values()), 1)
