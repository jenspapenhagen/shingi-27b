"""Typed readout math; independent of the inference transport."""
from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass
from typing import Protocol

LETTERS = string.ascii_uppercase + string.ascii_lowercase
MODEL_ID = "shingi-27b"


def describe(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def prompt_for(state, instructions, options):
    lines = "\n".join(f"[{LETTERS[i]}] {key}: {describe(value)}" for i, (key, value) in enumerate(options))
    text = f"State:\n{describe(state)}\n\nQuestion: {describe(instructions)}\nOptions:\n{lines}\n\nAnswer with the letter of the best option only."
    # Verified against Prism /apply-template with enable_thinking=False.
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def softmax(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("all candidate scores must be finite")
    peak = max(values)
    mass = [math.exp(x - peak) for x in values]
    return [x / sum(mass) for x in mass]


def choice_confidence(p):
    return 1.0 if len(p) == 1 else max(0.0, (len(p) * max(p) - 1) / (len(p) - 1))


def score_confidence(p):
    mode = max(range(len(p)), key=p.__getitem__)
    deviation = sum(value * abs(i - mode) for i, value in enumerate(p))
    uniform = sum(abs(i - (len(p) - 1) / 2) for i in range(len(p))) / len(p)
    return max(0.0, 1 - deviation / uniform)


@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    noul_temperature: float = 1.0
    noul_bias: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.temperature, self.noul_temperature, self.noul_bias)):
            raise ValueError("calibration must be finite")
        if self.temperature <= 0 or self.noul_temperature <= 0:
            raise ValueError("temperatures must be positive")


class Readout(Protocol):
    def infer(self, prompt: str, labels: list[str]) -> dict: ...


class DecisionEngine:
    def __init__(self, backend: Readout, calibration=Calibration(), *, model_id=MODEL_ID):
        self.backend = backend
        self.calibration = calibration
        self.model_id = model_id

    def distribution(self, state, instructions, options):
        if len(options) <= len(LETTERS):
            result = self.backend.infer(prompt_for(state, instructions, options), list(LETTERS[:len(options)]))
            logits = result["logits"]
            if len(logits) != len(options) or any(not math.isfinite(x) for x in logits):
                raise RuntimeError("incomplete or non-finite candidate logits")
            p = softmax([x / self.calibration.temperature for x in logits])
            return p, [result]
        # OpenJev-style approximate anchor composition. All labels retain mass;
        # this is not a single forward pass or an exact global softmax.
        n_chunks = math.ceil(len(options) / len(LETTERS))
        chunk_size = math.ceil(len(options) / n_chunks)
        chunks = [options[i:i + chunk_size] for i in range(0, len(options), chunk_size)]
        parts = [self.distribution(state, instructions, c) for c in chunks]
        winners = [max(range(len(p)), key=p.__getitem__) for p, _ in parts]
        final, traces = self.distribution(state, instructions, [c[w] for c, w in zip(chunks, winners)])
        values = [final[j] * value / p[w] for j, ((p, _), w) in enumerate(zip(parts, winners)) for value in p]
        total = sum(values)
        return [x / total for x in values], [trace for _, batch in parts for trace in batch] + traces

    def answer(self, state, q):
        instructions = q["instructions"]
        kind = q["type"]
        if kind == "choice":
            # Canonical order: reordering the same JSON map gives an identical prompt.
            options = sorted(q["criteria"].items(), key=lambda item: item[0])
        elif kind == "score":
            options = [(str(i), value) for i, value in enumerate(q["criteria"])]
            instructions = describe(instructions) + " Rate along the ordered levels below (lowest first)."
        else:
            criteria = q.get("criteria") or {}
            options = [("yes", criteria.get("true") or "The statement is true."),
                       ("no", criteria.get("false") or "The statement is false.")]
        p, traces = self.distribution(state, instructions, options)
        if kind == "noul":
            # Work from logits, not clipped probabilities, to retain the tails.
            logits = traces[0]["logits"]
            log_odds = (logits[0] - logits[1]) / self.calibration.temperature / self.calibration.noul_temperature + self.calibration.noul_bias
            yes = softmax([log_odds, 0.0])[0]
            answer = {"type": kind, "noul": yes}
        else:
            probabilities = {key: value for (key, _), value in zip(options, p)}
            if kind == "choice":
                winner = max(range(len(p)), key=p.__getitem__)
                answer = {"type": kind, "choice": options[winner][0], "probabilities": probabilities,
                          "confidence": choice_confidence(p)}
            else:
                answer = {"type": kind, "score": sum(i * value for i, value in enumerate(p)),
                          "legend": dict(options), "probabilities": probabilities, "confidence": score_confidence(p)}
        return answer, traces

    def evaluate(self, request):
        answers, traces = {}, {}
        for key, question in request["questions"].items():
            answers[key], traces[key] = self.answer(request["state"], question)
        calls = [x for group in traces.values() for x in group]
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": sum(x["input_tokens"] for x in calls), "output_tokens": 0}}, traces
