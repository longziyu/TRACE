"""Bounded asynchronous explanation with deterministic claims and safe templates.

Only validated claim slots are rendered by the checked backend. Arbitrary generated
prose is intentionally not trusted by the deterministic verifier.
"""
from dataclasses import dataclass
import json
import queue
import threading
import time
import numpy as np


def expected_claims(record):
    k, j = record.selected, record.comparator
    reason = ("no_admissible" if record.no_admissible else
              "risk_screening" if not record.admissible[j] else "weighted_cost")
    return {"selected": k, "comparator": j, "reason": reason,
            "selected_risk": round(record.risk[k], 6),
            "comparator_risk": round(record.risk[j], 6),
            "contributions": [round(x, 6) for x in record.contributions],
            "uncertainty": "predictions_are_uncertain"}


def verify(claims, record):
    expected = expected_claims(record)
    if not isinstance(claims, dict) or set(claims) != set(expected):
        return False
    for key, value in expected.items():
        if key in {"selected_risk", "comparator_risk", "contributions"}:
            try:
                a, b = np.asarray(claims[key], dtype=float), np.asarray(value)
                if a.shape != b.shape or not np.isfinite(a).all() or not np.allclose(a, b, atol=1e-5, rtol=0):
                    return False
            except (TypeError, ValueError):
                return False
        elif claims[key] != value:
            return False
    return True


def render(claims, record):
    selected, other = claims["selected"], claims["comparator"]
    prefix = f"Selected {record.behavior} (candidate {selected}) compared with candidate {other}. "
    if claims["reason"] == "no_admissible":
        reason = "No candidate passed risk screening; the controller prioritized the lowest screening score. "
    elif claims["reason"] == "risk_screening":
        reason = "The comparison candidate failed risk screening. "
    else:
        reason = f"The weighted cost contributions favoring the selection are {claims['contributions']}. "
    return prefix + reason + "These are predicted consequences, not a guarantee of collision-free motion."


class TemplateBackend:
    def __call__(self, record):
        return expected_claims(record)


class HFClaimsBackend:
    """Load a locally downloaded instruction model; never downloads implicitly."""
    def __init__(self, model_path, device="cpu", adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.device = device
        self.torch = torch

    def __call__(self, record):
        prompt = ("Return ONLY a JSON object with exactly these fields: selected, comparator, reason, "
                  "selected_risk, comparator_risk, contributions, uncertainty. "
                  "reason is no_admissible, risk_screening, or weighted_cost. "
                  "uncertainty must be predictions_are_uncertain. No causal claims. Record: "
                  + json.dumps(record.dictionary()))
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**tokens, max_new_tokens=256, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return json.loads(self.tokenizer.decode(output[0, tokens.input_ids.shape[1]:], skip_special_tokens=True))


@dataclass
class ExplanationResult:
    record: object
    text: str
    fallback: bool
    submitted_at: float
    completed_at: float
    error: str | None = None


class AsyncExplainer:
    def __init__(self, backend=None):
        self.backend = backend or TemplateBackend()
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.canceled = 0
        self.submitted = 0
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def submit(self, record):
        self.submitted += 1
        try:
            self.requests.put_nowait((record, time.monotonic()))
        except queue.Full:
            try:
                self.requests.get_nowait()
                self.canceled += 1
            except queue.Empty:
                pass
            self.requests.put_nowait((record, time.monotonic()))

    def _work(self):
        while not self.stop_event.is_set():
            try:
                record, started = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            fallback, error = False, None
            try:
                claims = self.backend(record)
                if not verify(claims, record):
                    raise ValueError("Generated claims do not match the decision record")
            except Exception as exc:
                claims, fallback, error = expected_claims(record), True, str(exc)
            result = ExplanationResult(record, render(claims, record), fallback, started, time.monotonic(), error)
            try:
                self.results.put_nowait(result)
            except queue.Full:
                try:
                    self.results.get_nowait()
                    self.canceled += 1
                except queue.Empty:
                    pass
                self.results.put_nowait(result)

    def poll(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=1)


def is_current(result, current, controller, previous, timestamp):
    old = result.record
    if timestamp-old.timestamp > controller.cfg.explanation_max_age or timestamp < old.timestamp:
        return False
    if current.behavior != old.behavior or current.no_admissible != old.no_admissible:
        return False
    ids = [old.selected, old.comparator]
    reevaluated = controller.reevaluate([old.candidates[i] for i in ids], previous, timestamp)
    if reevaluated is None:
        return False
    admissible, costs = reevaluated
    if list(admissible) != [old.admissible[i] for i in ids]:
        return False
    old_delta = old.costs[old.comparator]-old.costs[old.selected]
    if np.sign(costs[1]-costs[0]) != np.sign(old_delta):
        return False
    return np.allclose(np.asarray(old.candidates[old.selected])[0],
                       np.asarray(current.candidates[current.selected])[0],
                       atol=controller.cfg.explanation_action_tolerance, rtol=0)
