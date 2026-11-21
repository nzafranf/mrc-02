"""OpenAI-compatible chat-completions client (requests only) with disk cache, retries and rate limit.

Used by `mrc.llm_qa --backend openrouter|gemini` so API models go through the same prompts,
parser, grounding and output format as the local LLM runs.

Keys are read from the environment only (OPENROUTER_API_KEY, GEMINI_API_KEY) and are never printed or logged.
Cache: experiments/06-llm-api-offtheshelf/cache/<model_slug>.jsonl, key = sha256(model + messages + params). Each line stores
the full raw JSON response, a timestamp and the model id the provider reported.

na_prob from logprobs (crude, only with --logprobs and only if the provider returns them):
  p1 = exp(logprob of the first generated token). If the output parses as no-answer, na_prob = p1,
  otherwise na_prob = 1 - p1. This is the confidence in the first token, not a calibrated probability.
Without logprobs na_prob is absent and the threshold cannot be tuned.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from mrc.paths import ROOT
CACHE_DIR = ROOT / "experiments/06-llm-api-offtheshelf/cache"

BACKENDS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "model": "google/gemma-4-31b-it:free"},
    # Google AI Studio OpenAI-compatible endpoint; requests go to <base>/chat/completions.
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
}


class ApiKeyMissing(RuntimeError):
    pass


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def cache_key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def mock_responder(payload: dict) -> dict:
    """Fake provider for tests: cycles through exact span, near-copy, no-answer and junk by prompt hash."""
    user = payload["messages"][-1]["content"]
    m = re.search(r"Konteks: (.*?)\n\nPertanyaan:", user, re.S)
    words = (m.group(1) if m else user).split()
    k = int(hashlib.sha256(user.encode()).hexdigest(), 16) % 4
    if k == 0:
        text = " ".join(words[3:6])  # exact span
    elif k == 1:
        text = "Jawaban: " + " ".join(words[3:5]).lower() + " sekitar"  # near-copy, needs snapping
    elif k == 2:
        text = "TIDAK ADA JAWABAN"
    else:
        text = "Maaf, saya tidak yakin tentang zxqv wvut."  # junk
    resp = {"id": "mock", "model": payload["model"] + "-mock-v0",
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    if payload.get("logprobs"):
        resp["choices"][0]["logprobs"] = {"content": [{"token": text.split()[0], "logprob": -0.3,
                                                       "top_logprobs": []}]}
    return resp


class ApiClient:
    def __init__(self, backend: str, model: str | None = None, base_url: str | None = None, rpm: float = 15,
                 mock: bool = False, max_retries: int = 6, timeout: float = 120, logprobs: bool = False,
                 max_tokens: int = 24, reasoning_effort: str | None = None, cache_dir: Path = CACHE_DIR):
        spec = BACKENDS[backend]
        self.backend, self.mock = backend, mock
        self.model = model or spec["model"]
        self.base_url = (base_url or spec["base_url"]).rstrip("/")
        self.rpm, self.max_retries, self.timeout = rpm, max_retries, timeout
        self.params: dict = {"temperature": 0, "max_tokens": max_tokens}
        if logprobs:
            self.params.update({"logprobs": True, "top_logprobs": 5})
        if reasoning_effort:
            self.params["reasoning_effort"] = reasoning_effort
        self._key = None
        if not mock:
            self._key = os.environ.get(spec["key_env"])
            if not self._key:
                raise ApiKeyMissing(f"{spec['key_env']} is not set. Export it in your shell (never put it in "
                                    f"a file in this repo), or use --mock for an offline test.")
        self.cache_path = cache_dir / f"{slug(self.model + ('-mock' if mock else ''))}.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r
        self.n_calls = self.n_cached = 0
        self._last_call = 0.0
        self.returned_models: set[str] = set()

    def _throttle(self) -> None:
        wait = self._last_call + 60.0 / self.rpm - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _post(self, payload: dict) -> dict:
        if self.mock:
            return mock_responder(payload)
        import requests
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            self._throttle()
            r = None
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                err = f"network error {type(e).__name__}"
            else:
                if r.status_code == 200:
                    return r.json()
                err = f"HTTP {r.status_code}"
                if r.status_code == 429 and re.search(r"per-day|PerDay|requests per day", r.text, re.I):
                    # daily quota (OpenRouter free-models-per-day, Google GenerateRequestsPerDay...): retrying cannot help
                    raise RuntimeError(f"daily quota exhausted, cached responses are kept; rerun later: {r.text[:200]}")
                if r.status_code != 429 and r.status_code < 500:
                    # the body may explain the problem; it never contains our key
                    raise RuntimeError(f"{err}: {r.text[:300]}")
            if attempt == self.max_retries:
                raise RuntimeError(f"giving up after {attempt + 1} attempts ({err})")
            delay = min(2.0 ** attempt * 2, 120)
            ra = r.headers.get("Retry-After") if r is not None else None
            if ra:
                try:
                    delay = max(delay, float(ra))
                except ValueError:
                    pass
            print(f"  {err}; retry {attempt + 1}/{self.max_retries} in {delay:.0f}s", file=sys.stderr, flush=True)
            time.sleep(delay)
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict]) -> dict:
        """Return the cached or fresh record: {key, ts, model_requested, model_returned, response}."""
        key = cache_key(self.model, messages, self.params)
        if key in self.cache:
            self.n_cached += 1
            rec = self.cache[key]
        else:
            payload = {"model": self.model, "messages": messages, **self.params}
            resp = self._post(payload)
            rec = {"key": key, "ts": dt.datetime.now().isoformat(timespec="seconds"),
                   "model_requested": self.model, "model_returned": resp.get("model"), "response": resp}
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.cache[key] = rec
            self.n_calls += 1
        if rec.get("model_returned"):
            self.returned_models.add(rec["model_returned"])
        return rec


def extract(rec: dict) -> tuple[str, float | None]:
    """(text, logprob of the first generated token or None)."""
    try:
        ch = rec["response"]["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "", None
    text = (ch.get("message") or {}).get("content") or ""
    lp = None
    lpd = ch.get("logprobs")
    content = lpd.get("content") if isinstance(lpd, dict) else None
    if content:
        lp = content[0].get("logprob")
    return text, lp


def crude_na_prob(first_logprob: float, is_noans: bool) -> float:
    """See module docstring: confidence in the first token, flipped when the output is an answer."""
    p1 = math.exp(first_logprob)
    return p1 if is_noans else 1.0 - p1
