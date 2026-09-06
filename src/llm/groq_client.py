"""
groq_client.py
--------------
Thin wrapper around the Groq API for the CreditRisk AI Copilot.

Architectural principle (see README "AI Copilot" section):
    DATA -> ANALYTICS/ML -> VERIFIED METRICS -> LLM EXPLANATION
The LLM never computes PD, risk scores, or model metrics itself — it
only explains numbers that were already computed by src/modeling.py,
src/evaluation.py, etc. and passed in as context.

Configuration (never hard-code the key):
    GROQ_API_KEY   - required, read from environment / Streamlit secrets
    GROQ_MODEL     - optional, defaults to a currently-supported Groq
                     chat model; override via env var if Groq deprecates it
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are the CreditRisk AI Copilot, an assistant embedded inside a \
consumer credit-risk analytics application (a portfolio project, not a production \
lending system).

You help users interpret VERIFIED analytical outputs that are supplied to you as \
context in each request (portfolio statistics, model metrics, risk-segment tables, \
feature importances, monitoring results, customer-level model outputs). You must:

- Never fabricate a metric, model result, or customer fact that is not present in \
the supplied context. If something is not available, say so explicitly.
- Never independently calculate or re-derive credit risk yourself — you explain \
numbers the ML pipeline already produced, you do not produce new ones.
- Never claim a customer "will" default — only that the model estimates a given \
probability of default and assigns a given risk band.
- Never provide an actual lending decision (approve/reject/extend credit). If asked, \
explain that this application explains model outputs and risk indicators; it does \
not make lending decisions, then explain what the model output does show.
- Clearly distinguish statistical association from causation.
- Clearly distinguish "what the model output says" from "your interpretation of it."
- Explain technical concepts (PD, ROC-AUC, KS, PSI, calibration, SHAP, etc.) in \
plain business language when asked.
- Mention relevant limitations when they materially affect the answer (e.g., this \
model is trained on a single historical snapshot of Taiwanese credit-card accounts \
from 2005 and has no macroeconomic inputs).
- This is a portfolio/educational project. Say so if asked whether this is a real \
bank system.
"""


@dataclass
class GroqClientError(Exception):
    message: str


@dataclass
class CreditRiskCopilot:
    model: str = DEFAULT_MODEL
    api_key: str | None = field(default=None)
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("GROQ_API_KEY")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from groq import Groq
        except ImportError as e:
            raise GroqClientError("The 'groq' package is not installed. Run: pip install groq") from e
        if not self.api_key:
            raise GroqClientError(
                "GROQ_API_KEY is not set. Set it as an environment variable or in "
                "Streamlit secrets (see .env.example)."
            )
        self._client = Groq(api_key=self.api_key)
        return self._client

    def chat(self, user_message: str, context: dict, history: list[dict] | None = None) -> dict:
        """
        Returns a dict: {"success": bool, "content": str, "latency_s": float, "error": str|None}
        Never raises — callers (Streamlit UI) should always get a renderable result,
        because the core dashboard must keep working even if the LLM is unavailable.
        """
        start = time.time()
        try:
            client = self._get_client()
        except GroqClientError as e:
            return {"success": False, "content": None, "latency_s": 0.0, "error": str(e)}

        context_block = _format_context(context)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-10:])  # keep a bounded window of recent turns
        messages.append({
            "role": "user",
            "content": f"[VERIFIED PROJECT CONTEXT]\n{context_block}\n\n[USER QUESTION]\n{user_message}",
        })

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=700,
            )
            content = response.choices[0].message.content
            return {"success": True, "content": content, "latency_s": time.time() - start, "error": None}
        except Exception as e:  # noqa: BLE001 - surface any API/network/rate-limit error uniformly
            return {"success": False, "content": None, "latency_s": time.time() - start, "error": str(e)}


def _format_context(context: dict) -> str:
    import json

    return json.dumps(context, indent=2, default=str)
