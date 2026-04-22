"""HTTP client for the local Ollama server."""
from __future__ import annotations

import httpx


class OllamaClient:
    """Thin wrapper around the Ollama ``/api/generate`` endpoint.

    Args:
        base_url:    Base URL of the Ollama server (e.g. ``http://127.0.0.1:11434``).
        model:       Model name (e.g. ``llama3:8b``).
        temperature: Sampling temperature; lower values give more deterministic
                     output.  Defaults to ``0.2``.
    """

    def __init__(self, base_url: str, model: str, temperature: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Ollama and return the model's text response.

        Raises:
            httpx.HTTPStatusError: When the server returns a non-2xx status.
            httpx.TimeoutException: When the request times out (300 s limit).
        """
        url = f"{self.base_url}/api/generate"
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": 42,
            },
        }
        with httpx.Client(timeout=300.0, trust_env=False) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return str(response.json()["response"])

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* using the Ollama embeddings API.

        Uses ``/api/embeddings`` which is supported by all Ollama versions.

        Raises:
            httpx.HTTPStatusError: When the server returns a non-2xx status.
            httpx.TimeoutException: When the request times out.
            KeyError: When the response does not contain an ``embedding`` field.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return list(response.json()["embedding"])

    def health_check(self) -> bool:
        """Return *True* when the Ollama server is reachable."""
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
