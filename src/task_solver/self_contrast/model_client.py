"""Lightweight multi-provider LLM client using raw HTTP requests.

Supports OpenAI, Anthropic, Google Gemini, Ollama, vec-inf
(OpenAI-compatible), and vLLM (OpenAI-compatible) endpoints without any
framework dependency such as autogen or langchain.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests


log = logging.getLogger(__name__)


class LLMClient:
    """Multi-provider LLM client backed by ``requests.post``.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. ``"gpt-4o"``, ``"gemini-2.5-pro"``).
    base_url : Optional[str]
        Explicit endpoint URL.  When provided for local / vec-inf / vLLM
        models the client uses the OpenAI-compatible chat completions API.
    api_key : Optional[str]
        API key. Falls back to the appropriate environment variable when
        ``None``. OpenAI-compatible local endpoints also check
        ``LOCAL_API_KEY`` to match the rest of the codebase.
    temperature : float
        Sampling temperature sent with every request.
    timeout : Optional[int]
        HTTP request timeout in seconds. When omitted, local / custom
        OpenAI-compatible endpoints use a longer timeout than cloud APIs.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.provider = self._detect_provider()

        if not self.api_key:
            self.api_key = self._resolve_api_key()

        if not self.base_url:
            self.base_url = self._resolve_base_url()

        self.timeout = self._resolve_timeout(timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        force_json: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a single system+user message pair and return the response text.

        Parameters
        ----------
        system_prompt : str
            System message content.
        user_prompt : str
            User message content.
        force_json : bool
            Request JSON output via ``response_format`` when supported.
        temperature : Optional[float]
            Per-call override; uses instance default when ``None``.

        Returns
        -------
        str
            Model response text, or ``""`` on error.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        temp = temperature if temperature is not None else self.temperature
        return self._dispatch(messages, temp, force_json=force_json)

    # ------------------------------------------------------------------
    # Provider detection & defaults
    # ------------------------------------------------------------------

    def _detect_provider(self) -> str:  # noqa: PLR0911
        lowered = self.model_name.lower()
        if "gpt" in lowered or "o1-" in lowered or "o3-" in lowered or "o4-" in lowered:
            return "openai"
        if "claude" in lowered or "anthropic" in lowered:
            return "anthropic"
        if "gemini" in lowered or "google" in lowered:
            return "google"

        base = (self.base_url or "").lower()
        if "localhost" in base or "11434" in base:
            return "ollama"

        if self.base_url:
            return "openai_compatible"

        local_keywords = ["llama", "mistral", "qwen", "phi", "deepseek", "gemma"]
        if any(kw in lowered for kw in local_keywords):
            return "ollama"

        return "openai"

    def _resolve_api_key(self) -> Optional[str]:
        env_map = {
            "openai": ("OPENAI_API_KEY",),
            "anthropic": ("ANTHROPIC_API_KEY",),
            "google": ("GOOGLE_API_KEY",),
            "openai_compatible": ("LOCAL_API_KEY", "OPENAI_API_KEY"),
            "ollama": ("LOCAL_API_KEY",),
        }
        for var in env_map.get(self.provider, ()):
            value = (os.getenv(var) or "").strip()
            if value:
                return value
        return None

    def _resolve_base_url(self) -> str:
        defaults = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "google": "https://generativelanguage.googleapis.com/v1beta",
            "ollama": "http://localhost:11434/v1",
        }
        return defaults.get(self.provider, "https://api.openai.com/v1")

    def _resolve_timeout(self, configured_timeout: Optional[int]) -> int:
        if configured_timeout is not None:
            return configured_timeout
        if self.provider == "ollama":
            # Local / vec-inf backends can take much longer than cloud APIs.
            return 600
        return 120

    # ------------------------------------------------------------------
    # Dispatch to provider-specific request builders
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        *,
        force_json: bool = False,
    ) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(messages, temperature)
        if self.provider == "google":
            return self._call_google(messages, temperature)
        return self._call_openai_compatible(
            messages, temperature, force_json=force_json
        )

    # ------------------------------------------------------------------
    # OpenAI / Ollama / vec-inf / vLLM (all OpenAI-compatible)
    # ------------------------------------------------------------------

    def _call_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        *,
        force_json: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if force_json and self._supports_response_format():
            payload["response_format"] = {"type": "json_object"}
        if self._should_disable_thinking():
            # Qwen3 enables chain-of-thought <think> blocks by default; on local
            # inference servers that can make responses extremely slow. This key
            # is supported by vLLM >= 0.8 and SGLang. Older servers ignore it.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/chat/completions"
        return self._post(endpoint, payload, headers, extractor=self._extract_openai)

    def _supports_response_format(self) -> bool:
        lowered = self.model_name.lower()
        if lowered.startswith(("gpt-", "o1-", "o3-", "o4-", "gpt-5")):
            return True
        return self.provider in ("ollama", "openai_compatible")

    def _should_disable_thinking(self) -> bool:
        """Disable Qwen3 thinking mode on local/vec-inf endpoints.

        Qwen3 models have thinking enabled by default, generating a long
        ``<think>`` block before producing visible output.  This can make
        code-generation prompts extremely slow on local inference servers.
        """
        if self.provider not in ("ollama", "openai_compatible"):
            return False
        return "qwen" in self.model_name.lower()

    @staticmethod
    def _extract_openai(data: Dict[str, Any]) -> str:
        return data["choices"][0]["message"]["content"].strip()

    # --- Anthropic (Claude) ---

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        system_text = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system" and system_text is None:
                system_text = msg["content"]
            else:
                filtered.append(msg)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": filtered,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if system_text:
            payload["system"] = system_text

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        endpoint = f"{self.base_url}/messages"
        return self._post(endpoint, payload, headers, extractor=self._extract_anthropic)

    @staticmethod
    def _extract_anthropic(data: Dict[str, Any]) -> str:
        return data["content"][0]["text"].strip()

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------

    def _call_google(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        if not self.api_key:
            log.error("GOOGLE_API_KEY is required for Gemini API.")
            return ""

        system_text = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system" and system_text is None:
                system_text = msg["content"]
            else:
                filtered.append(msg)

        contents = []
        for msg in filtered:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        endpoint = f"{base}/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"content-type": "application/json"}
        return self._post(endpoint, payload, headers, extractor=self._extract_google)

    @staticmethod
    def _extract_google(data: Dict[str, Any]) -> str:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    # ------------------------------------------------------------------
    # Shared HTTP helper
    # ------------------------------------------------------------------

    def _post(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        *,
        extractor: Any,
    ) -> str:
        t0 = time.monotonic()
        log.info(
            "HTTP start %s %s -> %s",
            self.provider,
            payload.get("model", "?"),
            endpoint,
        )
        try:
            resp = requests.post(
                endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            elapsed = time.monotonic() - t0
            log.info(
                "HTTP %s %s → %s (%.1fs)",
                self.provider,
                payload.get("model", "?"),
                resp.status_code,
                elapsed,
            )
            if resp.status_code >= 400:
                log.error(
                    "HTTP %s response body: %s",
                    resp.status_code,
                    resp.text[:500],
                )
            resp.raise_for_status()
            return extractor(resp.json())
        except Exception as exc:
            elapsed = time.monotonic() - t0
            log.error(
                "LLM call failed (%s, timeout=%ss, elapsed=%.1fs): %s",
                self.provider,
                self.timeout,
                elapsed,
                exc,
            )
            return ""
