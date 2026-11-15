import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from typing import List, Optional

from app.config import Config

logger = logging.getLogger(__name__)

# Fallback models in priority order for resilience against timeouts (read timeout) and 503/429 spikes
DEFAULT_FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
]


class GeminiClient:
    """
    HTTP client for Google Gemini API.
    Communicates directly with the Gemini REST API with zero external SDK bloat,
    featuring temperature tuning, exponential backoff retry on timeouts and HTTP 503/429/504,
    and automatic fallback across resilient Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = Config.LLM_TEMPERATURE,
        max_output_tokens: int = Config.LLM_MAX_OUTPUT_TOKENS,
        timeout: Optional[int] = None,
    ):
        self.api_key = Config.GEMINI_API_KEY if api_key is None else api_key
        self.model_name = model_name if model_name is not None else Config.GEMINI_MODEL
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout if timeout is not None else getattr(Config, "LLM_REQUEST_TIMEOUT", 60)

    @property
    def active_model(self) -> str:
        """Returns the primary configured model name."""
        return self.model_name or os.getenv("GEMINI_MODEL") or Config.GEMINI_MODEL

    @property
    def endpoint_url(self) -> str:
        """Dynamically evaluate endpoint URL for the active model."""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.active_model}:generateContent"

    def is_configured(self) -> bool:
        """Checks if a valid API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def generate(self, system_instruction: str, prompt: str) -> str:
        """
        Sends system instruction and user prompt to Google Gemini and returns the generated text.
        Automatically retries on timeouts and HTTP 503/429/504 and falls back to
        alternative models if latency spikes or demand spikes persist.

        :param system_instruction: System prompt enforcing grounding rules.
        :param prompt: User prompt containing retrieved context and question.
        :return: Generated answer string.
        """
        if not self.is_configured():
            raise ValueError(
                "GEMINI_API_KEY is not configured in your .env file. "
                "Please obtain an API key from https://aistudio.google.com/app/apikey "
                "and set GEMINI_API_KEY=<your_key> in .env."
            )

        # Build candidate models starting with active_model, followed by fallbacks without duplicates
        primary = self.active_model
        candidate_models: List[str] = [primary]
        for m in DEFAULT_FALLBACK_MODELS:
            if m not in candidate_models:
                candidate_models.append(m)

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "topP": 0.95,
            }
        }
        body_bytes = json.dumps(payload).encode("utf-8")

        last_error: Optional[Exception] = None

        for model in candidate_models:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            max_retries = 2

            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        f"Calling Google Gemini API (model: {model}, attempt {attempt}/{max_retries}, timeout: {self.timeout}s)..."
                    )
                    req = urllib.request.Request(
                        endpoint,
                        data=body_bytes,
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )

                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        status_code = response.getcode()
                        response_data = json.loads(response.read().decode("utf-8"))

                        if status_code != 200:
                            raise RuntimeError(f"Gemini API returned unexpected status {status_code}: {response_data}")

                        candidates = response_data.get("candidates", [])
                        if not candidates:
                            block_reason = response_data.get("promptFeedback", {}).get("blockReason")
                            if block_reason:
                                raise RuntimeError(f"Gemini request blocked by safety filter: {block_reason}")
                            return "No response generated by model."

                        candidate = candidates[0]
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        if parts and "text" in parts[0]:
                            answer_text = parts[0]["text"].strip()
                            if model != primary:
                                logger.info(f"Successfully generated answer using fallback model '{model}'.")
                            else:
                                logger.info(f"Successfully received answer from Google Gemini ({model}).")
                            return answer_text

                        return "Unable to parse generated text from Gemini response."

                except (socket.timeout, TimeoutError) as te:
                    logger.warning(f"Timeout on model '{model}' (attempt {attempt}/{max_retries}): {te}")
                    last_error = TimeoutError(f"Request to Gemini model '{model}' timed out after {self.timeout}s")
                    if attempt < max_retries:
                        time.sleep(1.0)
                        continue
                    else:
                        logger.warning(f"Model '{model}' timed out repeatedly. Falling back to alternative model...")
                        break

                except urllib.error.HTTPError as e:
                    error_body = e.read().decode("utf-8") if e.fp else str(e)
                    try:
                        err_json = json.loads(error_body)
                        err_msg = err_json.get("error", {}).get("message", error_body)
                    except Exception:
                        err_msg = error_body

                    logger.warning(f"Gemini API HTTP {e.code} for '{model}': {err_msg}")

                    if e.code in (401, 403):
                        # Key unauthorized - fail fast immediately
                        raise PermissionError(
                            f"Invalid or unauthorized Gemini API key: {err_msg}. "
                            "Verify your key at https://aistudio.google.com/app/apikey."
                        )
                    elif e.code == 404:
                        # Model not available in this region or API version
                        logger.warning(f"Model '{model}' not found (HTTP 404). Falling back...")
                        last_error = RuntimeError(f"Gemini model '{model}' not found (HTTP 404): {err_msg}")
                        break
                    elif e.code in (500, 502, 503, 504, 429):
                        # Transient capacity issue, gateway timeout, or rate limit
                        last_error = RuntimeError(f"Gemini API error (HTTP {e.code}): {err_msg}")
                        if attempt < max_retries:
                            backoff = attempt * 1.5
                            logger.info(f"Transient HTTP {e.code} on '{model}'. Retrying in {backoff:.1f}s...")
                            time.sleep(backoff)
                            continue
                        else:
                            logger.warning(f"Exhausted retries for '{model}' on HTTP {e.code}. Falling back to next model...")
                            break
                    elif e.code == 400:
                        last_error = ValueError(f"Gemini API Bad Request ({model}): {err_msg}")
                        break
                    else:
                        last_error = RuntimeError(f"Gemini API error (HTTP {e.code}): {err_msg}")
                        break

                except urllib.error.URLError as ue:
                    is_timeout = (
                        isinstance(ue.reason, (socket.timeout, TimeoutError))
                        or "timed out" in str(ue.reason).lower()
                    )
                    if is_timeout:
                        logger.warning(f"URLError timeout for model '{model}' (attempt {attempt}/{max_retries}): {ue.reason}")
                        last_error = TimeoutError(f"Request to Gemini model '{model}' timed out after {self.timeout}s: {ue.reason}")
                        if attempt < max_retries:
                            time.sleep(1.0)
                            continue
                        else:
                            logger.warning(f"Model '{model}' timed out repeatedly. Falling back to alternative model...")
                            break
                    else:
                        logger.error(f"Network error connecting to Gemini API for '{model}': {ue.reason}")
                        last_error = ConnectionError(f"Failed to connect to Google Gemini API: {ue.reason}")
                        if attempt < max_retries:
                            time.sleep(1.0)
                            continue
                        break

                except Exception as ex:
                    logger.error(f"Unexpected error calling Gemini API with model '{model}': {ex}")
                    last_error = ex
                    break

        if last_error:
            raise last_error
        raise RuntimeError("Failed to generate answer from Google Gemini after trying all candidate models.")
