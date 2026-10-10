"""
model_client.py - Production Google GenAI client with key failover, retries,
temperature 0, caching, and diagnostics for WTQ 2026.
"""

import os
import time
import json
import random
import hashlib
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load local .env if available
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

# Allowed competition models
ALLOWED_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite"
]
DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiModelClient:
    """
    Robust Gemini client managing:
    - Primary and backup API keys
    - Exponential backoff with jitter
    - Zero temperature
    - Caching for reproducibility and quota preservation
    - Strict logging without secrets
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.primary_key = os.getenv("GOOGLE_API_KEY_PRIMARY", "").strip()
        self.backup_key = os.getenv("GOOGLE_API_KEY_BACKUP", "").strip()
        
        # Fallback to standard GOOGLE_API_KEY or GEMINI_API_KEY if primary not set
        if not self.primary_key:
            self.primary_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

        configured_model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        if configured_model not in ALLOWED_MODELS:
            # Check if stripped or formatted differently
            match = [m for m in ALLOWED_MODELS if configured_model.endswith(m)]
            if match:
                configured_model = match[0]
            else:
                raise ValueError(
                    f"Model '{configured_model}' is not in approved list: {ALLOWED_MODELS}"
                )
        self.model_name = configured_model
        
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), ".cache", "llm_responses")
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Diagnostics (Never contains API keys)
        self.diagnostics = {
            "model_name": self.model_name,
            "requests_count": 0,
            "cache_hits": 0,
            "retries_count": 0,
            "key_switches": 0,
            "active_key_slot": "primary" if self.primary_key else "none",
            "last_status": "INITIALIZED"
        }

        self.current_key = self.primary_key
        self._init_client(self.current_key)

    def _init_client(self, api_key: str):
        if not api_key:
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def get_cache_key(self, prompt: str, image_bytes: Optional[bytes] = None) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))
        hasher.update(prompt.encode("utf-8"))
        if image_bytes:
            hasher.update(hashlib.sha256(image_bytes).digest())
        return hasher.hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        if not self.use_cache:
            return None
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.diagnostics["cache_hits"] += 1
                    return data.get("text")
            except Exception:
                pass
        return None

    def _save_to_cache(self, cache_key: str, text: str) -> None:
        if not self.use_cache:
            return
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"model": self.model_name, "text": text, "timestamp": time.time()}, f, indent=2)
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: str = "image/png",
        response_mime_type: str = "application/json",
        max_retries: int = 3
    ) -> str:
        """
        Execute generation with caching, retry with backoff, key rotation on eligible failure.
        """
        cache_key = self.get_cache_key(prompt, image_bytes)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self.diagnostics["last_status"] = "CACHE_HIT"
            return cached_result

        if not self.current_key:
            raise RuntimeError(
                "No Google API key configured. Please set GOOGLE_API_KEY_PRIMARY in .env"
            )

        contents = []
        if image_bytes:
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type)
            )
        contents.append(prompt)

        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type=response_mime_type
        )

        keys_to_try = [("primary", self.primary_key)]
        if self.backup_key and self.backup_key != self.primary_key and not getattr(self, "_backup_disabled", False):
            keys_to_try.append(("backup", self.backup_key))

        last_error = None
        for key_name, key_val in keys_to_try:
            self._init_client(key_val)
            self.diagnostics["active_key_slot"] = key_name

            for attempt in range(max_retries):
                self.diagnostics["requests_count"] += 1
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config
                    )
                    text = response.text
                    self._save_to_cache(cache_key, text)
                    self.diagnostics["last_status"] = "SUCCESS"
                    return text

                except APIError as e:
                    last_error = e
                    status_code = getattr(e, "code", None)
                    is_rate_limit = status_code == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                    is_server_error = status_code in (500, 502, 503, 504)
                    is_perm_denied = status_code == 403 or "PERMISSION_DENIED" in str(e)

                    if is_perm_denied:
                        # 403 was observed to be intermittent during testing: retry the
                        # same key once after a short pause before treating it as unusable.
                        if attempt == 0:
                            self.diagnostics["retries_count"] += 1
                            time.sleep(3.0 + random.uniform(0.2, 0.8))
                            continue
                        if key_name == "backup":
                            self._backup_disabled = True
                            break
                        elif key_name == "primary" and len(keys_to_try) > 1:
                            self.diagnostics["key_switches"] += 1
                            break
                        raise

                    if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                        self.diagnostics["retries_count"] += 1
                        # Exponential backoff for rate limits: 3s, 7s, 15s
                        wait_seconds = (3.0 * (2 ** attempt)) + random.uniform(0.5, 1.5)
                        time.sleep(wait_seconds)
                        continue
                    elif is_rate_limit and key_name == "primary" and len(keys_to_try) > 1 and not getattr(self, "_backup_disabled", False):
                        self.diagnostics["key_switches"] += 1
                        break
                    else:
                        raise
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        self.diagnostics["retries_count"] += 1
                        time.sleep(2.0 + random.uniform(0.5, 1.0))
                        continue
                    raise

        self.diagnostics["last_status"] = f"FAILED: {type(last_error).__name__}"
        raise RuntimeError(f"All API attempts failed. Last error: {last_error}")
