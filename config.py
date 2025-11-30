"""Centralized configuration loader for NinjaZtrade.

Loads environment variables from a local `.env` file (ignored by git) and
provides a cached Settings object that can be imported throughout the codebase
without risking accidental exposure of API keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from the project-level .env if it exists.
_ENV_PATH = Path(__file__).with_name(".env")
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


def _split_origins(value: str | None) -> List[str]:
    if not value:
        return ["http://localhost:5173"]
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@dataclass(slots=True)
class Settings:
    """Runtime configuration sourced from environment variables."""

    binance_api_key: str | None = os.getenv("BINANCE_API_KEY")
    binance_secret_key: str | None = os.getenv("BINANCE_SECRET_KEY")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")

    secret_key: str | None = os.getenv("SECRET_KEY")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    allowed_origins: List[str] = field(default_factory=lambda: _split_origins(os.getenv("ALLOWED_ORIGINS")))
    rate_limit: str = os.getenv("RATE_LIMIT", "100/hour")
    environment: str = os.getenv("ENVIRONMENT", "development")

    def require(self, *keys: str) -> None:
        """Ensure required secrets are present before continuing."""
        missing = [key for key in keys if not getattr(self, key, None)]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
