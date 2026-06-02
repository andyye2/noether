#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from noether.io.providers import Provider

CONFIG_PATH = Path().home() / ".config" / "emmi" / "config.json"

# Each provider defines the env vars it needs:
PROVIDER_ENV_VARS = {
    Provider.HUGGINGFACE: ["HF_TOKEN"],
    Provider.AWS: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    Provider.AZURE: ["AZURE_STORAGE_CONNECTION_STRING"],  # placeholder
    Provider.GCP: ["GOOGLE_APPLICATION_CREDENTIALS"],  # placeholder
}

OPTIONAL_PROVIDER_ENV_VARS = {
    Provider.AWS: [
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ],
}


def get_credentials(provider: Provider) -> dict[str, Any]:
    """Retrieve credentials for a given storage provider.

    Priority:
    1. Environment variables
    2. Local config file (e.g. `~/.config/emmi/config.json`)

    Args:
        provider: Is an instance of Provider enum.

    Returns:
        - Dictionary of credentials.
    """

    if provider not in PROVIDER_ENV_VARS:
        raise ValueError(f"Unsupported provider: {provider.value}")

    optional = OPTIONAL_PROVIDER_ENV_VARS.get(provider, list())
    creds: dict[str, Any] = {}
    for var_name in PROVIDER_ENV_VARS[provider]:
        value = os.getenv(var_name) or _get_from_config(provider.value, var_name)
        if value is None and var_name not in optional:
            raise RuntimeError(f"Missing credential: {var_name} for provider {provider.value}")
        if value is not None and value != "":
            creds[var_name] = value

    # Also load optional env vars (include if present and non-empty)
    for opt_name in OPTIONAL_PROVIDER_ENV_VARS.get(provider, []):
        opt_val = os.getenv(opt_name) or _get_from_config(provider.value, opt_name)
        if opt_val is not None and opt_val != "":
            creds[opt_name] = opt_val

    # Normalize AWS region: allow either, expose only AWS_REGION if present:
    if provider is Provider.AWS:
        region = creds.get("AWS_REGION") or creds.get("AWS_DEFAULT_REGION")
        if region:
            creds["AWS_REGION"] = region
            creds.pop("AWS_DEFAULT_REGION", None)

    # Log only which keys were loaded to avoid leaking secrets
    logger.debug(f"Extracted credentials for provider {provider.value} (keys: {sorted(creds.keys())})")

    return creds


def _get_from_config(provider: str, key: str, default: Any = None) -> Any:
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
            return config.get(provider, {}).get(key, default)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in config file: {CONFIG_PATH}") from None
    return default
