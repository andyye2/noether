#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import json
from pathlib import Path

from noether.io.credentials import get_credentials
from noether.io.providers import Provider


def test_get_credentials_from_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    creds = get_credentials(Provider.HUGGINGFACE)
    assert creds["HF_TOKEN"] == "dummy-token"


def test_get_credentials_from_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_data = {"huggingface": {"HF_TOKEN": "from-config"}}
    config_path.write_text(json.dumps(config_data))
    monkeypatch.setattr("noether.io.credentials.CONFIG_PATH", config_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    creds = get_credentials(Provider.HUGGINGFACE)
    assert creds["HF_TOKEN"] == "from-config"
