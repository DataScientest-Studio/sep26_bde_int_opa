"""Placeholder test so CI (pytest) passes until real API tests exist (Stage 4)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_schema_file_exists():
    assert (ROOT / "sql" / "init.sql").is_file()


def test_compose_file_exists():
    assert (ROOT / "docker-compose.yml").is_file()
