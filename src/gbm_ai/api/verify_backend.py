from __future__ import annotations

import json

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager


def main() -> None:
    settings = get_settings()
    print("GBM CDSS BACKEND FOUNDATION CHECK")
    print("=" * 46)
    print(json.dumps(settings.safe_summary(), indent=2))

    database = DatabaseManager(settings)
    try:
        database.ping()
    except Exception as exc:
        print("\nPostgreSQL: NOT READY")
        print(f"Error type: {exc.__class__.__name__}")
        print(
            "Check GBM_DATABASE_URL, PostgreSQL service, database/user, "
            "password and network access."
        )
        raise SystemExit(1)
    finally:
        database.dispose()

    print("\nPostgreSQL: READY")
    print("SELECT 1: PASS")
    print("Phase 4 Step 1 database foundation: READY")


if __name__ == "__main__":
    main()
