"""Entry point: `ccproxy` or `python -m ccproxy`."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .config import ConfigError, from_env, load_dotenv


def main() -> int:
    load_dotenv()
    try:
        settings = from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from .app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
