import os

import uvicorn
from dotenv import load_dotenv

from api import app

load_dotenv(override=True)


def main() -> None:
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    log_level = os.getenv("FASTAPI_LOG_LEVEL", "info")

    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
