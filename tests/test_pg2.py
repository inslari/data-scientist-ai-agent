import asyncio
import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

async def test():
    from dotenv import load_dotenv
    load_dotenv(override=True)
    url = os.getenv("API_DATABASE_URL")
    url = url.replace("postgresql+psycopg2://", "postgresql://")
    async with AsyncConnectionPool(url, kwargs={"autocommit": True}) as pool:
        saver = AsyncPostgresSaver(pool)
        print("saver has pool?", hasattr(saver, "pool"))
        print("saver has conn?", hasattr(saver, "conn"))

asyncio.run(test())
