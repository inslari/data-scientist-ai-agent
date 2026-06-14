import asyncio
import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

async def test():
    from dotenv import load_dotenv
    load_dotenv(override=True)
    url = os.getenv("API_DATABASE_URL")
    if not url or not url.startswith("postgres"):
        print("No postgres URL found")
        return
    url = url.replace("postgresql+psycopg2://", "postgresql://")
    print(f"Connecting to {url}")
    async with AsyncConnectionPool(url, kwargs={"autocommit": True}) as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            tables = await cur.fetchall()
            print("Tables in public schema:")
            for t in tables:
                print(t)

asyncio.run(test())
