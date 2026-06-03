import re

with open("src/services/data_executor.py", "r") as f:
    content = f.read()

# 1. Update imports
if "from langchain_core.runnables import RunnableConfig" not in content:
    content = content.replace("from mcp.client.streamable_http import streamable_http_client", "from mcp.client.streamable_http import streamable_http_client\nfrom langchain_core.runnables import RunnableConfig")

# 2. Replace _fetch_all_tables
old_fetch = '''async def _fetch_all_tables(table_to_db: dict[str, str]) -> dict[str, str]:
    """
    Tabloları paralel çeker.
    Büyük tablolar için offset-based chunking yapar.
    """
    MCP_URL = settings.MCP_SERVER_URL

    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def fetch_one(table: str, db_name: str) -> tuple[str, str]:
                # Önce toplam satır sayısını öğren
                try:
                    count_result = await session.call_tool(
                        "run_sql_query",
                        {"query": f\'SELECT COUNT(*) FROM "{table}"\', "db_name": db_name}
                    )
                    count_text = ""
                    if getattr(count_result, "content", None):
                        for item in count_result.content:
                            if getattr(item, "type", "") == "text":
                                count_text = getattr(item, "text", "")
                    # Parse count
                    total_rows = _extract_count(count_text)
                except Exception:
                    total_rows = CHUNK_SIZE

                print(f"[execution] '{table}': {total_rows:,} satır tespit edildi")

                # Küçük tablo → direkt çek
                if total_rows <= CHUNK_SIZE:
                    result = await session.call_tool(
                        "run_sql_query",
                        {"query": f\'SELECT * FROM "{table}"\', "db_name": db_name}
                    )
                    if getattr(result, "content", None):
                        for item in result.content:
                            if getattr(item, "type", "") == "text":
                                return table, getattr(item, "text", "")
                    return table, str(result)

                # Büyük tablo → chunked fetch (max MAX_FETCH_ROWS)
                fetch_limit = min(total_rows, MAX_FETCH_ROWS)
                chunks_text = []
                header_written = False

                for offset in range(0, fetch_limit, CHUNK_SIZE):
                    limit = min(CHUNK_SIZE, fetch_limit - offset)
                    query = f\'SELECT * FROM "{table}" LIMIT {limit} OFFSET {offset}\'
                    result = await session.call_tool(
                        "run_sql_query",
                        {"query": query, "db_name": db_name}
                    )
                    chunk_text = ""
                    if getattr(result, "content", None):
                        for item in result.content:
                            if getattr(item, "type", "") == "text":
                                chunk_text = getattr(item, "text", "")

                    if not header_written:
                        chunks_text.append(chunk_text)
                        header_written = True
                    else:
                        # Sadece veri satırlarını al (header'ı atla)
                        data_lines = [l for l in chunk_text.splitlines()
                                      if l.strip().startswith("(")]
                        chunks_text.append("\\n".join(data_lines))

                    print(f"[execution] '{table}' chunk {offset//CHUNK_SIZE+1}: {offset+limit:,}/{fetch_limit:,}")

                return table, "\\n".join(chunks_text)

            results = await asyncio.gather(
                *[fetch_one(t, db) for t, db in table_to_db.items()],
                return_exceptions=True
            )

            output: dict[str, str] = {}
            for res in results:
                if isinstance(res, BaseException):
                    raise res
                table, raw = res
                output[table] = raw
            return output'''

new_fetch = '''async def _fetch_all_tables(table_to_db: dict[str, str], data_group_id: int) -> dict[str, str]:
    from api.core.database import SessionLocal
    from api.routes.data_group import run_data_group_query, QueryRequest
    
    async def fetch_one(table: str, db_name: str) -> tuple[str, str]:
        db = SessionLocal()
        try:
            try:
                req = QueryRequest(query=f'SELECT COUNT(*) FROM "{table}"')
                count_result = await run_data_group_query(data_group_id, req, db)
                count_text = str(count_result.get("result", "")) if count_result.get("status") == "success" else ""
                total_rows = _extract_count(count_text)
            except Exception:
                total_rows = CHUNK_SIZE

            print(f"[execution] '{table}': {total_rows:,} satır tespit edildi")

            if total_rows <= CHUNK_SIZE:
                req = QueryRequest(query=f'SELECT * FROM "{table}"')
                result = await run_data_group_query(data_group_id, req, db)
                if result.get("status") == "success":
                    return table, str(result.get("result", ""))
                return table, ""

            fetch_limit = min(total_rows, MAX_FETCH_ROWS)
            chunks_text = []
            header_written = False

            for offset in range(0, fetch_limit, CHUNK_SIZE):
                limit = min(CHUNK_SIZE, fetch_limit - offset)
                query = f'SELECT * FROM "{table}" LIMIT {limit} OFFSET {offset}'
                req = QueryRequest(query=query)
                result = await run_data_group_query(data_group_id, req, db)
                
                chunk_text = str(result.get("result", "")) if result.get("status") == "success" else ""

                if not header_written:
                    chunks_text.append(chunk_text)
                    header_written = True
                else:
                    data_lines = [l for l in chunk_text.splitlines() if l.strip().startswith("(")]
                    chunks_text.append("\\n".join(data_lines))

                print(f"[execution] '{table}' chunk {offset//CHUNK_SIZE+1}: {offset+limit:,}/{fetch_limit:,}")

            return table, "\\n".join(chunks_text)
        finally:
            db.close()

    results = await asyncio.gather(
        *[fetch_one(t, db) for t, db in table_to_db.items()],
        return_exceptions=True
    )

    output: dict[str, str] = {}
    for res in results:
        if isinstance(res, BaseException):
            raise res
        table, raw = res
        output[table] = raw
    return output'''
content = content.replace(old_fetch, new_fetch)


# 3. Replace execute_data_operations signature
content = content.replace("async def execute_data_operations(state: dict) -> dict:", 
"""async def execute_data_operations(state: dict, config: RunnableConfig) -> dict:
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        return {"error": "data_group_id is missing in config", "current_step": "execution_node"}
""")

# 4. Replace call to _fetch_all_tables
content = content.replace("raw_results = await _fetch_all_tables(table_to_db)", "raw_results = await _fetch_all_tables(table_to_db, data_group_id)")

# 5. Replace joined fetch loop
old_join = '''                        async with streamable_http_client(settings.MCP_SERVER_URL) as (read, write, _):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                
                                for offset in range(0, MAX_FETCH_ROWS, SAFE_CHUNK_SIZE):
                                    limit = min(SAFE_CHUNK_SIZE, MAX_FETCH_ROWS - offset)
                                    chunk_sql = f"{base_join_sql} LIMIT {limit} OFFSET {offset}"
                                    
                                    result = await session.call_tool(
                                        "run_sql_query",
                                        {"query": chunk_sql, "db_name": anchor_db}
                                    )
                                    chunk_text = ""
                                    if getattr(result, "content", None):
                                        for item in result.content:
                                            if getattr(item, "type", "") == "text":
                                                chunk_text = getattr(item, "text", "")'''
new_join = '''                        from api.core.database import SessionLocal
                        from api.routes.data_group import run_data_group_query, QueryRequest
                        db_conn = SessionLocal()
                        try:
                            for offset in range(0, MAX_FETCH_ROWS, SAFE_CHUNK_SIZE):
                                limit = min(SAFE_CHUNK_SIZE, MAX_FETCH_ROWS - offset)
                                chunk_sql = f"{base_join_sql} LIMIT {limit} OFFSET {offset}"
                                
                                req = QueryRequest(query=chunk_sql)
                                result = await run_data_group_query(data_group_id, req, db_conn)
                                chunk_text = str(result.get("result", "")) if result.get("status") == "success" else ""'''
content = content.replace(old_join, new_join)

# 6. Ensure db_conn is closed
old_close = '''                                    print(f"[execution] MERGED tablosu chunk {offset//SAFE_CHUNK_SIZE+1} bitti: {offset+limit:,}/{MAX_FETCH_ROWS:,}")'''
new_close = '''                                    print(f"[execution] MERGED tablosu chunk {offset//SAFE_CHUNK_SIZE+1} bitti: {offset+limit:,}/{MAX_FETCH_ROWS:,}")
                        finally:
                            db_conn.close()'''
content = content.replace(old_close, new_close)


with open("src/services/data_executor.py", "w") as f:
    f.write(content)

