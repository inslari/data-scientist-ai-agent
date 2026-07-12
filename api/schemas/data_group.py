from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from api.schemas.mcp import MCPConfigResponse
from api.schemas.csv import CSVFileResponse

class DataGroupBase(BaseModel):
    name: str
    source_type: str

class DataGroupResponse(DataGroupBase):
    id: int
    created_at: int
    user_id: Optional[int] = None

    mcp_config: Optional[MCPConfigResponse] = None
    csv_files: Optional[List[CSVFileResponse]] = None

    # Schema cache metadata — frontend'de "Son Analiz: ..." göstermek için
    schema_cache_updated_at: Optional[int] = None   # Unix timestamp, None = henüz analiz yok

    # Domain (sector) pack for this dataset. None/empty = global DOMAIN_PACK.
    domain_pack: Optional[str] = None

    # Domain/dataset description (markdown) used as pipeline context. None = none.
    domain_doc: Optional[str] = None

    # Talk-to-Data örnek soru önerileri. None/[] = henüz üretilmedi → UI boş gösterir.
    ttd_suggestions: Optional[List[str]] = None

    # Data Visualization öneri kartları. None/[] = henüz üretilmedi → UI boş gösterir.
    viz_suggestions: Optional[List[dict]] = None

    model_config = ConfigDict(from_attributes=True)


class QueryRequest(BaseModel):
    query: str


# ─────────────────────────────────────────────────────────────────────────────
# Logic Extracted from Routes (To Avoid Circular Dependencies)
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import duckdb
from sqlalchemy.orm import Session
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from api.models.data_group import DataGroupModel
from api.routes.mcp import _call_mcp_tool

from src.core.logging_config import get_logger
logger = get_logger(__name__)

async def get_data_group_schema_logic(group_id: int, db: Session, user_id: int):
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == user_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    if db_group.source_type == "mcp":
        if not db_group.mcp_config:
            raise HTTPException(status_code=400, detail="MCP config is missing for this data group")
        if db_group.mcp_config.schema_cache:
            logger.warning("[MCP] schema_cache hit, skipping MCP call")
            return {"status": "success", "schema": db_group.mcp_config.schema_cache}
        logger.info(f"[MCP] schema_cache miss, calling MCP server for get_schema | type={db_group.mcp_config.type} url={getattr(db_group.mcp_config, 'url', None)}")
        schema_text = await _call_mcp_tool(db_group.mcp_config, "get_schema")
        return {"status": "success", "schema": schema_text}
        
    elif db_group.source_type == "csv":
        schema_lines = []
        for csv_file in db_group.csv_files:
            table_name = os.path.splitext(csv_file.filename)[0].lower()
            table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
            schema_lines.append(f"Table: {table_name}")
            schema_lines.append(f"Description: CSV file with {csv_file.row_count} rows")
            schema_lines.append("Columns:")
            for col in csv_file.columns_schema:
                schema_lines.append(f"- {col['name']} ({col['type']})")
            schema_lines.append("")
        return {"status": "success", "schema": "\n".join(schema_lines)}
    
    raise HTTPException(status_code=400, detail="Unknown source type")

async def run_data_group_query_logic(group_id: int, req: QueryRequest, db: Session, user_id: int):
    # Central read-only guard (C2): this is the shared choke point for both graphs'
    # agent-generated SQL and the user-facing query endpoint. Enforcing here also
    # covers the CSV/DuckDB path, which otherwise executed arbitrary SQL unguarded.
    from src.core.sql_safety import validate_read_only
    ok, reason = validate_read_only(req.query)
    if not ok:
        return {"status": "error", "result": f"Rejected: {reason}"}

    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == user_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    if db_group.source_type == "mcp":
        if not db_group.mcp_config:
            raise HTTPException(status_code=400, detail="MCP config is missing for this data group")
        result_text = await _call_mcp_tool(db_group.mcp_config, "run_sql_query", {"query": req.query})
        return {"status": "success", "result": result_text}
        
    elif db_group.source_type == "csv":
        from api.core.storage import storage
        # SQLAlchemy ilişkisi (csv_files) ve tablo adlarını, thread'e geçmeden ÖNCE
        # düz veriye çıkar (session thread-safe değil). Bloklayan storage indirme +
        # senkron duckdb işini ayrı bir thread'de çalıştır → event loop bloklanmaz.
        csv_specs = [
            (re.sub(r'[^a-z0-9_]', '_', os.path.splitext(f.filename)[0].lower()), f.file_path)
            for f in db_group.csv_files
        ]
        query = req.query

        def _run_csv_query() -> dict:
            # For each CSV, download to a temp file if on MinIO, then register in DuckDB
            context_managers = []
            try:
                con = duckdb.connect(database=':memory:')
                for table_name, file_path in csv_specs:
                    ctx = storage.download_to_tempfile(file_path)
                    local_path = ctx.__enter__()
                    context_managers.append(ctx)

                    # TODO(dq-layer1b): Veri Kalitesi Katmanı 1b — yükleme-anı yapısal
                    # doğrulama. read_csv_auto'yu store_rejects=true + strict_mode ile
                    # oku, reject_errors/reject_scans'tan bozuk satırları (kolon-sayısı
                    # uyuşmazlığı, delimiter/quoting bozukluğu) say ve raporla. Şu an
                    # sadece Layer 1a (MCP-parse invaryantı) + Layer 2 (anlamsal) aktif;
                    # bkz. src/services/data_quality.py.
                    con.execute(f'CREATE VIEW "{table_name}" AS SELECT * FROM read_csv_auto(\'{local_path}\')')

                result_df = con.execute(query).df()

                # Sağlam serialization (bkz. src/core/row_codec.py): her hücre
                # literal-güvenli primitife indirgenir. Eski `str(row)`, pandas
                # Timestamp'i `Timestamp('...', tz='...')` (içteki virgül) olarak
                # yazıp istemcide kolon kaymasına yol açıyordu — asıl fraud bozulması.
                from src.core.row_codec import encode_result
                from src.core.config import settings
                columns = result_df.columns.tolist()
                # Transport tavanı: env-override'lı (settings.MAX_FETCH_ROWS). Tam veri
                # feature engineering'e ulaşsın diye yüksek; asıl training cap ayrı ve
                # stratified (feature_builder + MAX_TRAINING_ROWS). Sessiz 10k truncate YOK.
                result_text = encode_result(
                    columns, result_df.itertuples(index=False, name=None),
                    max_rows=settings.MAX_FETCH_ROWS,
                )
                return {"status": "success", "result": result_text}
            except Exception as e:
                return {"status": "error", "result": f"SQL Error: {str(e)}"}
            finally:
                for ctx in context_managers:
                    try:
                        ctx.__exit__(None, None, None)
                    except Exception:
                        pass

        return await run_in_threadpool(_run_csv_query)

    raise HTTPException(status_code=400, detail="Unknown source type")
