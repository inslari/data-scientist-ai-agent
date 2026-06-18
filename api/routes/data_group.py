import os
import json
import duckdb
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from api.core.database import get_db
from api.models.data_group import DataGroupModel, ChatMessageModel
from api.schemas.data_group import DataGroupResponse
from api.schemas.chat import ChatRequest, ChatResponse, ChatHistoryResponse
from src.graphs.talk_to_data_graph import build_ttd_graph
from src.core.config import settings as _ds_settings
from api.core.deps import get_current_user
from api.models.user import UserModel

router = APIRouter(prefix="/data-groups", tags=["Data Groups"])


from api.schemas.data_group import QueryRequest, get_data_group_schema_logic, run_data_group_query_logic

@router.get("", response_model=List[DataGroupResponse])
def get_data_groups(db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Returns a list of all data groups, including their associated MCP config or CSV files.
    """
    return db.query(DataGroupModel).filter(DataGroupModel.user_id == current_user.id).all()

@router.post("/{group_id}/schema")
async def get_data_group_schema(group_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Returns the database schema for the specified data group.
    """
    return await get_data_group_schema_logic(group_id, db, current_user.id)

@router.post("/{group_id}/query")
async def run_data_group_query(group_id: int, req: QueryRequest, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Executes a SQL query against the specified data group.
    """
    return await run_data_group_query_logic(group_id, req, db, current_user.id)

@router.delete("/{group_id}")
def delete_data_group(group_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Deletes a data group.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == current_user.id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")

    # If it's a CSV group, delete physical files from storage (MinIO or local)
    if db_group.source_type == "csv" and db_group.csv_files:
        from api.core.storage import storage
        for csv_file in db_group.csv_files:
            storage.delete(csv_file.file_path)

    db.delete(db_group)
    db.commit()
    return {"status": "success", "message": f"Data group with id {group_id} deleted successfully"}

class DataGroupUpdate(BaseModel):
    name: Optional[str] = None
    domain_pack: Optional[str] = None   # "" / None cleared via explicit send → global DOMAIN_PACK
    domain_doc: Optional[str] = None    # markdown dataset context; "" / None clears it

@router.patch("/{group_id}")
def update_data_group(group_id: int, req: DataGroupUpdate, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Updates a data group (e.g. renaming or setting its domain pack).
    Only fields explicitly present in the request body are modified.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == current_user.id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")

    fields_set = req.model_fields_set
    if "name" in fields_set and req.name is not None:
        db_group.name = req.name
    if "domain_pack" in fields_set:
        db_group.domain_pack = req.domain_pack or None
    if "domain_doc" in fields_set:
        db_group.domain_doc = req.domain_doc or None

    db.commit()
    db.refresh(db_group)
    return db_group



# ─────────────────────────────────────────────────────────────────────────────
# SQLite Schema Cache Helpers
# ─────────────────────────────────────────────────────────────────────────────
# db_context_node LLM schema analizinin sonucu (db_intelligence + raw_schema)
# DataGroupModel tablosunda kalıcı olarak saklanır.
# Sunucu yeniden başlasa da cache korunur.
# Refresh tetiklendiğinde cache silinir ve bir sonraki soruda yeniden üretilir.

import time as _time


def _read_db_cache(db_group: DataGroupModel) -> tuple[dict, str] | None:
    """
    SQLite'dan cache'lenmiş db_intelligence + raw_schema döner.
    Cache yoksa ya da silinmişse None döner.
    """
    if db_group.db_intelligence_cache and db_group.raw_schema_cache:
        return db_group.db_intelligence_cache, db_group.raw_schema_cache
    return None


def _write_db_cache(db, db_group: DataGroupModel, db_intelligence: dict, raw_schema: str) -> None:
    """
    db_intelligence + raw_schema'yı SQLite'a yazar.
    """
    db_group.db_intelligence_cache = db_intelligence
    db_group.raw_schema_cache = raw_schema
    db_group.schema_cache_updated_at = int(_time.time())
    db.commit()
    print(f"[schema-cache] SQLite'a yazıldı: data_group_id={db_group.id}")


def _clear_db_cache(db, db_group: DataGroupModel) -> None:
    """
    Cache'i temizler. Bir sonraki chat sorgusunda yeniden üretilecek.
    """
    db_group.db_intelligence_cache = None
    db_group.raw_schema_cache = None
    db_group.schema_cache_updated_at = None
    db.commit()
    print(f"[schema-cache] Cache temizlendi: data_group_id={db_group.id}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /{group_id}/refresh-schema
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/{group_id}/refresh-schema")
async def refresh_data_group_schema(group_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Data group'un db_intelligence cache'ini temizler.
    Bir sonraki chat sorgusunda db_context_node yeniden çalışarak
    güncel veriyle schema analizini yeniden yapacaktır.

    Ne zaman kullanılmalı:
    - MCP bağlantısının arkasındaki veritabanı değiştiğinde
    - CSV dosyaları güncellendiğinde / yeni dosya eklendiğinde
    - Veri grupları yeniden yapılandırıldığında
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == current_user.id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")

    had_cache = db_group.db_intelligence_cache is not None
    _clear_db_cache(db, db_group)

    return {
        "status": "success",
        "message": "Schema cache temizlendi. Bir sonraki chat sorgusunda yeniden üretilecek.",
        "had_cache": had_cache,
        "cache_cleared_at": int(_time.time()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /{group_id}/chat
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/{group_id}/chat", response_model=ChatHistoryResponse)
def get_chat_history(group_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Returns the long-term memory and all chat messages for a specific data group.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == current_user.id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    messages = db.query(ChatMessageModel).filter(
        ChatMessageModel.data_group_id == group_id
    ).order_by(ChatMessageModel.created_at.asc()).all()
    
    return {
        "long_term_memory": db_group.long_term_memory,
        "messages": messages
    }


@router.delete("/{group_id}/chat")
def clear_chat_history(group_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Deletes all chat messages for a specific data group and resets its long-term memory.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == current_user.id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")

    deleted = db.query(ChatMessageModel).filter(
        ChatMessageModel.data_group_id == group_id
    ).delete(synchronize_session=False)
    db_group.long_term_memory = None
    db.commit()

    return {"status": "success", "deleted": deleted}


@router.post("/{group_id}/chat", response_model=ChatResponse)
async def chat_with_data_group(
    group_id: int,
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Talk to Data — Doğal dil sorusunu alır, Talk to Data bağımsız
    LangGraph pipeline'ından (build_ttd_graph) geçirir:

      ttd_db_context_node → ttd_sql_node → ttd_chart_node → ttd_interpreter_node

    Schema Cache Davranışı (SQLite tabanlı):
    - İlk soruda ttd_db_context_node çalışır → LLM schema analizi yapılır → SQLite'a yazılır.
    - Sonraki sorgularda cache'den okuyarak ttd_db_context_node bypass edilir.
    - /refresh-schema çağrıldığında cache temizlenir → bir sonraki sorguda yenilenir.
    - Sunucu yeniden başlatılsa da cache SQLite'ta korunur.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == current_user.id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")

    
    try:
        graph = build_ttd_graph()

        # 1. Save user message to DB
        user_msg_model = ChatMessageModel(
            data_group_id=group_id,
            role="user",
            content=req.message,
            created_at=int(_time.time())
        )
        db.add(user_msg_model)
        db.commit()

        # 2. Get the last 3 messages from DB to pass to the graph
        past_msgs_db = db.query(ChatMessageModel).filter(
            ChatMessageModel.data_group_id == group_id
        ).order_by(ChatMessageModel.created_at.desc()).limit(3).all()
        
        # reverse them to chronological order
        past_msgs_db.reverse()

        messages = [
            {"role": m.role, "content": m.content}
            for m in past_msgs_db
        ]

        # ── SQLite cache check ──────────────────────────────────────────────
        cached = _read_db_cache(db_group)

        initial_state: dict = {
            "user_query":       req.message,
            "messages":         messages,
            "long_term_memory": db_group.long_term_memory,
            "data_group_id":    group_id,
            "domain_pack":      db_group.domain_pack or _ds_settings.DOMAIN_PACK,
            "domain_doc":       db_group.domain_doc or "",
        }

        if cached:
            db_intelligence, raw_schema = cached
            initial_state["db_intelligence"] = db_intelligence
            initial_state["raw_schema"]      = raw_schema
            updated_at = db_group.schema_cache_updated_at
            print(f"[ttd chat] SQLite cache hit: group_id={group_id}, "
                  f"cache_date={_time.strftime('%Y-%m-%d %H:%M', _time.localtime(updated_at)) if updated_at else '?'}")
        else:
            print(f"[ttd chat] SQLite cache miss: group_id={group_id} — ttd_db_context_node çalışacak")

        config = {"configurable": {"data_group_id": group_id, "user_id": current_user.id}}
        result = await graph.ainvoke(initial_state, config=config)

        # ── SQLite'a yaz (ttd_db_context_node çalıştıysa) ──────────────────
        if not cached and result.get("db_intelligence") and result.get("raw_schema"):
            _write_db_cache(db, db_group, result["db_intelligence"], result["raw_schema"])

        # Error handling
        error = result.get("error")
        if error:
            # Save error as AI response
            ai_msg_model = ChatMessageModel(
                data_group_id=group_id,
                role="assistant",
                content=f"Bir hata oluştu: {error}",
                created_at=int(_time.time())
            )
            db.add(ai_msg_model)
            db.commit()
            return ChatResponse(
                answer=ai_msg_model.content,
                sql=result.get("sql_query_raw"),
                chart=None,
                columns=[],
                rows=[],
                warning=None,
            )

        summary = result.get("sql_query_summary", "").strip()
        if not summary:
            row_count = len(result.get("sql_query_results", []))
            summary = f"Query completed, {row_count} row(s) returned."

        # 3. Save AI message to DB
        ai_msg_model = ChatMessageModel(
            data_group_id=group_id,
            role="assistant",
            content=summary,
            created_at=int(_time.time()),
            sql=result.get("sql_query_raw"),
            chart=result.get("sql_query_chart"),
            columns=result.get("sql_query_columns", []),
            rows=result.get("sql_query_results", []),
            warning=result.get("sql_query_warning")
        )
        db.add(ai_msg_model)

        # 4. Update long_term_memory if the graph returned a new one
        updated_memory = result.get("long_term_memory")
        if updated_memory is not None and updated_memory != db_group.long_term_memory:
            db_group.long_term_memory = updated_memory

        db.commit()

        return ChatResponse(
            answer=summary,
            sql=result.get("sql_query_raw"),
            chart=result.get("sql_query_chart"),
            columns=result.get("sql_query_columns", []),
            rows=result.get("sql_query_results", []),
            warning=result.get("sql_query_warning") or None,
        )

    except Exception as e:
        print(f"[chat_with_data_group] Hata: {e}")
        raise HTTPException(status_code=500, detail=f"Chat pipeline hatası: {str(e)}")
