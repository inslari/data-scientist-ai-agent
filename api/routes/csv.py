import io
import os
import time
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db, DATA_DIR
from api.core.jobs import enqueue_ttd_suggestions, enqueue_viz_suggestions
from api.core.storage import storage
from api.models.csv import CSVFileModel
from api.models.data_group import DataGroupModel
from api.schemas.csv import (
    CSVFileResponse,
    CSVUploadResponse,
    CSVSchemaResponse,
    CSVSampleDataRequest,
    CSVSampleDataResponse,
    CSVColumnStatsResponse
)
from api.core.deps import get_current_user
from api.models.user import UserModel
from api.routes.data_group import _clear_db_cache

router = APIRouter(prefix="/csv", tags=["CSV Upload and Analysis"])

# Local fallback directory (used when MinIO is not configured)
CSV_STORAGE_DIR = os.path.join(DATA_DIR, "csvs")
os.makedirs(CSV_STORAGE_DIR, exist_ok=True)

@router.post("/upload", response_model=CSVUploadResponse)
async def upload_csv(
    files: List[UploadFile] = File(...), 
    group_name: str = Form(None),
    data_group_id: int = Form(None),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Uploads one or more CSV files, saves them to disk, extracts schema using pandas, 
    and saves the metadata to the database under a DataGroup.
    If data_group_id is provided, adds to the existing data group.
    """
    data_group = None
    if data_group_id:
        data_group = db.query(DataGroupModel).filter(
            DataGroupModel.id == data_group_id,
            DataGroupModel.user_id == current_user.id
        ).first()
        if not data_group:
            raise HTTPException(status_code=404, detail="Data group not found")
            
    if not data_group:
        if not group_name:
            group_name = f"CSV Upload - {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
        data_group = DataGroupModel(
            name=group_name,
            source_type="csv",
            user_id=current_user.id,
            created_at=int(time.time())
        )
        db.add(data_group)
        db.commit()
        db.refresh(data_group)

    uploaded_files = []
    errors = []
    
    for file in files:
        filename = file.filename or ""
        if not filename.endswith('.csv'):
            errors.append({"filename": filename or "unknown", "error": "Only .csv files are allowed."})
            continue

        unique_filename = f"{uuid.uuid4()}_{filename}"
        # Object key for MinIO, or absolute path for local storage
        object_key = f"csvs/{unique_filename}" if storage.using_minio else os.path.join(CSV_STORAGE_DIR, unique_filename)

        contents = await file.read()
        file_size = len(contents)

        # Upload to storage (MinIO or local disk). storage.upload senkron (boto3 put /
        # disk yazma) → event loop'u bloklamamak için threadpool'da çalıştır.
        file_ref = await run_in_threadpool(storage.upload, contents, object_key)

        # Process CSV (schema + stats) using a temp file when using MinIO
        def process_csv_bytes(csv_bytes: bytes):
            df = pd.read_csv(io.BytesIO(csv_bytes))
            row_count = len(df)
            columns_schema = [
                {"name": col, "type": str(df[col].dtype)}
                for col in df.columns
            ]
            stats_df = df.describe(include='all').T
            stats_df['null_count'] = df.isnull().sum()
            stats_df['dtype'] = df.dtypes.astype(str)
            stats_df = stats_df.where(pd.notnull(stats_df), None)  # type: ignore
            stats_df.reset_index(inplace=True)
            stats_df.rename(columns={'index': 'column_name'}, inplace=True)
            stats_list = stats_df.to_dict(orient="records")
            return row_count, columns_schema, stats_list

        try:
            row_count, columns_schema, column_stats_cache = await run_in_threadpool(
                process_csv_bytes, contents
            )
        except Exception as e:
            # If parsing fails, remove from storage and record error
            storage.delete(file_ref)
            errors.append({"filename": filename, "error": f"Failed to parse CSV file: {str(e)}"})
            continue

        new_csv = CSVFileModel(
            filename=filename,
            file_path=file_ref,   # object key (MinIO) or local path
            file_size=file_size,
            row_count=row_count,
            columns_schema=columns_schema,
            column_stats_cache=column_stats_cache,
            created_at=int(time.time()),
            data_group_id=data_group.id,
            user_id=current_user.id
        )

        db.add(new_csv)
        db.commit()
        db.refresh(new_csv)
        uploaded_files.append(new_csv)

    # Clear cache if adding to an existing data group
    if data_group_id and uploaded_files:
        _clear_db_cache(db, data_group)

    # Yeni bir data group oluşturulduysa (mevcut gruba ekleme değil) ve en az bir
    # dosya yüklendiyse, Talk-to-Data örnek sorularını arka planda üret.
    if not data_group_id and uploaded_files:
        enqueue_ttd_suggestions(db, data_group.id)
        enqueue_viz_suggestions(db, data_group.id)

    return {
        "status": "success" if uploaded_files else "error",
        "message": f"Successfully uploaded {len(uploaded_files)} files." + (f" ({len(errors)} errors)" if errors else ""),
        "data": uploaded_files,
        "errors": errors
    }

@router.get("/files", response_model=List[CSVFileResponse])
def get_csv_files(db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Returns a list of all uploaded CSV files.
    """
    return db.query(CSVFileModel).filter(CSVFileModel.user_id == current_user.id).all()

@router.get("/{file_id}", response_model=CSVFileResponse)
def get_csv_file(file_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Returns details of a specific CSV file.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id, CSVFileModel.user_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
    return db_file

@router.get("/{file_id}/download")
def download_csv_file(file_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Downloads a specific CSV file.
    When MinIO is configured, redirects to a presigned URL.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id, CSVFileModel.user_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    if storage.using_minio:
        url = storage.get_download_url(db_file.file_path)
        return RedirectResponse(url=url)

    # Local fallback
    if not os.path.exists(db_file.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(
        path=db_file.file_path,
        filename=db_file.filename,
        media_type="text/csv"
    )

@router.delete("/{file_id}")
def delete_csv_file(file_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Deletes a CSV file from the database and removes it from storage.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id, CSVFileModel.user_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    storage.delete(db_file.file_path)

    db.delete(db_file)
    db.commit()

    data_group = db.query(DataGroupModel).filter(DataGroupModel.id == db_file.data_group_id).first()
    if data_group:
        _clear_db_cache(db, data_group)

    return {"status": "success", "message": f"CSV file with id {file_id} deleted successfully"}

@router.post("/{file_id}/schema", response_model=CSVSchemaResponse)
def get_csv_schema(file_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Returns the schema (columns and types) of a specific CSV file.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id, CSVFileModel.user_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
        
    return {
        "status": "success",
        "data": db_file.columns_schema
    }

@router.post("/{file_id}/sample-data", response_model=CSVSampleDataResponse)
async def get_csv_sample_data(file_id: int, req: CSVSampleDataRequest, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Returns the first N rows of the CSV file.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id, CSVFileModel.user_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    def get_sample(file_ref: str, limit: int):
        with storage.get_stream(file_ref) as stream:
            df = pd.read_csv(stream, nrows=limit)
            df = df.where(pd.notnull(df), None)  # type: ignore
            return list(df.columns), df.to_dict(orient="records")

    try:
        columns, rows = await run_in_threadpool(get_sample, db_file.file_path, req.limit)
        return {
            "status": "success",
            "data": {"columns": columns, "rows": rows}
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to read sample data: {str(e)}"}

@router.post("/{file_id}/column-stats", response_model=CSVColumnStatsResponse)
async def get_csv_column_stats(file_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Calculates and returns statistics for each column in the CSV file. Uses cached stats if available.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id, CSVFileModel.user_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    if db_file.column_stats_cache:
        return {"status": "success", "data": db_file.column_stats_cache}

    def calculate_stats(tmp_path: str):
        df = pd.read_csv(tmp_path)
        stats_df = df.describe(include='all').T
        stats_df['null_count'] = df.isnull().sum()
        stats_df['dtype'] = df.dtypes.astype(str)
        stats_df = stats_df.where(pd.notnull(stats_df), None)  # type: ignore
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'column_name'}, inplace=True)
        return stats_df.to_dict(orient="records")

    # İndirme (storage.download_to_tempfile) ve okuma birlikte threadpool'da çalışsın;
    # böylece senkron indirme de event loop'u bloklamaz.
    def download_and_calculate(file_ref: str):
        with storage.download_to_tempfile(file_ref) as tmp_path:
            return calculate_stats(tmp_path)

    try:
        stats_list = await run_in_threadpool(download_and_calculate, db_file.file_path)

        # Cache it for future
        db_file.column_stats_cache = stats_list
        db.commit()

        return {"status": "success", "data": stats_list}
    except Exception as e:
        return {"status": "error", "message": f"Failed to calculate column stats: {str(e)}"}
