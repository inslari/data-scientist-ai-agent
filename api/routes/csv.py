import os
import time
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db, DATA_DIR
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

router = APIRouter(prefix="/csv", tags=["CSV Upload and Analysis"])

CSV_STORAGE_DIR = os.path.join(DATA_DIR, "csvs")
os.makedirs(CSV_STORAGE_DIR, exist_ok=True)

@router.post("/upload", response_model=CSVUploadResponse)
async def upload_csv(
    files: List[UploadFile] = File(...), 
    group_name: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Uploads one or more CSV files, saves them to disk, extracts schema using pandas, 
    and saves the metadata to the database under a DataGroup.
    """
    if not group_name:
        group_name = f"CSV Upload - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
    data_group = DataGroupModel(
        name=group_name,
        source_type="csv",
        created_at=int(time.time())
    )
    db.add(data_group)
    db.commit()
    db.refresh(data_group)

    uploaded_files = []
    errors = []
    
    for file in files:
        if not file.filename.endswith('.csv'):
            errors.append({"filename": file.filename, "error": "Only .csv files are allowed."})
            continue
            
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(CSV_STORAGE_DIR, unique_filename)
        
        # Save the file to disk
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            
        file_size = os.path.getsize(file_path)
        
        # Process with pandas to get schema and row count
        try:
            df = pd.read_csv(file_path)
            row_count = len(df)
            
            columns_schema = []
            for col in df.columns:
                columns_schema.append({
                    "name": str(col),
                    "type": str(df[col].dtype)
                })
                
        except Exception as e:
            # If parsing fails, delete the file and record error
            if os.path.exists(file_path):
                os.remove(file_path)
            errors.append({"filename": file.filename, "error": f"Failed to parse CSV file: {str(e)}"})
            continue

        new_csv = CSVFileModel(
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            row_count=row_count,
            columns_schema=columns_schema,
            created_at=int(time.time()),
            data_group_id=data_group.id
        )
        
        db.add(new_csv)
        db.commit()
        db.refresh(new_csv)
        uploaded_files.append(new_csv)
        
    return {
        "status": "success" if uploaded_files else "error",
        "message": f"Successfully uploaded {len(uploaded_files)} files." + (f" ({len(errors)} errors)" if errors else ""),
        "data": uploaded_files,
        "errors": errors
    }

@router.get("/files", response_model=List[CSVFileResponse])
def get_csv_files(db: Session = Depends(get_db)):
    """
    Returns a list of all uploaded CSV files.
    """
    return db.query(CSVFileModel).all()

@router.get("/{file_id}", response_model=CSVFileResponse)
def get_csv_file(file_id: int, db: Session = Depends(get_db)):
    """
    Returns details of a specific CSV file.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
    return db_file

@router.delete("/{file_id}")
def delete_csv_file(file_id: int, db: Session = Depends(get_db)):
    """
    Deletes a CSV file from the database and removes it from disk.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
        
    # Remove from disk
    if os.path.exists(db_file.file_path):
        try:
            os.remove(db_file.file_path)
        except Exception as e:
            print(f"Failed to delete file {db_file.file_path}: {e}")
            
    db.delete(db_file)
    db.commit()
    return {"status": "success", "message": f"CSV file with id {file_id} deleted successfully"}

@router.post("/{file_id}/schema", response_model=CSVSchemaResponse)
def get_csv_schema(file_id: int, db: Session = Depends(get_db)):
    """
    Returns the schema (columns and types) of a specific CSV file.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
        
    return {
        "status": "success",
        "data": db_file.columns_schema
    }

@router.post("/{file_id}/sample-data", response_model=CSVSampleDataResponse)
def get_csv_sample_data(file_id: int, req: CSVSampleDataRequest, db: Session = Depends(get_db)):
    """
    Returns the first N rows of the CSV file.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
        
    try:
        df = pd.read_csv(db_file.file_path, nrows=req.limit)
        # Convert NaN/NaT to None for valid JSON
        df = df.where(pd.notnull(df), None)
        
        columns = list(df.columns)
        rows = df.to_dict(orient="records")
        
        return {
            "status": "success",
            "data": {
                "columns": columns,
                "rows": rows
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to read sample data: {str(e)}"}

@router.post("/{file_id}/column-stats", response_model=CSVColumnStatsResponse)
def get_csv_column_stats(file_id: int, db: Session = Depends(get_db)):
    """
    Calculates and returns statistics for each column in the CSV file.
    """
    db_file = db.query(CSVFileModel).filter(CSVFileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
        
    try:
        df = pd.read_csv(db_file.file_path)
        # describe(include='all') gives basic stats. We convert it to a dict.
        stats_df = df.describe(include='all').T
        # Add missing columns like null counts, unique counts if needed
        stats_df['null_count'] = df.isnull().sum()
        stats_df['dtype'] = df.dtypes.astype(str)
        
        # Convert NaN/NaT to None for valid JSON
        stats_df = stats_df.where(pd.notnull(stats_df), None)
        
        # Make the index (column names) a regular column
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'column_name'}, inplace=True)
        
        return {
            "status": "success",
            "data": stats_df.to_dict(orient="records")
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to calculate column stats: {str(e)}"}
