import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db
from api.models.data_group import DataGroupModel
from api.schemas.data_group import DataGroupResponse

router = APIRouter(prefix="/data-groups", tags=["Data Groups"])

@router.get("", response_model=List[DataGroupResponse])
def get_data_groups(db: Session = Depends(get_db)):
    """
    Returns a list of all data groups, including their associated MCP config or CSV files.
    """
    return db.query(DataGroupModel).all()

@router.delete("/{group_id}")
def delete_data_group(group_id: int, db: Session = Depends(get_db)):
    """
    Deletes a data group.
    If it's an MCP source, the MCP connection is deleted.
    If it's a CSV source, the CSV metadata and physical files are deleted.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    # If it's a CSV group, delete physical files
    if db_group.source_type == "csv" and db_group.csv_files:
        for csv_file in db_group.csv_files:
            if os.path.exists(csv_file.file_path):
                try:
                    os.remove(csv_file.file_path)
                except Exception as e:
                    print(f"Failed to delete file {csv_file.file_path}: {e}")
                    
    db.delete(db_group)
    db.commit()
    return {"status": "success", "message": f"Data group with id {group_id} deleted successfully"}
