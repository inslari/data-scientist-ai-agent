"""
api/core/csv_ingest.py
----------------------
CSV alma (ingest) yardımcıları — ham CSV byte'larından şema/istatistik çıkarma,
storage'a yazma ve `CSVFileModel` kaydı oluşturma.

Hem doğrudan CSV yüklemesi (`api/routes/csv.py`) hem de format dönüştürücü
(`api/routes/convert.py`) aynı yolu kullanır; böylece bir dosyanın veri
varlıklarına nasıl girdiği tek yerde tanımlıdır.
"""

import io
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from api.core.converters.csv_dialect import detect_dialect, looks_like_tr_number
from api.core.database import DATA_DIR
from api.core.storage import storage
from api.models.csv import CSVFileModel
from api.models.data_group import DataGroupModel

# MinIO yapılandırılmadığında kullanılan yerel dizin
CSV_STORAGE_DIR = os.path.join(DATA_DIR, "csvs")
os.makedirs(CSV_STORAGE_DIR, exist_ok=True)


def _read_csv_frame(csv_bytes: bytes) -> pd.DataFrame:
    """
    CSV'yi önce varsayılanlarla (UTF-8 + virgül) okur; bu başarısız olursa
    tespit edilen kodlama/ayırıcı ile bir kez daha dener.

    Fallback yalnızca dosya HİÇ okunamadığında devreye girer — daha önce bu
    dosyalar "Failed to parse CSV file" ile tamamen reddediliyordu. Okunabilen
    dosyaların davranışı değişmez; yanlış ayırıcı yüzünden tek sütuna çöken
    dosyaları düzeltmek kullanıcının onayına bırakılır (Dosya Dönüştürücü →
    CSV normalize), çünkü sessizce yeniden bölmek veriyi bozabilir.
    """
    try:
        return pd.read_csv(io.BytesIO(csv_bytes))
    except (UnicodeDecodeError, pd.errors.ParserError) as first_error:
        dialect = detect_dialect(csv_bytes)
        try:
            return pd.read_csv(
                io.BytesIO(csv_bytes),
                sep=dialect["delimiter"],
                encoding=dialect["encoding"],
                decimal=dialect["decimal"],
                thousands=dialect["thousands"],
            )
        except Exception:
            raise first_error


def detect_csv_warnings(csv_bytes: bytes, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Dosya okundu ama YANLIŞ okunmuş olabilir mi?

    Kritik durum: `;` ile ayrılmış UTF-8 bir dosya varsayılanlarla hatasız okunur
    — tek sütun olarak. Hata fırlamadığı için yükleme "başarılı" görünür ve
    kullanıcı sorunu ancak analiz saçmaladığında fark eder. Burada veriye
    dokunmuyoruz (sessizce yeniden bölmek veriyi bozabilir), yalnızca arayüzün
    kullanıcıyı Dosya Dönüştürücü'ye yönlendirebilmesi için bulguyu raporluyoruz.

    Dönen her bulgu: {"code": <i18n anahtarı>, "params": {...}}
    """
    warnings: List[Dict[str, Any]] = []
    dialect = detect_dialect(csv_bytes)

    # 1) Tek sütuna çökme — tespit farklı bir ayırıcıda ısrar ediyorsa
    if df.shape[1] <= 1 and dialect["delimiter"] != ",":
        label = {"\t": "tab"}.get(dialect["delimiter"], dialect["delimiter"])
        warnings.append({"code": "singleColumn", "params": {"delimiter": label}})

    # 2) UTF-8 olmayan kodlama (fallback devreye girmiş ya da girmek üzere)
    if (dialect["encoding"] or "").lower() not in ("utf-8", "utf-8-sig"):
        warnings.append({"code": "nonUtf8", "params": {"encoding": dialect["encoding"]}})

    # 3) Sayı sütunu metin olarak kalmış (1.234,56 → pandas bunu sayı yapmaz)
    if dialect["decimal"] == ",":
        text_number_columns = []
        for col in df.columns:
            # pandas 3'te metin sütunları `object` değil `StringDtype` — dtype'ı
            # ada göre değil, sayısal OLMAMASINA göre eliyoruz.
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) and sum(map(looks_like_tr_number, sample)) >= max(1, len(sample) // 2):
                text_number_columns.append(str(col))
        if text_number_columns:
            warnings.append({
                "code": "numbersAsText",
                "params": {
                    "columns": ", ".join(text_number_columns[:3]),
                    "count": len(text_number_columns),
                },
            })

    return warnings


def profile_csv_bytes(
    csv_bytes: bytes,
) -> Tuple[int, List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    CSV byte'larını ayrıştırıp (satır sayısı, kolon şeması, kolon istatistikleri,
    uyarılar) döndürür. CPU-bound — event loop içinde değil, threadpool'da çağır.
    """
    df = _read_csv_frame(csv_bytes)
    row_count = len(df)
    columns_schema = [{"name": col, "type": str(df[col].dtype)} for col in df.columns]

    stats_df = df.describe(include="all").T
    stats_df["null_count"] = df.isnull().sum()
    stats_df["dtype"] = df.dtypes.astype(str)
    stats_df = stats_df.where(pd.notnull(stats_df), None)  # type: ignore
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={"index": "column_name"}, inplace=True)
    column_stats = stats_df.to_dict(orient="records")
    warnings = detect_csv_warnings(csv_bytes, df)

    return row_count, columns_schema, column_stats, warnings


def store_csv_bytes(csv_bytes: bytes, filename: str) -> str:
    """
    CSV byte'larını storage'a (MinIO veya yerel disk) yazar ve dosya
    referansını döndürür. IO-bound — threadpool'da çağır.
    """
    unique_filename = f"{uuid.uuid4()}_{filename}"
    object_key = (
        f"csvs/{unique_filename}"
        if storage.using_minio
        else os.path.join(CSV_STORAGE_DIR, unique_filename)
    )
    return storage.upload(csv_bytes, object_key)


def create_csv_record(
    db: Session,
    *,
    filename: str,
    file_ref: str,
    file_size: int,
    row_count: int,
    columns_schema: List[Dict[str, str]],
    column_stats: List[Dict[str, Any]],
    data_group_id: int,
    user_id: int,
) -> CSVFileModel:
    """Profillenmiş ve storage'a yazılmış bir CSV için DB kaydını oluşturur."""
    new_csv = CSVFileModel(
        filename=filename,
        file_path=file_ref,
        file_size=file_size,
        row_count=row_count,
        columns_schema=columns_schema,
        column_stats_cache=column_stats,
        created_at=int(time.time()),
        data_group_id=data_group_id,
        user_id=user_id,
    )
    db.add(new_csv)
    db.commit()
    db.refresh(new_csv)
    return new_csv


def get_or_create_data_group(
    db: Session,
    *,
    user_id: int,
    data_group_id: int | None = None,
    group_name: str | None = None,
    default_name_prefix: str = "CSV Upload",
) -> Tuple[DataGroupModel, bool]:
    """
    Hedef data group'u getirir; verilmemişse yenisini oluşturur.

    Returns:
        (data_group, created) — `created` False ise mevcut bir gruba ekleniyor.
    """
    if data_group_id:
        group = (
            db.query(DataGroupModel)
            .filter(DataGroupModel.id == data_group_id, DataGroupModel.user_id == user_id)
            .first()
        )
        if not group:
            raise ValueError("Data group not found")
        return group, False

    name = group_name or f"{default_name_prefix} - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    group = DataGroupModel(
        name=name,
        source_type="csv",
        user_id=user_id,
        created_at=int(time.time()),
    )
    db.add(group)
    db.commit()
    db.refresh(group)
    return group, True
