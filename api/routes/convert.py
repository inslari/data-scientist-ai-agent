"""
api/routes/convert.py
---------------------
Dosya format dönüşümü uçları (ilk dönüştürücü: Excel → CSV).

Uçlar dönüştürücüye özel değildir; hepsi `api/core/converters` registry'si
üzerinden çalışır. Yeni bir dönüşüm eklendiğinde bu dosyaya dokunmak gerekmez.

    GET  /convert/converters   → kayıtlı dönüşümler + ayar şemaları
    POST /convert/inspect      → kaynak dosyanın yapısı (örn. Excel sayfaları)
    POST /convert/run          → dönüşümü çalıştır; çıktıyı indir ya da
                                 doğrudan Veri Varlıkları'na kaydet
"""

import io
import json
import zipfile
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy.orm import Session

from api.core.converters import ConversionError, get_converter, list_converters
from api.core.csv_ingest import (
    create_csv_record,
    get_or_create_data_group,
    profile_csv_bytes,
    store_csv_bytes,
)
from api.core.database import get_db
from api.core.deps import get_current_user
from api.core.jobs import enqueue_ttd_suggestions, enqueue_viz_suggestions
from api.core.storage import storage
from api.models.user import UserModel
from api.routes.data_group import _clear_db_cache
from api.schemas.convert import (
    ConvertInspectResponse,
    ConvertToAssetsResponse,
    ConvertersResponse,
)

from src.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/convert", tags=["File Conversion"])

MAX_FILES_PER_REQUEST = 10


def _require_converter(converter_id: str):
    converter = get_converter(converter_id)
    if not converter:
        raise HTTPException(status_code=404, detail=f"Unknown converter: {converter_id}")
    return converter


def _check_source(converter, filename: str, size: int) -> None:
    if not converter.accepts(filename):
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is not supported by this converter "
                   f"(expected: {', '.join(converter.source_extensions)})",
        )
    limit = converter.max_file_mb * 1024 * 1024
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"'{filename}' exceeds the {converter.max_file_mb} MB limit.",
        )


def _parse_options(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`options` must be a valid JSON object.")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="`options` must be a JSON object.")
    return parsed


def _download_headers(filename: str) -> Dict[str, str]:
    """
    İndirme başlıkları. ASCII fallback + RFC 5987 ile Türkçe karakterli dosya
    adları korunur. `Access-Control-Expose-Headers` şart: onsuz tarayıcı
    Content-Disposition'ı JS'e vermez ve dosya yanlış adla iner.
    """
    ascii_name = filename.encode("ascii", errors="replace").decode("ascii").replace("?", "_")
    return {
        "Content-Disposition": (
            f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
        ),
        "Access-Control-Expose-Headers": "Content-Disposition",
    }


@router.get("/converters", response_model=ConvertersResponse)
def get_converters(current_user: UserModel = Depends(get_current_user)):
    """Kayıtlı tüm dönüştürücüleri ve ayar şemalarını döndürür."""
    return {"status": "success", "data": list_converters()}


@router.post("/inspect", response_model=ConvertInspectResponse)
async def inspect_source(
    file: UploadFile = File(...),
    converter_id: str = Form(...),
    options: str = Form(None),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Dönüşüm öncesi keşif: kaynak dosyanın yapısını döndürür (Excel için
    sayfa listesi, CSV için tespit edilen ayarlar + önizleme). Hiçbir şey
    kalıcı olarak yazılmaz.

    `options` verilirse keşif o ayarlarla yapılır — `live_preview` açık olan
    dönüştürücülerde arayüz her ayar değişiminde burayı yeniden çağırır.
    """
    converter = _require_converter(converter_id)
    filename = file.filename or ""
    contents = await file.read()
    _check_source(converter, filename, len(contents))
    opts = _parse_options(options)

    try:
        data = await run_in_threadpool(converter.inspect, contents, filename, opts)
    except ConversionError as e:
        return {"status": "error", "filename": filename, "message": str(e)}
    except Exception as e:
        logger.warning(f"[convert] inspect başarısız ({converter_id}/{filename}): {e}")
        return {"status": "error", "filename": filename, "message": f"Failed to inspect file: {e}"}

    return {"status": "success", "filename": filename, "data": data}


@router.post("/run")
async def run_conversion(
    files: List[UploadFile] = File(...),
    converter_id: str = Form(...),
    options: str = Form(None),
    output: str = Form("download"),          # "download" | "assets"
    group_name: str = Form(None),
    data_group_id: int = Form(None),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Dönüşümü çalıştırır.

    output="download" → tek çıktı ise dosyanın kendisi, birden çok çıktı ise ZIP.
    output="assets"   → çıktılar CSV varlığı olarak kaydedilir (yalnızca CSV
                        üreten dönüştürücülerde).
    """
    converter = _require_converter(converter_id)
    if output not in ("download", "assets"):
        raise HTTPException(status_code=400, detail="`output` must be 'download' or 'assets'.")
    if output == "assets" and converter.target_extension != ".csv":
        raise HTTPException(
            status_code=400,
            detail="This converter's output cannot be saved to data assets.",
        )
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_FILES_PER_REQUEST} files can be converted per request.",
        )

    opts = _parse_options(options)

    # ------------------------------------------------------------------
    # 1) Dönüşüm (CPU-bound → threadpool)
    # ------------------------------------------------------------------
    produced: List[tuple] = []  # (source_filename, ConvertedFile)
    errors: List[Dict[str, str]] = []

    for upload in files:
        filename = upload.filename or ""
        contents = await upload.read()
        _check_source(converter, filename, len(contents))

        try:
            results = await run_in_threadpool(converter.convert, contents, filename, opts)
        except ConversionError as e:
            errors.append({"filename": filename, "error": str(e)})
            continue
        except Exception as e:
            logger.warning(f"[convert] dönüşüm başarısız ({converter_id}/{filename}): {e}")
            errors.append({"filename": filename, "error": f"Conversion failed: {e}"})
            continue

        produced.extend((filename, item) for item in results)

    # Birden çok kaynak dosya aynı çıktı adını üretebilir (örn. iki kitapta da
    # tek sayfa) — ZIP içinde ve varlık listesinde çakışmayı önle.
    seen_names: Dict[str, int] = {}
    for _, item in produced:
        count = seen_names.get(item.filename, 0)
        seen_names[item.filename] = count + 1
        if count:
            stem, _, ext = item.filename.rpartition(".")
            item.filename = f"{stem}_{count + 1}.{ext}" if stem else f"{item.filename}_{count + 1}"

    if not produced:
        detail = errors[0]["error"] if errors else "Conversion produced no output."
        if output == "download":
            raise HTTPException(status_code=400, detail=detail)
        return {
            "status": "error",
            "message": detail,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # 2a) İndirme — tek dosya doğrudan, çoklu çıktı ZIP
    # ------------------------------------------------------------------
    if output == "download":
        if len(produced) == 1:
            _, item = produced[0]
            return Response(
                content=item.content,
                media_type=item.media_type,
                headers=_download_headers(item.filename),
            )

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, item in produced:
                zf.writestr(item.filename, item.content)
        # Tek kaynak dosyadan çok çıktı çıktıysa ZIP adı o dosyadan türetilir.
        sources = {src for src, _ in produced}
        zip_name = (
            f"{next(iter(sources)).rsplit('.', 1)[0]}.zip"
            if len(sources) == 1
            else f"{converter.id}_output.zip"
        )
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers=_download_headers(zip_name),
        )

    # ------------------------------------------------------------------
    # 2b) Veri Varlıkları'na kaydet — CSV yükleme yoluyla aynı ingest hattı
    # ------------------------------------------------------------------
    try:
        data_group, created = get_or_create_data_group(
            db,
            user_id=current_user.id,
            data_group_id=data_group_id,
            group_name=group_name,
            default_name_prefix="Converted",
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Data group not found")

    saved = []
    converted_info = []

    for source_filename, item in produced:
        file_ref = await run_in_threadpool(store_csv_bytes, item.content, item.filename)
        try:
            row_count, columns_schema, column_stats, _ = await run_in_threadpool(
                profile_csv_bytes, item.content
            )
        except Exception as e:
            storage.delete(file_ref)
            errors.append({"filename": item.filename, "error": f"Failed to parse converted CSV: {e}"})
            continue

        record = create_csv_record(
            db,
            filename=item.filename,
            file_ref=file_ref,
            file_size=len(item.content),
            row_count=row_count,
            columns_schema=columns_schema,
            column_stats=column_stats,
            data_group_id=data_group.id,
            user_id=current_user.id,
        )
        saved.append(record)
        converted_info.append({
            "filename": item.filename,
            "size": len(item.content),
            "rows": item.meta.get("rows"),
            "columns": item.meta.get("columns"),
            "sheet": item.meta.get("sheet"),
            "source_filename": source_filename,
        })

    if not created and saved:
        _clear_db_cache(db, data_group)

    # Yeni grup oluştuysa Talk-to-Data / görselleştirme önerilerini arka planda üret.
    if created and saved:
        enqueue_ttd_suggestions(db, data_group.id)
        enqueue_viz_suggestions(db, data_group.id)

    return ConvertToAssetsResponse(
        status="success" if saved else "error",
        message=(
            f"{len(saved)} file(s) converted and saved."
            + (f" ({len(errors)} errors)" if errors else "")
        ),
        data_group_id=data_group.id if saved else None,
        data=saved,  # type: ignore[arg-type]
        converted=converted_info,  # type: ignore[arg-type]
        errors=errors,
    )
