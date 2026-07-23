from __future__ import annotations

"""Katman 1 terminus — T1 GÖLGE KOŞU (shadow). Üretim çıktısına DOKUNMAZ.

Ne yapar
────────
Eski ``build_features`` yolu çıktıyı üretmeye devam ederken, orkestratör AYNI ``X_train``
üzerinde paralel koşturulur ve iki sonuç karşılaştırılır. Fark **loglanır, ASLA fırlatılmaz**.

Neden T2'den ayrı bir faz
─────────────────────────
T2'nin bit-birebir sınavı altı yerel CSV'de yapılacak. Gölge koşu bunun kopyası DEĞİL —
**gerçek kullanıcı verisinde** çalışır ve asıl değeri şudur:

  **KAPSAM KAPISINI GERÇEK VERİDE SINAR.** Yerel altı CSV hiç tetiklemediği yolları
  (aggregation, time-series lag, domain pack, timestamp türevleri, kolon yeniden-adlandırma,
  portlanmamış plan op'ları) gerçek projeler tetikler. Soru: bunlar çıktığında
  ``deployable=False`` / ``UnsupportedPipelineScope`` DOĞRU tetikleniyor mu, yoksa beyan
  yüzeyinde bir olgu mu atlanmış? Gölge koşu bunu üretimi riske atmadan cevaplar.

Bu yüzden ``UnsupportedPipelineScope`` ve ``StructuralPreconditionViolation`` burada HATA
değil, **BEKLENEN ve DEĞERLİ BULGU**dur — kapsam kapısının çalıştığının kanıtı. Ayrı
``verdict`` değerleriyle raporlanırlar (``scope_rejected`` / ``precondition_rejected``),
"çöktü" diye değil.

MUTLAK KURAL: bu modül üretim koşusunu BOZAMAZ. Her şey ``try/except`` içinde; en kötü
ihtimalle ``verdict="shadow_failed"`` döner ve koşu eski yolla devam eder.
"""

import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Karşılaştırmada "aynı" sayılacak sayısal tolerans (float64 birikimli hata payı).
_ATOL = 1e-9


def run_shadow_comparison(
    x_train_pre_chain: pd.DataFrame,
    y_train,
    ctx: dict,
    x_train_expected: pd.DataFrame,
) -> dict:
    """Orkestratörü gölgede koştur, eski yolun çıktısıyla karşılaştır. ASLA fırlatmaz.

    ``x_train_pre_chain``: split SONRASI, post-split zincir ÖNCESİ X_train (orkestratörün
    girdisi). Pre-split adımlar (missing-indicator/impute/drop/zscore/ratio/zero-flag) bu
    çerçevede idempotent no-op'tur — kolonlar zaten var, null zaten dolu → T2'nin şekliyle
    birebir aynı kurulum.
    ``x_train_expected``: eski yolun ürettiği nihai X_train (referans).

    Dönen ``verdict``:
      ``identical``            — kolon seti + değerler birebir (aranan sonuç)
      ``value_divergence``     — kolonlar aynı, değerler farklı (İNCELE)
      ``column_divergence``    — kolon setleri farklı (İNCELE)
      ``scope_rejected``       — kapsam kapısı reddetti (BEKLENEN BULGU, hata değil)
      ``precondition_rejected``— yapısal ön-koşul ihlali (BEKLENEN BULGU, hata değil)
      ``shadow_failed``        — gölge koşunun KENDİSİ patladı (üretim etkilenmedi)
    """
    from src.services.inference.orchestrator import (
        StructuralPreconditionViolation, UnsupportedPipelineScope,
        check_pipeline_scope, fit_pipeline,
    )

    rapor: dict = {"verdict": "shadow_failed", "detail": "", "scope": {}}

    # 1) Kapsam kararı — fit patlasa BİLE bunu raporla (gölgenin asıl amacı bu).
    try:
        rapor["scope"] = check_pipeline_scope(ctx)
        if not rapor["scope"].get("deployable"):
            logger.info(f"[shadow] artifact DEPLOY EDİLEMEZ — beyansız olgular: "
                        f"{rapor['scope'].get('undeclared')} | ihlaller: "
                        f"{[v['fact'] for v in rapor['scope'].get('violations') or []]}")
    except Exception as exc:                       # kapsam kontrolü bile patlarsa yut
        rapor["detail"] = f"check_pipeline_scope hatasi: {exc}"
        logger.warning(f"[shadow] kapsam kontrolu basarisiz: {exc}")
        return rapor

    # 2) Orkestratörü koştur — kapsam/ön-koşul reddi BEKLENEN BULGUDUR
    try:
        x_new, artifact = fit_pipeline(x_train_pre_chain.copy(), y_train, ctx)
    except UnsupportedPipelineScope as exc:
        rapor.update(verdict="scope_rejected", detail=str(exc))
        logger.info(f"[shadow] KAPSAM KAPISI reddetti (beklenen davranis, bulgu): {exc}")
        return rapor
    except StructuralPreconditionViolation as exc:
        rapor.update(verdict="precondition_rejected", detail=str(exc))
        logger.info(f"[shadow] YAPISAL ON-KOSUL reddetti (beklenen davranis, bulgu): {exc}")
        return rapor
    except Exception as exc:
        rapor["detail"] = f"fit_pipeline hatasi: {type(exc).__name__}: {exc}"
        logger.warning(f"[shadow] fit_pipeline basarisiz (URETIM ETKILENMEDI): {exc}")
        return rapor

    # 3) Karşılaştır — önce kolon seti, sonra değerler
    try:
        eski, yeni = set(x_train_expected.columns), set(x_new.columns)
        eksik, fazla = sorted(eski - yeni), sorted(yeni - eski)
        rapor.update(n_expected=len(eski), n_shadow=len(yeni),
                     deployable=bool(rapor["scope"].get("deployable")))
        if eksik or fazla:
            rapor.update(verdict="column_divergence", missing=eksik, extra=fazla,
                         detail=f"eksik={eksik} fazla={fazla}")
            logger.warning(f"[shadow] KOLON SAPMASI — eski yolda var/gölgede yok: {eksik} | "
                           f"gölgede var/eskide yok: {fazla}")
            return rapor

        sapan: list = []
        for col in sorted(eski):
            try:
                a = pd.to_numeric(x_train_expected[col], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(x_new[col], errors="coerce").to_numpy(dtype=float)
                if a.shape != b.shape:
                    sapan.append((col, "shape"))
                    continue
                fark = float(pd.Series(abs(a - b)).fillna(0).max()) if len(a) else 0.0
                if fark > _ATOL:
                    sapan.append((col, fark))
            except Exception:                      # tek kolon patlarsa geri kalanı ölç
                sapan.append((col, "karsilastirilamadi"))

        if sapan:
            rapor.update(verdict="value_divergence", diverging=sapan[:20],
                         n_diverging=len(sapan),
                         detail=f"{len(sapan)}/{len(eski)} kolonda deger farki")
            logger.warning(f"[shadow] DEGER SAPMASI: {len(sapan)}/{len(eski)} kolon — "
                           f"ilk 5: {sapan[:5]}")
            return rapor

        rapor.update(verdict="identical", n_manifest=len(artifact.manifest))
        logger.info(f"[shadow] BIREBIR AYNI — {len(eski)} kolon, {len(artifact.manifest)} "
                    f"manifest kalemi, deployable={rapor.get('deployable')}")
        return rapor
    except Exception as exc:                       # karşılaştırma patlarsa da yut
        rapor["detail"] = f"karsilastirma hatasi: {exc}"
        logger.warning(f"[shadow] karsilastirma basarisiz: {exc}")
        return rapor
