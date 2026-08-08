"""TASK 2 harness — replay + label + aggregate + yan-yana kıyas. (proje 27 tip-fix doğrulaması)

KULLANICI RİTMİ: re-train SENDE. Bu harness re-train'i BEKLER (yeni champion ≠ eski b06771 kapısı).

Alt-komutlar:
    compare                       # SALT-OKUNUR — eski/yeni pencere yan-yana (istediğin zaman koşabilir)
    replay                        # yeni champion'a 500 satır (frozen'dan), source=api, backdated
    label                         # aynı 300 pozisyona aynı label'lar (frozen actual_label'dan)
    aggregate                     # run_monitoring_aggregation() (GC YOK)
    all                           # replay → label → aggregate → compare

YAZAN aşamalar ÇİFT KAPI:
    (1) env TASK2_CONFIRM=1  (yoksa yalnız planı basar, hiçbir şey yazmaz)
    (2) champion.id != ESKİ b06771  (re-train yapılmadıysa abort — yeni champion yok)

Windows PowerShell kullanım (proje kökünden ya da herhangi bir yerden — yollar __file__-tabanlı):
    $env:ALLOW_REMOTE_DB=1
    uv run python scripts/task2_harness.py compare                        # salt-okunur
    $env:TASK2_CONFIRM=1; uv run python scripts/task2_harness.py all       # re-train SONRASI tam akış

Bağlantı: ALLOW_REMOTE_DB=1 (DML yeter; DDL YOK → ALLOW_REMOTE_DB_OPS GEREKMEZ).
Frozen girdi: data/_task2_frozen_replay.json (SALT-OKUNUR üretildi, proje kökünde).
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

# Proje kökü = bu dosyanın bir üstü (scripts/ -> kök). cwd'den BAĞIMSIZ → her dizinden koşar.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("ALLOW_REMOTE_DB", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# TÜM modelleri kaydettir (mapper string-ilişkileri) — main.py/worker_main.py konvansiyonu.
for _m in ("mcp", "csv", "data_group", "project", "user", "job", "data_group_query",
           "db_intelligence_cache", "log_entry", "llm_usage", "ai_provider", "ai_model",
           "ai_role_assignment", "password_reset_request", "artifact", "artifact_event",
           "prediction", "monitoring_summary", "token_blacklist"):
    __import__(f"api.models.{_m}")

import pandas as pd  # noqa: E402

from api.core.database import SessionLocal  # noqa: E402
from api.core.serving_cache import get_serving_cache  # noqa: E402
from api.models.artifact import ArtifactModel  # noqa: E402
from api.models.prediction import PredictionModel  # noqa: E402
from api.models.monitoring_summary import MonitoringSummaryModel  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.services.inference.orchestrator import check_raw_input, transform_pipeline  # noqa: E402
# scoring iç fonksiyonlarını AYNEN kullan (davranış birebir): örnekleme kararı + numpy→python
from api.routes.scoring import _sample_input_row, _jsonable  # noqa: E402

PID = 27
OLD_ART = "b06771462ad0402287a5a4ee47f04eb3"
FROZEN = os.path.join(ROOT, "data", "_task2_frozen_replay.json")
MAP = os.path.join(ROOT, "data", "_task2_replay_map.json")

_CONFIRM = os.environ.get("TASK2_CONFIRM") == "1"


def _load_frozen() -> dict:
    with open(FROZEN, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _champion(db):
    """Yeni champion (mirror scoring.py champion lookup). Re-train yapılmadıysa OLD_ART döner → kapı yakalar."""
    champ = (db.query(ArtifactModel)
             .filter(ArtifactModel.project_id == PID,
                     ArtifactModel.is_champion.is_(True),
                     ArtifactModel.deployable.is_(True),
                     ArtifactModel.phase == "complete")
             .first())
    return champ


def _guard_write(champ):
    if champ is None:
        sys.exit("[abort] proje 27 için serve edilebilir champion YOK — önce re-train + promote.")
    if champ.id == OLD_ART:
        sys.exit(f"[abort] champion hâlâ ESKİ artifact ({OLD_ART}) — re-train yapılmamış. "
                 f"Yeni champion ayağa kalkınca tekrar koş.")
    if not _CONFIRM:
        print(f"[plan] YAZMA KAPALI (TASK2_CONFIRM≠1). Yeni champion={champ.id}. "
              f"Onaylamak için: TASK2_CONFIRM=1 ile koş.")
        sys.exit(0)


# ── ADIM 2: REPLAY (mirror scoring.predict; created_at backdated, source=api) ─────────────────
def replay():
    fr = _load_frozen()
    recs = fr["records"]
    db = SessionLocal()
    try:
        champ = _champion(db)
        _guard_write(champ)
        artifact, model = get_serving_cache().get(champ.id)
        required = list((artifact.raw_input_signature or {}).get("required") or [])
        print(f"[replay] yeni champion={champ.id}  required={len(required)} kolon")
        _dropped = sorted(set(recs[0]["input_row"].keys()) - set(required))
        print(f"[replay] frozen satırdan DÜŞEN kolon(lar) (yeni imzada yok): {_dropped}")

        # frozen satırları yeni imzaya PROJEKTE et (fazla kolonları — customer_id — düşür; dinamik)
        proj_rows = [{k: rec["input_row"][k] for k in required if k in rec["input_row"]}
                     for rec in recs]
        raw = pd.DataFrame(proj_rows)

        sig = check_raw_input(raw, artifact.raw_input_signature, artifact.tolerated_extra_columns)
        row_flags = sig.get("row_flags")
        warnings = {"unseen_categories": sig.get("unseen_categories", {}),
                    "out_of_range": sig.get("out_of_range", {})}
        X = transform_pipeline(raw, artifact, verify_input=False, verify_delta=True)

        mref = artifact.model_ref or {}
        threshold = mref.get("optimal_threshold")
        is_clf = hasattr(model, "predict_proba")
        if is_clf:
            proba = model.predict_proba(X)
            classes = list(getattr(model, "classes_", range(proba.shape[1])))
            if proba.shape[1] == 2:
                prob_pos = proba[:, 1]
                thr = threshold if threshold is not None else 0.5
                idx = (prob_pos >= thr).astype(int)
                preds = [_jsonable(classes[i]) for i in idx]
                probs = [float(p) for p in prob_pos]
            else:
                ai = proba.argmax(axis=1)
                preds = [_jsonable(classes[i]) for i in ai]
                probs = [{str(_jsonable(classes[j])): float(proba[r, j]) for j in range(proba.shape[1])}
                         for r in range(proba.shape[0])]
                threshold = None
        else:
            preds = [float(v) for v in model.predict(X)]
            probs = [None] * len(preds)
            threshold = None

        rate = settings.SERVING_INPUT_ROW_SAMPLE_RATE
        bk_col = getattr(artifact, "business_key_column", None)
        index_to_pid = {}
        for i, rec in enumerate(recs):
            pid_new = uuid.uuid4().hex
            index_to_pid[i] = pid_new
            sampled = _sample_input_row(pid_new, rate)
            flags = ({"has_unseen": bool(row_flags["has_unseen"][i]),
                      "has_out_of_range": bool(row_flags["has_out_of_range"][i])}
                     if row_flags else None)
            # business_key: mirror scoring → GÖNDERİLEN (projekte) satırdan. bk_col genelde customer_id
            # ve o düştüğü için NULL olur = ÜRETİM gerçeği (yeni champion customer_id kabul etmez). v1 beklenen.
            bk_val = proj_rows[i].get(bk_col) if bk_col else None
            db.add(PredictionModel(
                prediction_id=pid_new, project_id=PID, artifact_id=champ.id,
                created_at=int(rec["created_at"]),          # BACKDATE = eski pencere günü (yan-yana kıyas)
                source="api",
                input_row=(proj_rows[i] if sampled else None),
                input_row_sampled=sampled, prediction=preds[i], probability=probs[i],
                threshold=threshold, input_warnings=warnings, input_warning_flags=flags,
                business_key=(str(bk_val) if bk_val is not None else None),
            ))
        db.commit()
        with open(MAP, "w", encoding="utf-8") as fh:
            json.dump({"new_artifact_id": champ.id, "index_to_pid": index_to_pid}, fh)
        print(f"[replay] {len(recs)} yeni tahmin yazıldı (artifact={champ.id}, source=api, "
              f"backdate=eski gün). map → {MAP}")
    finally:
        db.close()


# ── ADIM 2b: LABEL (aynı 300 pozisyon, aynı label — frozen'dan; mirror labels._apply_labels) ──
def label():
    fr = _load_frozen()
    recs = fr["records"]
    with open(MAP, "r", encoding="utf-8") as fh:
        mp = json.load(fh)
    index_to_pid = {int(k): v for k, v in mp["index_to_pid"].items()}
    db = SessionLocal()
    try:
        champ = _champion(db)
        _guard_write(champ)
        now = time.time()
        pairs = {}                                    # yeni_pred_id → label (yalnız eski etiketli pozisyonlar)
        for i, rec in enumerate(recs):
            if rec["actual_label"] is not None:
                pairs[index_to_pid[i]] = rec["actual_label"]
        ids = list(pairs.keys())
        matched = 0
        for j in range(0, len(ids), 500):
            chunk = ids[j:j + 500]
            rows = (db.query(PredictionModel)
                    .filter(PredictionModel.project_id == PID,
                            PredictionModel.prediction_id.in_(chunk)).all())
            for r in rows:
                r.actual_label = pairs[r.prediction_id]
                r.label_backfilled_at = now
                matched += 1
        db.commit()
        print(f"[label] aynı {len(pairs)} pozisyona aynı label yazıldı (eşleşen={matched}). "
              f"(re-randomize YOK — frozen actual_label pozisyon-eşlemesi)")
    finally:
        db.close()


# ── ADIM 2c: AGGREGATE (GC YOK) ───────────────────────────────────────────────────────────────
def aggregate():
    db = SessionLocal()
    try:
        _guard_write(_champion(db))
    finally:
        db.close()
    from src.services.monitoring.aggregator import run_monitoring_aggregation
    n = run_monitoring_aggregation()
    print(f"[aggregate] {n} özet satırı yazıldı/güncellendi (GC KOŞMADI).")


# ── ADIM 3: COMPARE (SALT-OKUNUR — eski/yeni pencere yan yana) ─────────────────────────────────
def _fmt(x, w=10):
    if x is None:
        return " " * (w - 1) + "-"
    if isinstance(x, float):
        return f"{x:>{w}.4f}"
    return f"{str(x):>{w}}"


def compare():
    db = SessionLocal()
    try:
        rows = (db.query(MonitoringSummaryModel)
                .filter(MonitoringSummaryModel.project_id == PID)
                .order_by(MonitoringSummaryModel.window_start,
                          MonitoringSummaryModel.artifact_id).all())
    finally:
        db.close()
    old = next((r for r in rows if r.artifact_id == OLD_ART), None)
    new = next((r for r in rows if r.artifact_id != OLD_ART), None)
    print("\n=== TASK 2 — YAN YANA KIYAS (proje 27) ===")
    if old is None:
        print("[!] ESKİ pencere (b06771) bulunamadı.")
    if new is None:
        print("[!] YENİ pencere yok — replay+aggregate henüz koşmadı (ya da champion hâlâ eski).")
    print(f"\n{'':28}{'ESKİ (b06771)':>18}{'YENİ':>18}")

    def line(label, ov, nv):
        print(f"{label:28}{_fmt(ov,18)}{_fmt(nv,18)}")
    for lbl, attr in [("artifact_id", "artifact_id"), ("window_start", "window_start"),
                      ("n_total", "n_total"), ("n_sampled", "n_sampled"),
                      ("overall_status", "overall_status"), ("psi_overall", "psi_overall"),
                      ("psi_max", "psi_max"), ("psi_degenerate_count", "psi_degenerate_count"),
                      ("cbpe_gate_status", "cbpe_gate_status"),
                      ("realized_gate_status", "realized_gate_status"),
                      ("label_coverage", "label_coverage")]:
        line(lbl, getattr(old, attr, None) if old else None,
             getattr(new, attr, None) if new else None)
    # realized_metrics (roc/accuracy)
    orm = (old.realized_metrics or {}) if old else {}
    nrm = (new.realized_metrics or {}) if new else {}
    for k in ("roc_auc", "accuracy", "positive_rate_deviation"):
        line(f"realized.{k}", orm.get(k), nrm.get(k))
    # ── per-feature PSI yan yana ──
    opf = (old.psi_per_feature or {}) if old else {}
    npf = (new.psi_per_feature or {}) if new else {}
    feats = sorted(set(opf) | set(npf))
    print(f"\n{'PER-FEATURE PSI':28}{'ESKİ':>12}{'YENİ':>12}  not")
    for f in feats:
        o = opf.get(f)
        n = npf.get(f)
        op = (o or {}).get("psi") if o else None
        npv = (n or {}).get("psi") if n else None
        note = []
        if f not in npf:
            note.append("YENİDE YOK (imzadan düştü)")
        if f not in opf:
            note.append("YALNIZ yeni")
        if o and o.get("degenerate"):
            note.append(f"eski-degenerate:{o.get('reason')}")
        if n and n.get("degenerate"):
            note.append(f"yeni-degenerate:{n.get('reason')}")
        print(f"{f:28}{_fmt(op,12)}{_fmt(npv,12)}  {', '.join(note)}")
    print("\nBEKLENEN: products_number/active_member/tenure PSI DÜŞER · country+age KALIR · "
          "customer_id YENİDE YOK (imzadan+PSI'dan düştü).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if cmd == "replay":
        replay()
    elif cmd == "label":
        label()
    elif cmd == "aggregate":
        aggregate()
    elif cmd == "compare":
        compare()
    elif cmd == "all":
        replay()
        label()
        aggregate()
        compare()
    else:
        sys.exit(f"bilinmeyen komut: {cmd} (compare|replay|label|aggregate|all)")
