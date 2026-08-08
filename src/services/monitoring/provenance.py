"""Katman 2 Parça 3b — importance provenance: transformed feature → raw kolon, MANIFEST'ten (regex DEĞİL).

Manifest her türev adımın output←source eşlemesini `frozen_choice.selection`'da KAYITLI taşır
(ratio: {output, numerator, denominator}; missing/zero_flag: {source, output}; groupby: {group, value,
emits}); onehot/count `param_store` + `delta.added`'tan. Ölçüldü: 3 dataset'te %100 atıf, orphan %0
(regex proxy'nin %4-24'ü araç sınırıymış). resolve_to_raw recursive + CYCLE-GUARD, terminal SIKI
(`raw_input_signature.required` üyesi).
"""
from __future__ import annotations


def build_producer_map(manifest: list) -> dict:
    """output_col -> [doğrudan kaynak kolonlar]. Öncelik `frozen_choice.selection` (kayıtlı provenance)."""
    producer: dict = {}
    for e in manifest or []:
        sel = (e.get("frozen_choice") or {}).get("selection")
        if isinstance(sel, list):
            for it in sel:
                if not isinstance(it, dict):
                    continue
                out = it.get("output")
                if out and "numerator" in it and "denominator" in it:
                    producer[out] = [it["numerator"], it["denominator"]]   # ratio: 2 kaynak
                elif out and "source" in it:
                    producer[out] = [it["source"]]                          # missing_indicator/zero_flag
                if "emits" in it:                                           # groupby: (group,value)→çok output
                    srcs = [s for s in (it.get("group"), it.get("value")) if s]
                    for o in it["emits"]:
                        producer[o] = srcs
        # plan derived (config.specs) — chained çok-hop; selection taşımayan üreticiler
        for spec in ((e.get("config") or {}).get("specs") or []):
            if isinstance(spec, dict):
                nm = spec.get("name") or spec.get("output")
                ins = (spec.get("inputs") or spec.get("sources")
                       or ([spec["source"]] if spec.get("source") else []))
                if nm and ins:
                    producer.setdefault(nm, [c for c in ins if c])
    return producer


def _producer_from_params(manifest, param_store, producer):
    """onehot (`<src>_<val>`) + count (`count_<col>`) — selection yerine params+delta.added'tan."""
    oh_cols = ((param_store or {}).get("onehot_encode") or {}).get("columns") or []
    for e in manifest or []:
        for col in e.get("delta", {}).get("added", []):
            if col in producer:
                continue
            if col.startswith("count_"):
                producer[col] = [col[len("count_"):]]
            else:
                cand = [c for c in oh_cols if col == c or col.startswith(c + "_")]
                if cand:
                    producer[col] = [max(cand, key=len)]                    # en uzun eşleşen kaynak
    return producer


def resolve_to_raw(col: str, producer: dict, raw: set, seen: "set | None" = None) -> set:
    """col'u raw kolonlara indir (recursive). TERMINAL SIKI: `raw` = raw_input_signature.required üyesi.

    Ne raw ne producer'da → boş küme (ORPHAN; sessizce raw SAYILMAZ). CYCLE-GUARD (`seen`): bozuk/
    zincirli manifest döngü kursa da sonsuz recursion yok."""
    if col in raw:
        return {col}
    seen = seen if seen is not None else set()
    if col in seen or col not in producer:
        return set()
    seen.add(col)
    out: set = set()
    for s in producer[col]:
        out |= resolve_to_raw(s, producer, raw, seen)
    return out


def attribute_importance(feature_importance: dict, manifest: list, raw_required, param_store=None):
    """transformed importance {feat: val} → (raw_imp_fraction {raw_col: pay}, attributed_ratio).

    ÇOK-KAYNAKLI feature'da importance EŞİT bölünür: ratio(a,b) → val/2 her birine. Kısmi-türev tarzı
    atıf (her raw'ın çıktıya marjinal katkısı) daha doğru olurdu AMA model-agnostik DEĞİL (her model
    ailesi için ayrı Jacobian/SHAP) ve ölçüm maliyeti yüksek. Sıralama GÖRELİ → eşit bölme sıralamayı
    bozmuyor (ölçüldü: makul top-N). raw_imp TOPLAM importance'a NORMALİZE (model payı). attributed_ratio
    = eşlenebilen kütle / toplam (orphan = 1 − bu; bugün ~0 ama chained/egzotik feature ileride orphan üretebilir)."""
    fi = feature_importance or {}
    total = float(sum(fi.values()))
    if not fi or total <= 0:
        return {}, None
    raw = set(raw_required or [])
    producer = build_producer_map(manifest)
    _producer_from_params(manifest, param_store, producer)

    raw_imp: dict = {}
    attributed = 0.0
    for feat, val in fi.items():
        srcs = resolve_to_raw(feat, producer, raw)
        if not srcs:
            continue                                       # ORPHAN — attributed'a GİRMEZ (sıkı terminal)
        attributed += float(val)
        share = float(val) / len(srcs)
        for rc in srcs:
            raw_imp[rc] = raw_imp.get(rc, 0.0) + share
    return {c: v / total for c, v in raw_imp.items()}, attributed / total
