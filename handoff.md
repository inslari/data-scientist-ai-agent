# Handoff — DS Agent

**Branch:** `feat/clear-reporting`
**Son güncelleme:** 2026-07-23
**Aktif iş:** Katman 1 (inference artifact) — deployment + monitoring temeli

> ## ⏩ BURADAN DEVAM ET (sonraki sohbet)
>
> ### 🟢 DURUM (tek cümle)
> **KAPSAM BORCU KAPANDI** — üretim-kanıtı guard'ı 6 dataset'te de **never_produced = 0**, `produced_then_dropped`
> sepeti %100 açıklanmış ölçüm artefaktı (§2.15-2.18). Yolda 4. bir kök keşfedildi (hedef/id/timestamp drop'unun
> portlanmamış olması → seçim sapması + target-leakage yüzeyi). **Terminus'a hazırlık ölçütü 4/4 sağlandı.**
> Guard artık SİMETRİK (eksik üretim + fazla mevcudiyet) ve eski yolun YAPI kaynaklı garantileri açık ön-koşula
> çevrildi (§2.19) — yolda 2 sızıntı yüzeyi daha ölçüldü (benzersiz ID → target-encode = y'nin kopyası; dedup'sız
> çerçeve → tüm fitted_stat'lar kayar). `tests/test_layer1_impute_parity.py` = **80/80 yeşil**; tüm `tests/` =
> 712 passed / **7 failed** (9'du — 2 bayat test düzeltildi). Eski `feature_builder` HÂLÂ DOKUNULMADI (additive).
> **Commit'ler sende bekliyor** (sen atarsın).
>
> ### ⚠️ İDDİA DÜZELTMESİ (önceki handoff abartıyordu — kayda geçti)
> Önceki sürüm "TÜM adım-portları BİTTİ" diyordu. Bu, **post-split zincir için** doğruydu; ama mevcut parity
> harness'ında `fit_pipeline` ile `transform_pipeline` **AYNI girdi sözleşmesini** gördüğü için eksik port ikisinde
> birden eksik kalıyor → **fark doğmuyor, test yeşil kalıyordu.** 50/50 yeşil olması kapsamı kanıtlamıyordu.
> Guard (§2.15) tam da bu kör noktayı kapatmak için yazıldı.
>
> ### 📌 BU SESSION'DA NE YAPILDI (2026-07-23) — 37→50 test
> Önceki session orkestratör iskelesini bırakmıştı (37/37). Bu session Katman 1'in KALAN tüm adımlarını ekleyip
> uçtan-uca parity ağını kurdu. Her biri **tasarım→onay→yaz→doğrula** ritmiyle, ayrı ayrı:
> 1. **GroupbyDeviation** (§2.11, 37→40) — dev/zdev/grpstd; fitted_stat; seçim `records`'a gömülü; unseen→GLOBAL.
> 2. **SelectDrop** (§2.12, 40→44) — TERMINUS #3; `frozen_choice`; **fit DELEGE** (feature_builder selection dizisi,
>    kullanıcı `AskUserQuestion` onayı — reimplement değil), transform pür-drop + terminal fillna.
> 3. **ZScore/Normalize** (§2.13, 44→48) — **#5 fix**; plan-güdümlü, train-fit; eski leaky'den bilerek sapar →
>    formül-parity + deterministik sızıntı-delta (ROC A/B değil); additive; tam dinamik (kullanıcı özel istedi).
> 4. **Uçtan-uca gerçek-veri parity harness** (§2.14, 48→50) — TERMINUS ÖN-KOŞULU; `fit_pipeline == build_features
>    zinciri` credit(cls)+housing(reg) gerçek verisinde birebir + serving yolu. `_old_oracle_sequence` config-tam yapıldı.
>
> **Bu session'ın ana kazanımı:** Katman 1 adım-portlama FAZI KAPANDI; sıradaki faz (terminus) artık körlemesine değil,
> parity ağı üstünde güvenle yapılabilir. Yol boyunca yakalanan 2 gerçek incelik terminus'a taşındı (aşağıda).
>
> ### ⏭️ SIRADAKİ ADIM: T2 — YAPISAL CUTOVER (T0 ✅, T1 ✅)
> T1 yerel 6 dataset'te `identical` verdi ve kapsam kapısı gerçek pipeline'da sınandı (§2.21). **Kullanıcı
> tarafında kalan tek T1 işi:** birkaç GERÇEK proje koşusunu `FEATURE_PIPELINE_MODE=shadow` ile çalıştırmak
> (aggregation/ts-lag/domain-pack yolları yerel setlerde tetiklenmiyor) → flag kapısı (a) böyle kapanır.
> Sonra **T2**: `build_features` post-split zinciri bırakır, `fit_pipeline`/`transform_pipeline` çağırır;
> `_clean_data` YERİNDE kalır → metrik-NÖTR. **Önce determinizm kapısı, sonra bit-birebir şartı** — sapma varsa
> porting bug'ıdır, T3'e GEÇİLMEZ. Ayrıca §2.18'in **TERMINUS GÖREVİ** notu (T0'da KAPANDI):
> `build_features` `derived_flags`/`auto_features`'ı filtreli yayınlıyor → frozen kaynağı olarak yetersiz,
> `ratio_feature_meta` gibi filtrelenmemiş kayıt gerekiyor. Her yapısal değişiklikten sonra
> `scripts/layer1_coverage_inventory.py` yeniden koşulmalı (iki sepetli tablo = objektif ölçüt).
>
> ### ⏭️ SONRA: TERMINUS (Katman 1 kapanışı)
> `build_features`'ı **önce-split + orkestratör çağıracak** şekilde yeniden-yapılandır (üretim ML-yolu — İLK kez feature_builder
> değişir → tasarım→onay ritmiyle gir). **Harness'ın KANITLAMADIĞI, terminus'ta ele alınacak 2 ŞEY:**
> - (1) `_clean_data`'nın **dedup + currency** kısmı HENÜZ PORT EDİLMEDİ (ImputeStep yalnız impute; harness bunu upfront
>   `_clean_data` ile nötrledi). Terminus'ta: ya pre-split tut ya ayrı adım yap.
> - (2) impute/zscore'un **pre-split→post-split RELOKASYONU** = #4/#5 fiksinin fiili DEPLOY'u → train-dist≠full-dist veride
>   ML sonucu DEĞİŞİR → **offline_report regresyon testi ŞART** (tek-koşu güvenilmez; CLAUDE.md ±0.004 ROC gürültüsü).
>
> Terminus'tan SONRA: Artifact boş alanları (`raw_input_signature`/`training_reference`/`unseen_category_policy`/`model_ref`)
> → **serialize katmanı** (Artifact→disk, `pickle_io` çapası) → deployment guard + model registry → Katman 2 monitoring. Bkz. §5.
>
> **Ritim (değişmez):** tasarım → onay → yaz → doğrula. **ASLA commit etme** — kullanıcı atar.
> Adım-adım detay §2.6-2.14. Yol haritası §5. Kalıcı özet: memory `layer1-inference-artifact`.

---

## 1. Proje durumu özeti

İki büyük iş akışı:

1. **Rapor denetimi (Faz 1-3) — KAPANDI.** ~20 madde + 3 leak, katmanlı rapor
   (headline→explanation→evidence) mimarisi. Faz 3 (jargon glossları, ondalık disiplini,
   MCC uçları, editoryal-fluff guard, başlık tutarlılığı) churn + credit üzerinde TR+EN
   doğrulandı, sıfır ham-ondalık sızıntı, TR-leak guard yeşil. **Commit sende bekliyor.**

2. **Katman 1 — inference artifact (AKTİF, tasarım bitti, kod başladı).** Eğitim sonunda
   öğrenilen tüm fitted-state'i tek taşınabilir `Artifact`'e toplayıp prodüksiyonda tek ham
   satırı eğitimle BİREBİR AYNI `transform` yolundan geçirmek. Monitoring (katman 2) buna
   bağımlı, o yüzden önce bu.

---

## 2. Yapılan işler ve kararlar

### 2.1 Rapor denetimi (Faz 3) — tamamlanan
- `src/core/report_schema.py`: `_fmt_value` magnitude-tabanlı ondalık (≥1000 binlik ayraç
  2dp, ≥1 → 3dp, <1 → 3dp); `_METHOD_GLOSS` (bootstrap/wilson/fisher-z parantez-içi sade dil).
- `src/services/report_sections.py`: MCC yalnız iki uç (0 rastgele / 1 kusursuz / −1 ters),
  OOF açıkça yazıldı, 14-model açıklaması tek satır, `_EDITORIAL_FLUFF_PATTERNS` (bağlam-duyarlı:
  "robust dataset" yakalanır, "robust standard errors" yakalanmaz), tüm not-gömülü sayılar
  `_fmt_value`'ya bağlandı.
- `src/services/risk_assessment.py`: DI/brr/amp → `_fmt_value`.
- Testler: `test_faz3_number_format.py`, `test_faz3_jargon_gloss.py` (yeni); `test_synthesis_layered`,
  `test_performance_layered` (güncellendi). Hepsi yeşil.

### 2.2 Katman 1 — envanter + tasarım (kod öncesi turlar)
İki ölçüm turu (kod yazmadan) ile mevcut pipeline haritalandı:

- **Fitted-state envanteri:** `build_features` fit+transform'u tek monolitik fonksiyonda
  iç içe yapıyor. `_*_tt` fonksiyonları `(X_train, X_test)` alıp train'e fit + ikisine uygula
  yapıyor ama **fitted parametreleri return ETMİYOR** → param'lar transform sonrası çöpe atılıyor.
  Persist edilen tek şey: final model (`joblib` → `data/models/model_<uuid>.pkl`), `final_features`,
  `ratio_feature_meta`, eşik+provenance. **Tüm transform parametreleri persist EDİLMİYOR** — Katman 1'in kapatacağı boşluk.
- **Görülmemiş-kategori davranışı:** hepsi graceful (one-hot→0, target→prior, count→0, groupby→global).
  Çökme yok. Ama koda gömülü; artifact açık politika olarak taşıyacak.
- **Serialize altyapısı:** `joblib` var ama yalnız final model için. Transform-state için sıfırdan.
- **Refactor kararı = A (tam bölme), B DEĞİL.** Her `_tt` → `fit(X_train)→params` /
  `transform(X, params)` ikilisine. Gerekçe: inference ile eğitim aynı `transform`'u çağırsın
  → training/serving skew YAPISAL olarak imkânsız. B (sarmalayıcı) iki kod yolu bırakır ("biri
  güncellendi öteki unutuldu" sınıfı). `_fit_ratio_stats`/`_apply_ratio_stats` zaten şablon.
- **Refactor maliyeti (ölçüldü):** ~12 fit-taşıyan aşama. Zorluk: 4 persist-only (ratio-stats +
  TargetEncoder + 2 scaler), 7 kolay (liste/dict dondur), 4 orta (clean_data ayrıştır, groupby-dev,
  one-hot, zscore/normalize op). Gerçekten "zor" tekil transform YOK; asıl parça **orkestratör**.

### 2.3 Beş sızıntı (split-öncesi tüm-df istatistiği)
Değer-sızıntısı (feature değerine gömülü, hepsi aynı sınıf — train-only olmalı):
1. **#4 (bilinen):** `_clean_data:339-350` median/mode impute, tüm-df, split öncesi.
2-3. **#5 (YENİ bulundu):** `_FEATURE_OPS` `zscore` (`x.mean()/x.std()`) ve `normalize`
   (`x.min()/x.max()`) tüm-df'den — LLM planı bu op'ları seçerse tetiklenir. `feature_builder.py:1577,1581`.

Daha hafif (global skaler / seçim, değere test satırı gömmüyor): `_aggregate_if_needed`
reference_date/mid quantile'ları (RFM yolu); ratio çift-seçimi + missing-indicator top-30 +
zero-flag prevalence (feature-selection leakage sınıfı, metriğe etkisi minimal).

**Fix yönü:** A refactor'unda impute + zscore/normalize `fit(X_train)→param` / `transform(X,param)`'a
taşınır; tüm-df istatistik yolu hiçbir imzada kalmaz.

### 2.4 Artifact şeması (1.1) — onaylandı
`Artifact` alanları: `step_manifest` (o koşuda fiilen çalışan adımlar, sıralı; koşullu → run-başına),
`param_store` (inline JSON vs pickle-ref, sklearn versiyonuyla), `raw_input_signature` (domain-aware,
kademeli katılık), `unseen_category_policy`, `training_reference` (PSI+CBPE baseline), `model_ref`
(registry: uuid+proje+versiyon), `final_features`.

Kritik kararlar:
- **Manifest granülaritesi = mantıksal işlem, kaynak fonksiyon DEĞİL.** `_clean_data` → 4 ayrı adım
  (cast=rowwise, drop_duplicates=skip, currency=parametric, impute=parametric).
- **Type taksonomisi:** `rowwise` (state yok) / `parametric` (`param_kind`: `fitted_stat` = dağılım
  sayısı, leakage-hassas / `frozen_choice` = yapısal karar) / `skip` (train-only, inference'ta yok).
- **Şema düzeltilmiş hedefi tanımlar** (zscore parametric, impute train-fit) — mevcut sızıntılı hali değil.
- **Drop listeleri (#1):** `re_derive: false` — inference donmuş kümeyi uygular, yeniden türetmez →
  sıra-bağımsız (kod teyidi: dört selection adımı da saf kolon-drop). Kommütatiflik gerekmez bile.
- **İmza katılığı (#2):** kademeli — `required` (eksik/tip-uyumsuz → ABORT) + `ignored`/`extra`
  (tolere+logla). Domain-aware: binary/categorical dar tolerans + değer-domain kontrolü
  (`HasCrCard=0.7` → ABORT), continuous geniş + aralık-uyarısı.
- **Selection terminal invariant (#3):** `select_drop` en yüksek order; hiçbir fitted_stat ondan
  sonra gelemez (kod teyidi: scale=son fitted_stat, sonrası yalnız drop+sabit fillna).
- **final_features hizası:** `run_all_then_reindex` — tüm adımları koş, sonra `reindex(fill_value=0)`
  (mevcut `_encode_categorical_tt` mekanizmasının aynısı).

### 2.5 fit/transform sözleşmesi (1.2) — onaylandı, imza düzeyinde
- Ortak `TransformStep` protokolü: `fit(X_train)→params`, `transform(X, params)→X'`.
- parametric+rowwise tek protokol (rowwise'da fit None döner); skip ayrı `TrainOnlyStep`
  (`apply_train`, transform YOK).
- Orkestratör: `fit_pipeline(df)→(X, artifact)` + `transform_pipeline(raw_row, artifact)→X_ready`;
  ikisi de AYNI `step.transform`'u çağırır → skew imkânsız.
- unseen_category fit-içi toplanır (vocab zaten öğreniliyor); training_reference ayrı adanmış geçiş
  (ham train snapshot transform-öncesi + evaluation'dan baseline metrik kopyası).

### 2.6 Kod — YAZILDI (commit sende, her adım ayrı parçalar)
Konum: `src/services/inference/` (paralel paket, `feature_builder` DOKUNULMUYOR). Her adım
`TransformStep` sözleşmesini uygular, SELF-CONTAINED (feature_builder import ETMEZ — inference
runtime yalın kalmalı), eski `_*_tt`'yi REPLİKA eder, parity eski-yola karşı doğrular.

- **protocol.py + __init__.py:** TransformStep/TrainOnlyStep protokolleri, tip alias'ları, `Artifact`.
- **ImputeStep** (parametric/fitted_stat, **#4 train-fit**). `numeric_median` (uygulanan, int yuvarlanmış)
  + `numeric_median_raw` (ham) ayrı — İncelik 2. A/B koşuldu → etki **0** (medyanlar train==full).
- **RatioFinalizeStep** (parametric/fitted_stat). `_fit_ratio_stats`/`_apply_ratio_stats` replikası;
  winsor zaten train-fit'ti → A/B YOK, yalnız parity. None-drop × reindex çatışmasız (frozen params).
- **CountEncodeStep** (parametric/fitted_stat). `_count_encode_tt` replikası; değer→freq map, görülmemiş→0.
  **KANONİK unseen kaydı** `{known, strategy, monitor, known_limitation}` — TÜM encoder'lar aynı şekil
  (Katman 2 tek yerden okur). `known_limitation`: unseen→OOD-0 Faz 0 sentinel'i işaretli (düzeltilmez,
  aday A/B'li ayrı iş). A/B YOK (sadık kopya).
- **Parity — STANDART: fixture + gerçek-kolon İKİSİ birden** (bu adımdan itibaren her adımda).
  `tests/test_layer1_impute_parity.py`: **10/10 yeşil.** Çok-adımlı zincir parity (impute→ratio→count)
  = orkestratörün habercisi. Gerçek-kolon: credit `loan_grade` (nadir G/F kategorileri fixture'da yok).

### 2.7 GİDİŞAT KARARLARI (onaylı — kayda geçti)
- **Katman 1 terminus:** İkiz-implementasyon borcu (yeni Step + eski `_tt` yan yana) ~15 adımda büyür.
  Terminus: **tüm adımlar + orkestratör bitince, eski `_tt`'ler ya yeni Step'e delege edilir ya emekliye
  ayrılır → tek implementasyon, parity yapısal garanti.** Şimdi çevrilmiyor (validation önce), ama borç
  SINIRLI, sürüklenme yok. Her Step docstring'inde "eski yol referans; değişirse replika+parity birlikte
  güncellenmeli" eşiği var → terminus'ta risk kalkar.
- **Orkestratör invariant'ı (saklı, orada çözülecek):** manifest `order` ELLE atanmayacak — fiili
  fit-döngüsü sırasından TÜREYECEK; koşullu adımlar o koşuda fiilen çalışanları kaydedecek. **Manifest
  sadakati orkestratörün asıl test edilecek invariant'ı** (şimdi isimlendirildi).
- **A/B ilkesi netleşti:** yalnız SIZINTI-DÜZELTEN adımlar (impute #4, zscore #5) A/B taşır;
  sadık-yeniden-yazım adımları (ratio_finalize, count_encode, ...) yalnız parity.
- **Frontend:** Katman 2 sonunda, backend sağlamlaşınca (provenance, tahmini↔gerçekleşen, deterministik
  durum rozeti, önem-ağırlıklı drift). Şimdi backend.

### 2.8 Encoder üçlüsü — YAZILDI (commit sende, 4 parça)
`decide_encode_route` + `OneHotStep` + `TargetEncodeStep`. Eski `_encode_categorical_tt` (tek fonksiyonda
kardinaliteyle dallanma) → üç concern'e bölündü. **26/26 parity yeşil.**

- **Parça 1 — `protocol.py`: `fit_transform` sözleşmesi + `DefaultFitTransform` mixin + E invariant'ı.**
  Keşif E: sklearn `TargetEncoder.fit_transform` ≠ `transform` (cross-fitting; sklearn dokümanı birebir
  doğruladı). Yakalanmasaydı target parity sessizce YANLIŞ şeyi kanıtlardı. Mixin = fit+transform (diğer TÜM
  adımlar; Impute/Ratio/Count retrofit edildi). TargetEncode mixin'i KULLANMAZ, override eder.
  **INVARIANT (imza + fail-closed):** `transform` imzasında y YOK → serving hedef geçemez → cross-fitting
  ULAŞILAMAZ; `fit_transform` y zorunlu, None→RuntimeError → serving (hedefsiz) çağıramaz. Böylece
  serving-parity (transform==transform) ve train-parity (fit_transform==fit_transform) İKİ AYRI iddia.
- **Parça 2 — `steps.py`: `decide_encode_route` (TransformStep DEĞİL) + `OneHotStep` (frozen_choice).**
  Route: ≤10→onehot, >10→target; **nunique YALNIZ fit-anı**, sonuç frozen_choice, encoder `config["columns"]`'una
  enjekte (RatioFinalize deseniyle aynı). INVARIANT A testli: inference'ta kardinalite şişse bile karar donmuş,
  yeniden ölçülmez. OneHot: get_dummies(drop_first) + dummy-vocab dondur + `transform`'da o vocab'a reindex
  (unseen→all_zero, ilk gerçek reindex adımı). Lokal reindex=dummy vocab; orkestratör final_features reindex=
  tüm-çerçeve güvence (iki katman).
- **Parça 3 — `steps.py`: `TargetEncodeStep` (fitted_stat, İLK PICKLE) + `check_sklearn_compat`.**
  Sürüm kapısı (C): `check_sklearn_compat` patch→warn / minor|major→abort (sklearn pickle-binary uyumu
  minor'da kırılır: public-API-stable != pickle-stable). `enforce_sklearn_compat` FAIL-CLOSED: abort→
  `SklearnVersionMismatch` firlat, model HİÇ skorlamaz; `transform` baştan çağırır (savunma-derinliği).
  Abort mesajı actionable ("sklearn X ile eğitildi, runtime Y; yeniden eğit ya da runtime eşle"). İKİ FARKLI
  başarısızlık modu: sklearn runtime hatası→graceful drop (eski 1219-1222 parity); sürüm→kasıtlı hard-abort.
- **Parça 4 — parity testleri (16 yeni, 26 toplam).** A invariant (şişmiş-kard→hâlâ onehot), D (onehot
  all_zero≡referans, count OOD'dan ayrı known_limitation), E (transform y-YOK imzası, fit_transform y-zorunlu
  fail-closed, cross-fit≠düz), C (verdict tablosu + fail-closed abort + serving version-mismatch→hiç skorlamaz),
  target serving+train İKİ parity, unseen→prior, 4-adım zincir (impute→ratio→count→onehot).
  Gerçek-kolon: onehot=credit loan_grade/intent/home_ownership; target=superstore Sub-Category (YARI-GERÇEK,
  y sentetik → "iki impl. aynı" kanıtlar, "gerçek target'ta doğru" DEĞİL; yorumda yazılı).
- **B netleşti:** churn+credit'te >10 kardinalite YOK → target path yapısal INERT (#4 gibi). Gerçek yüksek-kard
  yalnız superstore/customers'ta. Zincir artık: impute→ratio_finalize→count_encode→[route]→onehot→target.

### 2.9 Scaler ailesi — YAZILDI (commit sende, 4 parça)
`_scale_features_tt` → `pickle_io` + `InfImputeStep` + `decide_scaler_route` + `RobustScaleStep` +
`StandardScaleStep`. **36/36 parity yeşil** (26→36, +10).

- **Parça 1 — `pickle_io.py` (YENİ) + TargetEncode PickleRef refactor.** `PickleRef`(estimator+sklearn_version)
  + `make_pickle_ref`/`load_pickle_ref` (tek sürüm-kapısı, fail-closed). `check_sklearn_compat`/
  `enforce_sklearn_compat`/`SklearnVersionMismatch` steps.py'den BURAYA taşındı → 3 pickle tüketici (Target,
  Robust, Standard) tek yerden. **Serialize task#6 çapası:** PickleRef sonra JSON-metadata + .pkl blob'a
  serialize olacak → serialize "PickleRef yaz/oku"a iner. TargetEncode refactor DAVRANIŞ değiştirmedi
  (`test_target_refactor_behavior_unchanged`: transform çıktısı == gate'siz doğrudan estimator.transform).
- **Parça 2 — `InfImputeStep` (fitted_stat, inline).** inf→train-median→fillna(0.0). **DEĞER zaten train-fit
  (≠#4, A/B YOK).** TEK fark: inf_cols SEÇİM peek'i (eski: train OR test) YAPISAL kapatıldı (fit yalnız X_train).
  Kanıt DETERMİNİSTİK (`test_inf_peek_closed_deterministic`: küme farkı, ROC değil). Postcondition: çıkışta
  inf KALMAZ — öğrenilmemiş prod-inf → 0.0 (`test_inf_inference_stays_finite`), nan'a dokunulmaz.
- **Parça 3 — `decide_scaler_route` + `RobustScaleStep`(PICKLE#2) + `StandardScaleStep`(PICKLE#3).** route:
  outlier_cols→robust, diğerleri→standard; **encode_route ile AYNI frozen_choice şekli** (executor-anahtarlı
  disjoint listeler + audit; `test_routes_share_frozen_choice_shape`) → orkestratör iki route'u tek desenle
  ele alır. INVARIANT (`test_scaler_route_invariant_inflated_outlier`): inference'ta outlier profili değişse
  bile scaler seçimi donmuş. Robust/Standard ortak `_ScalerStepBase`, ama AYRI Step (encoder-split precedent'i).
- **Parça 4 — parity (10 yeni).** scaler zincir (inf→route→robust→standard) == `_scale_features_tt` fixture+
  gerçek-kolon; pickle_io passthrough+gate; mixin doğrulaması (`test_scaler_mixin_no_crossfit`: üçü fit_transform
  ==fit+transform, yalnız TargetEncode farklı → granülarite kanıtı); paylaşılan pickle-kapısı fail-closed.
- **outlier_cols kaynağı:** `meta.get_outlier_cols()` → `has_outliers` full-df profilinden (üst-katman §2.3
  selection-peek; bu adımda düzeltilmez, route KARARI donunca deployment-doğru). Zincir: ...→onehot→target→
  **inf_impute→robust→standard**. Cross-family tam zincir = ORKESTRATÖR'ün işi (elle-zincir borcu, §2.7/memory).

### 2.10 Orkestratör iskelesi — YAZILDI (commit sende, 4 parça)
`orchestrator.py` (YENİ) + `RatioConstructStep` portu. Mevcut adımları gerçek `build_features` sırasıyla
replay eder; elle-zincir borcunu 8 adımda kapatır. **37/37 yeşil** (41 → 4 elle-zincir silindi → 37).

- **Parça 1 — `orchestrator.py`.** `PIPELINE` (tek doğruluk kaynağı, `StepSpec`/`RouteSpec`) + `fit_pipeline`
  (adımları sırayla fit_transform, manifest `order`'ı **fiili döngü indeksinden** türetir — elle atama YOK) +
  `transform_pipeline` (manifest'i replay, **YALNIZ `step.transform`** → E pipeline-düzeyi invariant; route/skip
  atlanır, sonda `final_features` reindex) + `STEP_REGISTRY`.
- **Parça 2 — `RatioConstructStep` (rowwise).** `_build_ratio_features`:1934-1936 row-wise çekirdeği. Geçici
  `_build_ratio_stand_in` (test iskelesi) buraya FORMELLEŞTİ → stand-in emekli (kalıcı ikiz olmadı). KİLİT:
  `(num/den).replace([inf,-inf], nan)` — inf→NaN (Faz 0 sentinel kökü). Çift-seçimi AYRI (config'ten gelir);
  bu adım yalnız bölme. rowwise → fit None, config manifest'te taşınır (serving'de geri yüklenir).
- **Parça 3 — testler (5 yeni).** `test_ratio_construct_inf_nan_parity` (eski `_build_ratio_features`'la, 20-satır
  fixture, payda-sıfır satır NaN); `test_orchestrator_multistep_parity` (fit==eski-oracle-dizisi + serving==
  training, target yok); `test_orchestrator_manifest_order_is_enumerate`; `test_legacy_ref_source_drift`
  (KOŞULLU-ATLAMA netliğiyle); `test_orchestrator_transform_pipeline_never_fit_transforms` (monkeypatch — E kanıtı).
- **Parça 4 — SİLME (orkestratör parity yeşil OLDUKTAN sonra).** 4 elle-zincir: `test_chain_impute_then_ratio_finalize`,
  `test_chain_impute_ratio_count`, `test_chain_impute_ratio_count_onehot`, `test_scaler_chain_mod_a_equivalence` +
  `_build_ratio_stand_in` + `_scaler_fixture`. Çok-adım parity artık TEK yol: orkestratör (bakım ikiye katlanmaz).
  KALDI: tüm tek-adım `*_mod_a`, `*_real_column_*` (scaler real-column dahil), invariant'lar.
- **build_features gerçek sıra haritası:** `_clean_data`(2133)→`_build_ratio_features`(2289)→SPLIT→
  `_finalize_ratio_features_tt`(2398)→`_count_encode_tt`(2427)→`_groupby_deviation_tt`(2428)→
  `_encode_categorical_tt`(2431)→`_scale_features_tt`(2432)→`_select_features_tt`(2433,YOK)→univariate(2440)→leakage(2443).

### 2.11 GroupbyDeviation — YAZILDI (commit sende, 3 parça) — 37→40 test
`GroupbyDeviationStep` (`steps.py`) `_groupby_deviation_tt`(feature_builder.py:1444-1544) portu. PIPELINE'da
`CountEncode` ile `encode_route` arasına (kaynak 2427→2428→2431 birebir). `_GROUPBY_*` sabitleri (`MAX_CARD=50`,
`MAX_NUM/CAT_COLS=3`, `MIN_GROUP_SIZE=30`, `MAX_FEATURES=30`) `feature_builder`'dan kopya (self-contained).

- **Parça 1 — `GroupbyDeviationStep` (parametric / fitted_stat).** Her seçili (cat,num) çifti → `dev_/zdev_/grpstd_`.
  **İKİ-boyutlu donma:** (1) *seçim* — cat (nunique↑, cap 3) × num (`-norm_var`, cap 3) + güvenilir-grup kapısı
  (≥30 satır, cat'te <2 güvenilir grup → atla) + `MAX_FEATURES` cap — YALNIZ fit-anı, `records` listesine GÖMÜLÜ
  (kayıt=seçim kanıtı, ayrı frozen_choice YOK); (2) *istatistik* — `grp_mean`/`grp_std`(ddof=0)/`global_*` train'den.
  `grp_mean.where(reliable, global)` unguvenilir grubu fit'te global'e sabitler → `transform`'da `map().fillna(global)`
  hem unguvenilir hem unseen'i tek yolda global'e taşır. **Unseen → GLOBAL** (kanonik `strategy="global"`, count'un
  `map_to_zero` OOD-sentinel'inden AYRI; dağılım-altı 0 değil nötr global istatistik).
- **Parça 2 — PIPELINE'a düşürme + STEP_REGISTRY.** `StepSpec(GroupbyDeviationStep, "_groupby_deviation_tt",
  config_from=exclude←encode_exclude)`. Encode ÖNCESİ (ham cat = grup anahtarı). `legacy_ref` drift testi otomatik
  doğrular (2428 pozisyonu). 8-satır orkestratör fixture'ında NO-OP (grup<30) → mevcut parity kırılmadı.
- **Parça 3 — testler (3 yeni).** `test_groupby_deviation_fixture_parity` (65-satır: A×30+B×30 güvenilir + C×5
  unguvenilir + test'te unseen Z; dev/zdev/grpstd train+test tarafı eski `_groupby_deviation_tt` ile birebir +
  unseen→global assert'i); `test_groupby_deviation_real_column_parity_credit` (gerçek-kolon); `test_groupby_deviation_in_orchestrator`
  (reliable-grup fixture fit_pipeline→manifest'te groupby + serving==training). **A/B YOK** (sadık port, sızıntı düzeltmez).

### 2.12 SelectDrop — YAZILDI (commit sende) — TERMINUS #3 — 40→44 test
`SelectDropStep` (`steps.py`) — 4 selection alt-op'unu (low-var/high-corr + supervised univariate + leakage tespiti +
domain-leakage) TEK donmuş drop-listeye indirger. `parametric`/`frozen_choice`. PIPELINE'da **EN SON** (scaler'dan sonra).

- **DELEGE çatalı (kullanıcıya soruldu → onaylandı, `AskUserQuestion`):** iki seçenek — (a) delege: fit mevcut
  `_select_features_tt`/`_select_univariate_tt`/`_detect_leakage_tt`/`_apply_domain_leakage_drops_tt`'yi çağırır;
  (b) reimplement: ~270 satır (122'si `_detect_leakage`) self-contained yeniden-yaz. **(a) seçildi.** Gerekçe: serving'de
  selection = orkestratörün terminal `reindex(final_features)`'ı → transform pür-drop, learned numeric state YOK, kolon
  listesi yeter. "Yalın inference runtime" hedefi transform yolu için; fit eğitim-anı (feature_builder mevcut). Reimplement
  = serving'de hiç çalışmayan 270 satır için salt drift riski. Terminus zaten "eski `_tt` → delege → tek impl." diyordu.
- **`fit` (eğitim-anı):** LAZY+LOKAL `feature_builder` import (modül-yükleme çekmez → inference import yalın). Diziyi
  build_features:2433-2447 SIRASIYLA çalıştırır: low-var/high-corr → `to_numeric.fillna(0)` (ANOVA NaN kabul etmez) →
  univariate (yalnız cls/reg + y) → **leakage (`y is not None` guard** — build_features'ta tail supervised-dal-only) →
  domain. Hayatta-kalan set çıkar → `{drop_cols(orijinal sıra), provenance{kolon:detektör}, surviving}` DONAR.
- **`transform` (eğitim+serving AYNI, self-contained):** donmuş `drop_cols`'u düşür + terminal `to_numeric.fillna(0)`
  (build_features:2435; idempotent → serving==training). feature_builder import YOK.
- **Orkestratör:** `StepSpec(SelectDropStep, "_select_features_tt", config←{intent,meta,domain_leakage_report,
  protected_high_card})` PIPELINE EN SON. `_old_oracle_sequence`'a selection kuyruğu + groupby (no-op) eklendi →
  `test_orchestrator_multistep_parity` yeniden yeşil (aksi halde fit selection yapar, oracle yapmaz → kolon-seti farkı).
- **Testler (4 yeni):** `test_select_drop_frozen_drop_parity` (delege-wiring == build_features dizisi; const→low-var,
  corr çifti→high-corr düşer); `test_select_drop_transform_is_frozen` (**re_derive:false** — train'de düşen serving'de
  yüksek-varyanslı olsa da düşer, kalan sabit olsa da kalır); `test_select_drop_is_terminus` (PIPELINE[-1]==SelectDrop);
  `test_select_drop_in_orchestrator` (manifest EN YÜKSEK order + low-var final_features'ta yok + serving==training).

### 2.13 ZScore/Normalize (#5 fix) — YAZILDI (commit sende) — 44→48 test — **KATMAN 1 ADIMLARI TAMAM**
`ZScoreStep`+`NormalizeStep` (`steps.py`) — plan-güdümlü `zscore`/`normalize` derived-feature'ları. `parametric`/
`fitted_stat`. PIPELINE'da **Impute↔RatioConstruct arası** (legacy_ref `_apply_plan_derived_features`@2260, kaynak
sırası `_clean_data`@2133 < 2260 < `_build_ratio_features`@2289 → drift testi sadık).

- **#5 sızıntı:** eski `_FEATURE_OPS["zscore"]`(1577)=`(x−x.mean())/x.std()` ve `normalize`(1581)=min-max, `cols[0]`
  TÜM df'ten (split öncesi, satır 2260) → test dağılımı eğitime sızar. **Fiks:** μ/σ (min/max) YALNIZ `fit`(X_train)'den;
  `transform` donmuş istatistiği uygular (#4 disiplini, imza-düzeyi skew-imkânsız). Çıktı `replace([inf,-inf],0).fillna(0)` (op:1631).
- **Eski ile parity YOK (bilerek sapar — eski leaky, yeni doğru).** Doğrulama iki ayrı iddia: (a) **FORMÜL-parity** —
  transform == μ/σ elle fit-çerçevesinden (op formülü sadık); (b) **SIZINTI-delta** — full-df μ/σ ≠ train-only (train-dist
  ≠ full-dist'te) → test satırlarında çıktı farklı = düzeltmenin kaldırdığı sızıntının **deterministik** kanıtı
  (ROC A/B DEĞİL — CLAUDE.md "leakage deterministik çözülürse ROC'a tercih et"; kullanıcı `AskUserQuestion` onayı).
- **Additive (kullanıcı onayı):** `feature_builder` DOKUNULMADI (eski leaky yol terminus'a kadar kalır); gerçek fiks
  (pre-split skip + post-split uygula) terminus/entegrasyona ertelendi. Diğer çatal (reimplement) reddedildi.
- **DİNAMİK (kullanıcı özel istedi):** kolon adı GÖMÜLÜ DEĞİL. `config["specs"]`=plandan `[(çıktı_adı, girdi), ...]`.
  `test_zscore_normalize_real_column_dynamic` glob CSV + `select_dtypes("number").columns[0]` programatik seçimle kanıtlar.
- **Testler (4 yeni):** `test_zscore_normalize_formula_parity` (fixture, iki op); `test_zscore_normalize_leakage_delta`
  (train-dist≠full-dist → test satırında farklı); `test_zscore_normalize_real_column_dynamic` (gerçek-kolon, hardcode yok);
  `test_zscore_in_orchestrator` (specs→param_store'da train-fit μ; serving==training). **NOT:** z afin(input) → corr=1 →
  select_drop high-corr onu düşürür (DOĞRU, ham ile gereksiz); üretim kanıtı `param_store`'dan, final_features üyeliği değil.

### 2.14 UÇTAN-UCA gerçek-veri parity harness — YAZILDI (commit sende) — 48→50 — TERMINUS ÖN-KOŞULU
Terminus (üretim rewire) körlemesine olmasın diye: `fit_pipeline`'ın `build_features`'ın GERÇEK post-split zinciriyle
gerçek datasette birebir olduğunu kanıtlar. **Kullanıcı `AskUserQuestion` ile "parity harness ÖNCE" seçti.** Additive.

- **`_old_oracle_sequence` config-tam yapıldı:** zscore/normalize (train-fit: frame=X_train) + ratio + count/groupby
  exclude←ctx + univariate(intent) + leakage(meta/protected←ctx) + domain. Hem 8-satır testini hem gerçek-veri harness'ını
  besler (8-satırda ekstra config yok → geriye-uyumlu). Kullanılan `_finalize_ratio_features(X_train)` = RatioFinalizeStep'in
  X_train tarafıyla birebir (ikisi de train'e fit+apply).
- **`_real_dataset_parity_harness(csv, intent)`:** (a) `_clean_data`'yı BİR KEZ upfront çalıştır (dedup+currency+impute —
  fit_pipeline'ın port ETMEDİĞİ ön-temizlik) → ImputeStep/_clean_data ikisi de no-op → #4 nötr; (b) split; (c) ctx DİNAMİK
  (numeric kolonlar programatik, hardcode YOK); (d) `fit_pipeline(X_train) == _old_oracle_sequence(X_train)` birebir (set
  + allclose); (e) `transform_pipeline(X_test)` serving yolu (final_features + satır sayısı korunur). y sentetik-deterministik.
- **2 test:** `test_orchestrator_real_dataset_parity_classification` (credit), `..._regression` (housing). İkisi de GEÇER (skip değil).
- **HARNESS'IN KAPSAMADIĞI (terminus'ta ele alınmalı):** (1) `_clean_data` dedup+currency PORT-DIŞI (upfront nötrlendi);
  (2) impute/zscore pre-split→post-split RELOKASYONU = #4/#5'in fiili deploy'u (ML sonucu değişir → offline_report regresyonu).

### 2.15 ÜRETİM-KANITI GUARD'I + KAPSAM ENVANTERİ — YAZILDI (commit sende) — 50→57 test
**Kullanıcı sırayı bilinçli tersine çevirdi: guard (8.2) ÖNCE, envanter (8.1) sonra.** Gerekçe: *"guard envanterin
kendisi"* — elle çıkarılmış kalem listesi eksik olabilir, guard olamaz. **Kod söylesin, biz değil.**

- **Guard (`orchestrator.py`):** `transform_pipeline` sonunda, reindex'ten ÖNCE, her `final_features` kolonu için
  onu fiilen üreten bir adım çalıştı mı diye bakılır. Eksikse `MissingFeatureProduction` → **skorlama HİÇ yapılmaz**
  (fail-closed; yanlış skor > sessiz bozulma). Terminal `reindex(fill_value=0)` artık yalnız **politikası tanımlı**
  boşlukları doldurur: adımlar opt-in `zero_fill_policy(params)` beyan eder (bugün tek beyan: `OneHotStep` → donmuş
  `dummy_vocab`, çünkü all_zero ≡ referans kategori, §2.8 D). Tanımsız eksik = HATA.
- **İKİ BAŞARISIZLIK MODU (ölçüm confound'u yakalandı):** ilk koşuda liste şişti — kendi `select_drop`'umuz yabancı
  bir `final_features` listesini tanımayıp kolonları düşürüyordu. Guard ikiye ayrıldı: **`never_produced`** (kolon
  replay boyunca HİÇ var olmadı → *port borcu*) vs **`produced_then_dropped`** (üretildi, sonraki adım düşürdü →
  *artifact bütünlüğü hatası*; port etmek ÇÖZMEZ). Üretimde de ayrı teşhisler.
- **Envanter aracı (`scripts/layer1_coverage_inventory.py`, YENİ, kalıcı):** fit tarafı **gerçek `build_features`**
  (offline_report `build_state` ile, LLM'siz), transform tarafı orkestratör → fark = portlanmamış kalemler.
  Sınıflandırma da bize ait DEĞİL: her eksik kolon `feature_report`'un KENDİ attribution alanlarıyla etiketlenir
  (`ratio_features`/`derived_flags`/`auto_features`/`derived_features`/`domain_features`).
- **SONUÇ — 6 dataset, 3 KÖK boşluğa yakınsadı** (credit %39 · housing %46 · churn %57 · superstore %32 ·
  transactions %41 · student %67 üretilemiyor):
  1. **Ratio ÇİFT SEÇİMİ** (66 kolon, baskın) — `RatioConstructStep` yalnız row-wise BÖLMEyi portladı; *hangi
     çiftler* kararı (`_build_ratio_features` seçim yarısı) portlanmadı.
  2. **`_build_zero_missing_flags`** (`is_zero_*`) — hiç portlanmadı.
  3. **`_add_missing_indicators`** (`*_isna`) — hiç portlanmadı.
  4. *(kaskad, bağımsız değil)* `grpstd_*`/`zdev_*`/`count_*` sapmaları: GroupbyDeviation/CountEncode PORTLU ama
     seçimleri girdi-bağımlı → yukarıdaki 3 boşluk yüzünden farklı çerçeve görüp farklı seçiyorlar. 1-3 kapanınca düşer.
- **ENVANTERİN SINIRI (dürüstlük notu — atlanmamalı):** guard **gördüğünü kesin** söyler, **görmediği hakkında
  hiçbir şey** söylemez. Bu 6 dataset'in tetiklemediği kalemler listede YOK ama kapsandıkları KANITLANMADI:
  `_normalize_column_names` (çarpışma yok), date türevleri (`_infer_semantic_type` bu CSV'lerde timestamp
  etiketlemedi), `_FEATURE_OPS`'un kalan 6 op'u (offline plan `derived_features` üretmiyor), domain recipe'leri
  (pack 'none'), `_aggregate_if_needed`, ts-lag, critic `drop_features`/`forced_leakage_drops`, id/ts/target drop'ları.
  **"Listede yok" ≠ "kapsanıyor"** — bunlar *ölçülmemiş bilinmeyen*. Yeni modalite geldikçe script yeniden koşulmalı.

### 2.16 Gözden geçirme düzeltmeleri (tasarım-Claude'un incelemesi) — YAZILDI (commit sende)
- **(a) SELF-CONTAINED invariant'ının KAPSAMI koda yazıldı.** Kural artık "her Step self-contained" değil,
  **"her `transform` self-contained"**: `transform` (serving+eğitim) `feature_builder` import ETMEZ; `fit`
  (yalnız eğitim-anı) lazy delege edebilir. `protocol.py`'de gerekçeli bölüm + `SelectDropStep` docstring'inde
  geri-referans. **Ölçüt yazıldı:** yeni bir adım delege isterse → *"transform'u delegesiz yazabiliyor musun?"*
  Yazamıyorsa adım yanlış bölünmüştür. (Amaç: ileride biri ya panik yapsın ya bahane üretsin diye değil.)
- **(d) `frozen_choice` ŞEKİL TUTARSIZLIĞI DÜZELTİLDİ.** Önce: route'larda seçim manifest kaleminde, groupby'da
  `params["records"]`'a gömülü → "seçim nerede" sorusunun cevabı ADIMA GÖRE değişiyordu (Katman 2 her adıma ayrı
  el sokacaktı; scaler turunda özellikle kurulan route-shape tutarlılığı bozuluyordu). Şimdi: **tek manifest alanı
  `frozen_choice = {kind, selection, ...}`** — route (`kind="route"`) ve adım-içi seçim (`kind="pairs"`) AYNI
  şekilde. Adımlar opt-in `frozen_selection(params)` beyan eder. `param_store` hâlâ KAYNAK; bu görünürlük
  katmanı (kopya değil türev) — sapma testle yakalanıyor.
- **(e) `_old_oracle_sequence` İDDİA ENFLASYONU düzeltildi.** Test dosyasının harness başlığına
  KANITLANAN/KANITLANMAYAN/BORÇ/KAPSAM olarak açıkça yazıldı: kanıtlanan *"fit_pipeline == elle rekonstrüksiyon"*,
  **`== build_features`'ın kendisi DEĞİL**. Oracle = silinen 4 elle-zincirin yerine geçen TEK merkezî elle-zincir
  (daha iyi, ama hâlâ borç); sadakati bugün yalnız `test_legacy_ref_source_drift`'in **göreli sıra** kontrolüyle
  korunuyor (değer/config sadakati DEĞİL). Terminus'ta build_features orkestratörü fiilen çağırınca borç
  KENDİLİĞİNDEN düşer.
- **(b)+(c) kabul edildi, ilke metni güncellendi** → §4 madde 8 (aşağıda).
- **Yeni testler (7):** guard'ın iki modu ayırması · sessiz-0 regresyon kilidi · sağlıklı artifact'te yanlış-pozitif
  yok · `zero_fill_policy` beyanı · `frozen_choice` tek-şekil (route ≡ pairs) · frozen_choice ↔ param_store tutarlılığı.

### 2.17 KAPSAM KAPISI + ratio çift-seçimi frozen — YAZILDI (commit sende) — 57→64 test
**Kullanıcı kararı: spekülatif port YERİNE fail-closed reddetme.** *"Ölçemediğimiz yolu desteklediğimizi iddia
etmektense desteklemediğimizi söylemek dürüst ve çok daha ucuz."*

- **Kapsam kapısı (`check_pipeline_scope` + `UnsupportedPipelineScope`).** `fit_pipeline` **ilk adımdan ÖNCE**
  ctx'te beyan edilen kapsam olgularını denetler; desteklenmeyen yol tetiklendiyse **artifact KURULMAZ**
  ("bu koşu deploy edilemez"). Konum bilinçli: serving'de değil **artifact-build anında** — serving'de patlamak
  "model var ama skorlanamıyor" demek, hatayı en geç yerde bulmak olurdu. `_SCOPE_RULES` tek doğruluk kaynağı:
  `modality`(tabular) · `aggregated` · `timeseries_lag` · `domain_features` · `date_features` · `column_renames` ·
  `unported_ops` (portlu: `PORTED_FEATURE_OPS = {zscore, normalize}`).
  **Fail-open YOK:** beyan EDİLMEMİŞ olgu ihlal sayılmaz (test/deney artifact'i kurulabilsin) ama artifact
  `deployable=False` damgalanır → sessiz "destekleniyor" varsayımı imkânsız. `Artifact.supported_scope` alanı yeni.
  **Kapsam genişletme sırası (kural):** port et → guard'ı koş (eksik-üretim 0) → `_SCOPE_RULES`'tan kaldır. Üçü
  birden yapılmadan kapsam genişlemez.
- **Ratio çift seçimi — "port et" DEĞİL, "pre-split sabit → artifact'e frozen yaz" (8.1 kategorisi #2).**
  Kullanıcının işaret ettiği kısayol DOĞRULANDI: `feature_report["ratio_feature_meta"]` (feature_builder.py:2491)
  `{ratio_kolonu: (pay, payda)}` tutuyor ve **X_train'e göre FİLTRELENMİYOR** → selection'da sonradan düşen oranlar
  da içinde, yani KURULAN tüm çiftlerin tam kaydı. (`ratio_features` anahtarı hayatta-kalan listesi → frozen kaynağı
  olarak YANLIŞ olurdu.) Seçim mantığı yeniden yazılmadı; karar OKUNUYOR. `RatioConstructStep.frozen_selection`
  manifest'e kanonik `{kind:"pairs", source:"feature_report.ratio_feature_meta", selection:[...]}` yazıyor.
  **Dayanak testle çitlendi** (`test_ratio_meta_is_unfiltered_superset_of_survivors`): biri ileride meta'ya
  `X_train.columns` filtresi koyarsa test kırılır — kısayolun dayanağı sessizce çürümez.
  **KAYITLI, KAPANMADI:** seçim hâlâ split ÖNCESİ tüm df'ten (§2.3 selection-peek sınıfı). Bu adım kararı sadakatle
  TAŞIR, peek'i kapatmaz. Kapatmak ayrı iş (seçimi fit(X_train)'e almak, InfImpute gibi deterministik kanıtla).

**GUARD TABLOSU — ratio kablolaması sonrası (6 dataset, `HİÇ ÜRETİLMEDİ`):**

| Dataset | Önce | Sonra |
|---|---|---|
| credit | 18/46 (%39) | **4/46 (%9)** |
| housing | 12/26 (%46) | **1/26 (%4)** |
| churn | 12/21 (%57) | **2/21 (%10)** |
| superstore | 7/22 (%32) | **3/22 (%14)** |
| transactions | 13/32 (%41) | **0/32 (%0)** ✅ |
| student | 16/24 (%67) | **1/24 (%4)** |

**KASKAD İDDİASI — DOĞRULANDI ama KISMEN (varsayılmadı, ölçüldü).** "grpstd/zdev/count sapmaları 1-3 kapanınca
düşer" iddiası: ratio kapanınca transactions TAMAMEN temizlendi ve superstore'da `grpstd_Quantity_by_Category`
düştü → kaskad GERÇEK. Ama çoğu grpstd/zdev DURUYOR → ratio tek kaynak değildi. Kalan sapmaların
`is_zero_*`/`*_isna` boşluğundan geldiği HİPOTEZ (groupby seçimi çerçeve-bağımlı: cat×num seçimi frame'de hangi
kolonlar varsa ona göre), **henüz kanıtlanmadı** — flag'ler portlanınca guard yeniden koşulup doğrulanacak.
Çözülmezse ayrı bir kök var demektir.

**KALAN 3 KÖK (11 kolon, 6 dataset toplamı):** `_build_zero_missing_flags` (`is_zero_*`, 4 dataset) ·
`_add_missing_indicators` (`*_isna`, 1) · groupby seçim sapması (`grpstd_*`/`zdev_*`, 5 — muhtemelen kaskad).

### 2.18 Kalan kök boşluklar + 4. KÖK keşfi — YAZILDI (commit sende) — 64→70 test — **KAPSAM BORCU KAPANDI**

**Karar (a) SADAKAT-ÖNCE — açıkça işaretlendi.** is_zero/`_isna` seçimleri `fit(X_train)`'e ALINMADI; eski yolda
nerede yapılıyorsa orada bırakılıp KARAR taşındı (ratio ile aynı kategori). **Peek borcu AÇIK ve kayıtlı.**
Gerekçe kullanıcının önerisiydi (amaç kapsam, sızıntı düzeltme değil; peek'ler aynı sınıf → terminus sonrası TEK
turda toplu) + **kod okunurken bulunan EK gerekçe:** `_build_zero_missing_flags` seçimi saf prevalence kapısı
DEĞİL — `force_zero_cols` üzerinden **oran motoruna bağlı bir invariant** taşıyor (sıfır-payda kolonları
prevalence'tan VE cap'ten muaf, garanti flag alır). Seçimi yeniden türetmek bu motorlar-arası kuplajı inference
paketinin içinde yeniden kurmak olurdu → tam da kaçındığımız drift riski. Donmuş listeyi taşımak invariant'ı
bedavaya korur. *(b) seçilseydi bu kuplaj sessizce kopardı.*

- **`MissingIndicatorStep`** (rowwise/frozen). `<col>_isna`. **SIRA KİLİDİ: impute'tan ÖNCE** (kaynak
  `_add_missing_indicators`@2130 < `_clean_data`@2133) — sonra çalışsaydı null kalmaz, bayrak hep 0 olurdu
  (sinyal sessizce silinir). Hem PIPELINE sırası hem DAVRANIŞ testli.
- **`ZeroFlagStep`** (rowwise/frozen). `is_zero_<col>`. #10 binary-guard seçimde zaten uygulanmış, tekrarlanmaz.
- **`DropColumnsStep`** (rowwise/frozen) — **4. KÖK, ölçülerek bulundu.**

**KASKAD HİPOTEZİ ÇÜRÜDÜ (varsayılmadı, ölçüldü).** "grpstd/zdev sapmaları flag'ler portlanınca düşer" iddiası
TUTMADI: flag'ler portlandı, sapma DURDU. Groupby'ın kendi seçim mantığına bakıldı ve gerçek kök bulundu:

> **Guard'ın KÖR NOKTASI — eksik KALDIRMA.** Guard yalnız *eksik üretim* görür, *fazladan mevcudiyet* görmez.
> Eksik bir kaldırma "kolon üretilmedi" olarak GÖRÜNMEZ; bunun yerine **downstream SEÇİM SAPMASI** üretir.
> Ölçülen vaka (credit): hedef `loan_status` çerçevede kalınca `GroupbyDeviation`'ın `_norm_var` sıralamasında
> **BİRİNCİ** sıraya oturdu (binary p≈0.22 → normalize varyans yüksek) ve `cb_person_cred_hist_length`'i
> top-3'ten attı → `grpstd_cb_person_cred_hist_length_by_*` üretilemedi.
> Doğrudan ölçüm: `HEDEF VAR → ['loan_status','loan_amnt','loan_int_rate']` · `HEDEF YOK → ['loan_amnt',
> 'loan_int_rate','cb_person_cred_hist_length']`.

- **AYRICA SIZINTI YÜZEYİ (yeni, kayda geçti):** hedef çerçevede kalırsa groupby/count **HEDEFTEN** feature
  türetir (`dev_<target>_by_X` = hedefin grup ortalamasından sapması) → doğrudan target leakage. `build_features`'ta
  imkânsız (hedef split öncesi düşer); **orkestratörde engelleyen hiçbir şey YOKTU.** `DropColumnsStep` kapatır,
  iki test kilitler (`test_target_in_frame_distorts_groupby_selection`, `..._creates_target_derived_feature`).
- `legacy_ref` = `_find_related_target_cols` (build_features:2251 — o blokta fiilen çağrılan fonksiyon, drift
  testi konumu doğrular). PIPELINE'da Impute ↔ ZScore arası.

**GUARD TABLOSU — NİHAİ (6 dataset, iki sepet):**

| Dataset | never_produced | produced_then_dropped | Açıklanamayan |
|---|---|---|---|
| credit | **0/46** | 30 (hepsi `select_drop (frozen)`) | **0** |
| housing | **0/26** | 17 (hepsi `select_drop (frozen)`) | **0** |
| churn | **0/21** | 15 (hepsi `select_drop (frozen)`) | **0** |
| superstore | **0/22** | 13 (hepsi `select_drop (frozen)`) | **0** |
| transactions | **0/32** | 22 (hepsi `select_drop (frozen)`) | **0** |
| student | **0/24** | 19 (hepsi `select_drop (frozen)`) | **0** |

`produced_then_dropped` sepeti artık **tek tek açıklanıyor** (`_explain_dropped`): her kolon ya artifact'in KENDİ
donmuş `drop_cols`'unda ya encoder tarafından tüketilmiş; **açıklanamayan = 0**. Terminus'ta `final_features`
artifact'in kendi çıktısı olacağı için bu sepet zaten boşalacak.

**TERMINUS'A HAZIRLIK ÖLÇÜTÜ — 4/4 SAĞLANDI:** ✅ never_produced=0 (6 dataset) · ✅ produced_then_dropped hepsi
açıklanmış ölçüm artefaktı · ✅ kapsam dışı kalemler `supported_scope`'ta beyan edilmiş · ✅ guard tablosu basıldı.

**⚠️ TERMINUS GÖREVİ (ratio dersinin tekrarı):** `build_features` bugün `derived_flags` ve `auto_features`'ı
X_train'e göre FİLTRELİ yayınlıyor (hayatta-kalan listesi) → frozen kaynağı olarak YETERSİZ. Envanter scripti bu
yüzden seçim fonksiyonlarını doğrudan çağırıyor. Terminus'ta bu ikisi için de `ratio_feature_meta` gibi
**filtrelenmemiş** bir kayıt yayınlanmalı, yoksa üretim kablolaması hayatta-kalanları alır ve groupby'ın seçim
yaptığı ara çerçeve eski yoldan sapar.

### 2.19 SİMETRİK GUARD + YAPISAL-GARANTİ ENVANTERİ — YAZILDI (commit sende) — 70→80 test

**Guard'ın asimetrisi kapatıldı.** Guard "üretilmedi mi" soruyor, "fazladan var mı" sormuyordu; bu asimetri 4. kökü
sakladı. Artık `fit_pipeline` her adım kaleminde eğitim kolon-setini kaydediyor (`columns_out`), `transform_pipeline`
her adım sınırında **iki yönde** karşılaştırıyor → `FrameShapeDivergence` (eksik VE fazla), sapma **adım düzeyinde
lokalize** ediliyor (sonda değil). Fazla kolon için §2.4'ün "tolere et" tasarımıyla uzlaşma: **tolerans mümkün ama
BEYAN edilmeli** (`Artifact.tolerated_extra_columns`); beyansız fazla kolon = hata.

> **KAPSAM DÜRÜSTLÜĞÜ (abartma yok):** bu kontrol **serving↔eğitim** sapmasını yakalar. 4. kökün KENDİSİ bir
> **fit-tarafı** sorunuydu (orkestratörün fit girdisi build_features'ınkinden farklıydı) ve bu kontrol onu
> YAKALAMAZDI — fit ve transform aynı fazla kolonu görüyordu. Fit tarafının karşılığı yapısal ön-koşullar +
> kapsam envanteri scripti. Docstring'e de böyle yazıldı.

**YAPISAL-GARANTİ ENVANTERİ — 4 aday tek tek ÖLÇÜLDÜ (tahmin yok):**

| # | Aday | Sonuç | Karşılık |
|---|---|---|---|
| 1 | Hedefin split öncesi düşürülmesi | ✅ aynı sınıf (§2.18) | `DropColumnsStep` + ön-koşul + 2 test |
| 2 | ID/timestamp düşürülmesi | ❌ **AYNI SINIF, DAHA AĞIR** | ön-koşul + regresyon kilidi |
| 3 | Split'in konumu | ⚠️ sapma YOK, ama sözleşme yazılmamıştı | not + `deduplicated` olgusu |
| 4 | `_clean_data` dedup'ı ("skip") | ❌ **MASUM DEĞİL** | `_SCOPE_RULES`'a olgu + test |

- **(2) ID — ölçülen sonuç beklenenden ağır.** Benzersiz **string ID** `encode_route` eşiğini (>10 kardinalite)
  aşıp **TargetEncoder**'a gidiyor; serving çıktısı `y`'nin **BİREBİR kendisi** oluyor
  (`allclose(encoded, y) == True`, 60 satır) → kolon target'ın kopyası. **Integer ID** ise groupby'ın `-norm_var`
  sıralamasına girip gerçek kolonu top-N'den atıyor (ölçüm: `['cust_id_int','olcum']` vs `['olcum']`).
  Eski yolda İMKÂNSIZ (ID'ler split öncesi düşer). Regresyon kilidi: `test_unique_id_target_encoded_reproduces_y_exactly`.
- **(4) dedup — "skip'tir, masumdur" YANLIŞ çıktı.** Satır üyeliğini değiştirdiği için ondan SONRAKİ **her**
  `fitted_stat` kayıyor: ölçüm — impute medyanı dedup'sız **3.0**, dedup'lu **2.0**. Ayrıca PIPELINE'da hiç `skip`
  tipi adım YOK, `STEP_REGISTRY`'de dedup YOK → tasarım niyeti vardı, implementasyon yoktu. Çözüm: `deduplicated`
  kapsam olgusu (beyansız → `deployable=False`; `False` → artifact kurulmaz).
- **(3) split — sapma bulunmadı ama sözleşme AÇIK DEĞİLDİ.** Orkestratör pre/post-split sınırını çöktürüyor: tek
  çerçeve alıyor. Row-wise adımlar etkilenmez; `Impute`/`ZScore` için pre→post geçişi **zaten #4/#5 fiksinin
  kendisi** (kasıtlı). Sözleşme artık yazılı: *`fit_pipeline`'ın girdisi X_train'dir ve dedup edilmiş olmalıdır.*

**YAPISAL GARANTİ PRENSİBİ (kayda geçti):** eski yolun bazı garantileri hiçbir fonksiyonda yazılı değildi —
pipeline'ın SIRASINDAN/YAPISINDAN geliyordu. Adımlara çevirirken o garantiler sessizce kaybolur ve **eski yolda
imkânsız olan sızıntı yüzeyleri açılır**. Bu yüzden yapısal garantiler artık AÇIK ÖN-KOŞUL: beyan edilir ve
fit-anında denetlenir (`check_structural_preconditions`, drop adımından sonra → `StructuralPreconditionViolation`,
artifact kurulmaz).

**KARAR KURALI (iki soru, sırayla):**
1. *"Bu adımın doğruluğu kendi kodundan mı, pipeline'daki KONUMUNDAN mı geliyor?"* → Konumundansa yapısal garantidir.
2. *"O konum orkestratörde YENİDEN ÜRETİLEBİLİYOR mu?"*
   - **Evet → ADIM olarak portla.** Örnek: hedef/id drop'u — konumu (encode ve groupby'dan önce) orkestratörde
     birebir kurulabiliyor → `DropColumnsStep`.
   - **Hayır → ÖN-KOŞUL olarak beyan et ve denetle.** Örnek: dedup — `build_features` onu split ÖNCESİ tüm df'te
     yapıyor; orkestratöre koysak post-split X_train'de yapardık ve *"tüm df'i dedup'la sonra böl"* ≠ *"böl sonra
     train'i dedup'la"*. Konum taşınamıyor → upstream'de kalır, orkestratör onu **beyan olarak talep eder ve denetler**.

Aynı prensip, iki farklı sonuç; ayırt edici olan **konumun taşınabilirliği**.

**Guard tablosu DEĞİŞMEDİ** (6 dataset: `never_produced=0`, `produced_then_dropped` %100 açıklanmış) — simetrik
guard ve ön-koşullar yanlış-pozitif üretmiyor, `deployable=True` korundu.

### 2.20 TERMINUS PLANI (onaylı) + **T0 BİTTİ** — 80→85 test — `feature_builder`'a İLK DOKUNUŞ (additive)

**Omurga kararı:** terminus'ta iki tür değişiklik var — **yapısal** (kim çağırıyor) ve **istatistiksel** (#4/#5).
**Ayrılıyorlar.** Birlikte yapılırsa ROC oynadığında sebep fiks mi porting bug'ı mı ayırt edilemez.

| Faz | Ne | Davranış | Beklenen ΔROC |
|---|---|---|---|
| **T0** ✅ | Beyan yüzeyi + filtrelenmemiş meta | additive | — |
| **T1** | Gölge koşu (flag'li, kısa) | üretim çıktısı eski yol | — |
| **T2** | Yapısal cutover (`_clean_data` YERİNDE kalır) | metrik-NÖTR | **tam 0 (bit-birebir)** |
| **T3** | #4 impute relokasyonu | metrik-etkili | ≈0, veri-bağımlı |
| **T4** | #5 zscore/normalize relokasyonu | metrik-etkili | referans setlerde inert |
| **T5** | Eski `_tt` emekliliği + flag kaldırma | — | 0 |

**Kilitlenen kararlar (kullanıcı onayı):**
- **T1 gölge fazı: EVET, kısa ve amaçlı.** Asıl değeri T2 ile örtüşmüyor: T2 altı CSV'de, gölge **gerçek kullanıcı
  verisinde** ve asıl olarak **kapsam kapısını sınıyor** — beyan edilmemiş modalite (aggregate/ts/domain pack)
  gerçek veride çıkınca `deployable=False` doğru tetikleniyor mu? Birkaç koşu, sonra kapat.
- **Flag çıkış ölçütü: OLAY-TABANLI, "N koşu" değil.** Üç kapı: (a) gölgede sıfır divergence, (b) T2 altı dataset
  bit-birebir, (c) T3/T4 A/B raporlanmış ve kullanıcı onayından geçmiş. Üçü tamamsa T5. **Flag kaldırma AYRI
  backlog kalemi** (ölçütsüz flag kalıcı ikize döner — tasarımın en büyük riski).
- **T2 bit-birebir ŞART.** Önce determinizm kapısı (aynı arm iki kez → bit-birebir mi?). Değilse porting bug'ı
  vardır, T3'e GEÇİLMEZ. Gevşek "gürültü içinde" eşiği kullanılmıyor — T2'de hiçbir istatistik değişmediği için
  tam eşitlik ulaşılabilir olmalı.
- **ARTIFACT'İN T2'DEKİ YERİ (kilitlendi):** `fit_pipeline` artifact ÜRETİR ama **diske YAZILMAZ, kullanılmaz** —
  serialize AYRI faz. Gerekçe: T2'nin bit-birebir şartını serialize riskiyle bulandırma. **VE: artifact ASLA
  `ProjectStepModel.step_data`'ya GİRMEZ** — manifest adım başına `columns_out` taşıyor; worker'ın sanitize filtresi
  listeleri 5 satıra kırpar (sessiz bozulma) ve checkpoint payload'ı şişer. Artifact diske yazılır, state'te yalnız
  yol taşınır (`dump_split` deseni). Bu karar ŞİMDİ kilitlendi, serialize fazına ertelenmedi.

**T0 — YAPILAN (hepsi additive, davranış değişmedi):**
- **`assemble_pipeline_ctx(...)` (YENİ, SAF fonksiyon)** — beyan yüzeyi. Üç tür bilgi: yapısal ön-koşullar
  (`deduplicated`/`target_col`/`drop_columns`) · donmuş seçimler (ratio çiftleri, flag kolonları, zscore/normalize
  spec'leri, exclude/outlier) · kapsam olguları. `feature_report["pipeline_ctx"]` olarak yayınlanıyor.
  **MEKANİZMANIN KİLİDİ:** burada beyan edilmeyen olgu orkestratörde "bilinmiyor" sayılır → artifact
  `deployable=False`. Yani **bir olguyu eklemeyi unutmak "sessizce destekleniyor" sonucunu DOĞURAMAZ.**
  `test_pipeline_ctx_declares_every_scope_fact` unutmayı anında yakalar.
- **FAİL-OPEN KAPATILDI (T0'da bulundu):** `modality` build_features'ta tabular yolda **boş string**. Ham beyan
  edilseydi kapsam kuralı `bool(v)` üzerinden onu SESSİZCE geçerdi — *beyan var ama değer yok*, en kötü hâl.
  `assemble_pipeline_ctx` artık `modality or "tabular"` ile niyeti açıkça yazıyor (testli).
- **FİLTRELENMEMİŞ META BORCU KAPANDI (ratio dersinin tekrarı):** `missing_indicator_meta` + `derived_flag_meta`
  (`{üretilen: kaynak}`, **X_train'e göre filtrelenMEZ**). `auto_features`/`derived_flags` hayatta-kalan listesi
  olarak KALIYOR (kontrast). **Somut kanıt:** credit'te meta **2** `_isna` kaydediyor, filtreli liste **1** —
  `loan_int_rate_isna` kurulup selection tarafından düşürülmüş; filtreli liste frozen kaynağı olsaydı o kolon
  seçimden düşer ve groupby'ın ara çerçevesi eski yoldan saparadı. İki çit: kaynak-düzeyi
  (`test_flag_metas_are_unfiltered_supersets`) + davranışsal (`test_flag_meta_superset_on_real_data`).
  *(Not: burada ad-çözümleme ratio'dan FARKLI olarak güvenli — `_isna` suffix'i / `is_zero_` prefix'i SABİT,
  `ratio_a_b`'nin "hangi alt çizgi bölüyor" belirsizliği yok. Yine de meta yayınlanıyor: sorun ad-çözümleme değil,
  FİLTRELENME idi.)*
- **Envanter scripti artık TAKLİT ETMİYOR:** ctx tek kaynaktan — `feature_report["pipeline_ctx"]`. Böylece envanter
  üretimin fiilen kullanacağı kablolamayı ÖLÇÜYOR, benzerini değil.
- **ÖLÇÜT KORUNDU:** 6 dataset — `never_produced=0`, `produced_then_dropped` %100 açıklanmış, `deployable=True`.
- Testler: 5 yeni (85 toplam). Tüm `tests/` = **717 passed / 7 failed** (kalan 7 altyapı, değişmedi) →
  `feature_builder`'a ilk dokunuş regresyon üretmedi.

### 2.21 **T1 — GÖLGE KOŞU BİTTİ** (commit sende) — 85→91 test

`FEATURE_PIPELINE_MODE` (`legacy`|`shadow`|`orchestrator`, **varsayılan legacy**) + `src/services/inference/shadow.py`.
Eski yol çıktıyı üretmeye devam eder; `shadow` modunda orkestratör AYNI `X_train` üzerinde paralel koşar, fark
**loglanır, ASLA fırlatılmaz**. `build_features` diff'i minimal tutuldu (2 kanca) — T2'nin bit-birebir şartı için önemli.

- **Kanca konumu:** post-split zincir ÖNCESİ `X_train` yakalanır (yalnız shadow modunda kopyalanır → legacy'de
  maliyet SIFIR, testli). Pre-split adımlar bu çerçevede idempotent no-op → T2'nin şekliyle birebir aynı kurulum.
- **`pipeline_ctx` TEK KAYNAK:** hem gölge hem `feature_report` aynı `assemble_pipeline_ctx` çağrısını tüketir → ikisi sapamaz.
- **Sonuç `feature_report["shadow_report"]`'a düşer** (kanıt toplama), legacy'de `{}`.

**SONUÇ — 6 dataset, `verdict=identical`, kolon sayıları build_features ile birebir** (46·26·21·22·32·24),
hepsi `deployable=True`.

> **Bu, envanter scriptinden DAHA GÜÇLÜ bir kanıt ve §2.14'ün açık borcunu kapatıyor:** karşılaştırma artık
> `_old_oracle_sequence`'a (elle rekonstrüksiyon) değil, **`build_features`'ın GERÇEK `X_train` çıktısına** karşı.
> İlk kez *"fit_pipeline == build_features'ın kendisi"* gösterildi — §2.14'te "KANITLANMAYAN" diye işaretlediğimiz iddia.

**KAPSAM KAPISI GERÇEK PIPELINE'DA SINANDI (gölgenin ASIL amacı):**

| Senaryo | verdict | deployable |
|---|---|---|
| temiz (tabular) | `identical` | True |
| `modality=timeseries` | **`scope_rejected`** | False |
| plan'da portlanmamış op (`log`) | **`scope_rejected`** | False |
| plan'da portlu op (`zscore`) | `identical` | True |

Kapsam/ön-koşul reddi gölgede **hata değil BULGU** — ayrı verdict'lerle (`scope_rejected`/`precondition_rejected`),
"çöktü" diye değil. Üretim koşusu her durumda etkilenmedi.

**⚠️ DÜRÜSTLÜK NOTU — zscore'un `identical` çıkması #5'i DOĞRULAMAZ.** Gölge çerçevesinde `z_*` kolonu eski yol
tarafından (full-df, leaky) ZATEN üretilmiştir; `ZScoreStep.fit` `name in X_train.columns` görüp ATLAR → adım
**no-op** → eşitlik bedava gelir. Yani o satır yapısal yolun sağlamlığını gösterir, #5'in yeni değerinin eskiyle
aynı olduğunu DEĞİL (aynı olmaması gerekiyor, bilerek sapıyor). #5 T4'e kadar inert kalır; gerçek kanıtı
deterministik sızıntı-delta testidir. Kilit: `test_zscore_step_is_inert_in_shadow`.

**Testler (6 yeni):** gölge fırlatmaz (fit patlatıldı → `shadow_failed`, üretim etkilenmez) · kapsam reddi bulgu
olarak raporlanır · kolon/değer sapması ayrı verdict · varsayılan `legacy` · zscore inert dürüstlük kilidi.
Tüm `tests/` = **723 passed / 7 failed** (kalan 7 altyapı, değişmedi).

**T1 ÇIKIŞ DURUMU:** yerel 6 dataset'te sıfır divergence ✅. **Flag kapısı (a) için gereken: GERÇEK kullanıcı
projelerinde birkaç koşu** (`FEATURE_PIPELINE_MODE=shadow` ile worker) — orada aggregation/ts-lag/domain-pack
gibi yerel setlerin tetiklemediği yollar çıkacak. Sonra kapat.

---

## 3. Başarısız / iptal olan denemeler

- **#4 A/B ölçümü — etki SIFIR çıktı (ama dürüst bulgu).** churn: 0 null kolon → yapısal inert.
  credit: 2 null kolon (person_emp_length, loan_int_rate) ama **train-median = full-median tam olarak**
  (4.0 ve 10.99) → saf #4 etkisi **0**. Yani referans raporlar #4 yüzünden ŞİŞMEMİŞ. **Fix yine gerekli**
  (deployment'ta split yok, impute değeri bir yerden gelmeli + kayan-dağılımda patlar).
- **A/B harness confound'u yakalandı:** dış null-doldurma, `_add_missing_indicators`'ın (impute'tan
  ÖNCE çalışan) `<col>_isna` feature'larını üretmesini engelliyor → NEW kolda Δacc=0.0013 bu confound,
  impute değeri değil. Gerçek fix'te (split-sonrası impute) missing-indicator korunur. Medyanlar aynı
  olduğu için yeniden koşmak israf — bulgu kesin.
- **Tek-model (logistic_regression) A/B:** `_build_oof_blend_candidate` tek modelde 3 değer dönüp
  4 beklendiği için `ValueError`. Çözüm: iki deterministik model (logistic_regression + naive_bayes) —
  blend/stacking de deterministik, ΔROC saf sinyal.
- **Dataset karışıklığı (denetimde 3 kez):** telco_churn / creditcard / gerçek dataset karıştırıldı.
  Zararsızdı (gitignored çıktı) ama deployment öncesi harness'a dataset-imza guard'ı notu düşüldü
  (raw_input_signature bunu çözüyor).

---

## 4. Kısıtlamalar ve kurallar (MUTLAK)

1. **ASLA `git commit` çalıştırma.** Kullanıcı tüm commit'leri kendi atar. İstisna yok.
   Kod parçaları AYRI gösterilir ki ayrı commit'lensin (protocol / impute / parity = 3 commit).
2. **Tam dinamik — hardcode YOK.** Kolon/sınıf/dataset adı gömme; her şey run'dan türer.
3. **Deterministik hesaplanabilen LLM'e verilmez.** Sayı/eşik/format kod işi.
4. **TR + EN her ikisi; TR-leak guard yeşil kalmalı** (rapor tarafı).
5. **Split-öncesi tüm-df istatistik = sızıntı.** fit yalnız X_train, transform yalnız param;
   tüm-df'den istatistik hesaplayan yol imza düzeyinde bırakılmaz.
6. **Eski `_tt` / `_clean_data` kodu DURSUN** — yeni protokol paralel kurulur, eski yol parity
   referansı. Tüm adımlar + orkestratör parity sağlayınca eski yol emekliye ayrılır. Erken silme.
7. **Parity baştan, sonda değil.** Her yeni adımda parity testi büyür; 12 adım sonra iğne aranmaz.
8. **Sızıntı doğrulaması — İLKE GÜNCELLENDİ (2026-07-23).** Eski metin: *"yalnız sızıntı-düzelten adımlar
   A/B taşır"*. Yeni metin: **"bir sızıntı DETERMİNİSTİK olarak kanıtlanabiliyorsa A/B YERİNE o kullanılır;
   A/B yalnız deterministik kanıt mümkün değilken."** Gerekçe: ROC'un ±0.004 gürültü tabanı çoğu sızıntı
   deltasından büyük → A/B o farkı zaten OKUYAMAZ (#4'ün A/B'si sıfır çıktı; #5'te delta kanıtı kesin sonuç
   verdi). A/B gerektiğinde: deterministik model (LR+NB), aynı seed, dürüst büyüklük raporu, sadakat guard'ı
   (n_train/n_test eşleşmesi).
9. **Tasarım → onay → yaz → doğrula** ritmi. Bir adımı yazmadan somut planını göster, onay bekle.
10. **Sandbox:** test/harness'ta `API_DATABASE_URL="sqlite:///..."` + `PYTHONIOENCODING=utf-8`
    (remote app DB'ye asla dokunma; Türkçe cp1252 çökmesini önle).

---

## 5. Sıradaki görevler

### Yakın (Katman 1 refactor — A yolu) — **TÜM ADIMLAR TAMAM (12 adım + orkestratör), 48/48 test**
BİTEN: ImputeStep(#4), **ZScore/Normalize(#5)**, RatioConstruct, RatioFinalize, CountEncode, GroupbyDeviation,
encode_route+OneHot+TargetEncode, InfImpute, scaler_route+Robust/Standard, SelectDrop(terminus #3), orchestrator.
✅ Encoder ailesi ✅ Scaler ailesi ✅ Selection(terminus) ✅ #4 impute ✅ #5 zscore/normalize — HEPSİ BİTTİ.
**Sıradaki = Katman 1 KAPANIŞI (aşağıda 4-6): terminus (eski `_tt` delege) → Artifact alanları → serialize.**
4. **Kalan:** `normalize_column_names`, `ratio_construct` (zincir stand-in'ini gerçek adımla değiştirir),
   `missing_indicator`, `zscore`/`normalize` (**#5 fix — A/B taşır**), `currency_normalize`, `drop_duplicates` (skip).
2. **`zscore`/`normalize` op'ları (#5 fix):** `_FEATURE_OPS`'tan çıkarılıp parametric adıma
   dönüştürülür (train mean/std/min/max). #4 ile aynı disiplin.
3. **Orkestratör:** `fit_pipeline` + `transform_pipeline`. İkisi aynı `step.transform`'u çağırır.
   `transform_pipeline` manifest'i sırayla replay eder, `final_features`'a reindex.
4. **Parity — orkestratör düzeyi:** `transform_pipeline(X_train_raw) == fit_pipeline'ın X_train'i`.
   Tüm adımlar geçince eski `_tt`/`_clean_data` emekliye.
5. **Diğer alanların doldurulması:** `raw_input_signature` (domain-aware guard), `training_reference`
   (PSI reference + baseline metrik kopyası), `unseen_category_policy`, `model_ref` (registry).
6. **Serialize katmanı:** Artifact → disk (inline JSON + sklearn pickle'lar + sklearn versiyon kontrolü).

### Orta (deployment)
7. **Dataset-imza guard'ı:** raw_input_signature ile inference-başı doğrulama (required eksik/tip-uyumsuz
   → abort). Dataset karışıklığını da çözer.
8. **Model registry:** uuid + proje + versiyon; `data/models/model_<uuid>.pkl`'i artifact'e bağla
   (başıboş .pkl yerine).

### Sonra (Katman 2 — monitoring)
9. **PSI (input drift):** training_reference.per_feature (raw-space) referansıyla prod dağılımı kıyası.
10. **CBPE (performance estimation):** baseline_metrics çapasıyla drift-sonrası düşüş tahmini.
11. **Görülmemiş-kategori sayacı:** unseen_category_policy monitörü (veri-kalite sinyali).

### Rapor tarafı (açık)
- Faz 3 commit'leri kullanıcıda bekliyor (denetim kapandı, temiz).

### Katman 1 DIŞI backlog (2026-07-23'te işaretlendi)
- **TEKRARLI ID + TARGET ENCODING — bugün ÜRETİMDE duran ML-doğruluk açığı (Katman 1 borcu DEĞİL).**
  Tekrarlı bir kimlik kolonu (örn. 40 müşteri × 10 işlem) `_infer_semantic_type`/meta tarafından "category"
  etiketlenir (identifier eşiği near-unique >0.95, tekrarlı ID onu geçmez) → kardinalite >10 olduğu için
  **TargetEncoder**'a gider → kolon artık *müşteri-düzeyi hedef ortalaması*dır. Ölçüldü: `y` ile korelasyon
  **0.292** — leakage dedektörünün eşiğinin ALTINDA → `final_features`'a girer, **modele girer**. Sessiz.
  (Benzersiz ID'de encoded == `y` birebir olduğu için dedektör yakalıyor; asıl risk TEKRARLI olanda.)
  **Bu yüzey `build_features`'ta da AYNEN var** — supervised yolda `_detect_leakage_tt` de encode'dan SONRA
  çalışır (2443), yani kardinalite tabanlı ID kuralı ham ID'yi göremez; her iki yolda da tek koruma
  semantic_type drop'udur. Yani port regresyonu DEĞİL, paylaşılan açık. Kullanıcı ayrı ele alacak.
- **`tests/test_pg.py` üretim Postgres'ine bağlanıyor — GÜVENLİK/HİJYEN, CLAUDE.md ihlali.** Collection anında
  `188.132.198.201:5432`'ye gidiyor (`API_DATABASE_URL` sandbox'a ayarlıyken bile — bağlantı dizesi hardcoded),
  **şifreyi konsola basıyor**, ve 30s timeout'la TÜM test koşusunu durduruyor (`--ignore` gerekiyor). CLAUDE.md:
  *"Tests/migrations must NEVER hit the remote app DB."* → Kullanıcı ele alacak.
- **Kalıcı-başarısız: 8 sanılıyordu, 9'du → 2'si düzeltildi → 7 kaldı.** `tests/` = **696 passed / 7 failed /
  1 skipped** (test_pg.py hariç). Katman 1 işi bunların HİÇBİRİNE katkı yapmadı (tracked tek dosya değişmedi);
  kalan 7'nin hepsi altyapı (db mapper × 6, pytest-asyncio × 1) — rapor/ML davranışıyla ilgili kırık YOK.
  - `test_db_logging.py` × 6 — SQLAlchemy mapper init (`CSVFileModel` locate edilemiyor). Ayrı iş.
  - `test_pg2.py::test` — `async def` + pytest-asyncio eksik. Ayrı iş.
  - ✅ **`test_error_analysis_layered::test_bolumde_ham_deger_gosteriliyor` — DÜZELTİLDİ.** Rapor turundan
    (c3a54bc) geliyordu ama davranış regresyonu DEĞİLdi: Faz 3 ondalık disiplini (`_fmt_value`, ≥1 → 3dp) metne
    `41.975` basıyor, test ham `41.9747` string'ini birebir arıyordu. Ham değer hâlâ gösteriliyor (ölçekli değil)
    → assert `_fmt_value`'ya bağlandı + "ölçekli değer sunulmasın" iddiası biçimlendirilmiş formu da kapsayacak
    şekilde güçlendirildi.
  - ✅ **`test_leakage_guard::test_metric_semantic_exempts_even_if_int` — DÜZELTİLDİ (fixture).** Rapor turundan
    DEĞİL, 9 commit önceden (998d779, 2026-06-29): o commit "near-unique bir int kolon, semantic_type='metric'
    olsa bile identifier'dır" kuralını bilerek getirdi (customer_id yanlış-pozitifi). Test ondan ÖNCE yazılmıştı
    (af5bc35). Eski fixture `randint(0,100000)` × 1500 satır → ~%100 benzersiz, yani gerçek bir ölçüm gibi DEĞİL
    tam da bir ID gibi davranıyordu. Fixture gerçekçi kardinaliteye çekildi (`randint(0,400)` → ~400 uniq/1500);
    muafiyetin asıl hedefi budur. Kural KORUNDU (gevşetilmedi).

---

## 6. Kilit dosyalar

| Dosya | Rol |
|---|---|
| `src/services/feature_builder.py` | Mevcut FE pipeline (2525 satır); `build_features`, `_clean_data`, `_*_tt`, `_FEATURE_OPS`. Eski yol — DOKUNULMUYOR. |
| `src/services/ml_trainer.py` | `train_model`, eşik (`_select_oof_threshold`), `joblib.dump` model persist. |
| `src/services/inference/protocol.py` | YENİ — TransformStep/TrainOnlyStep, Artifact. |
| `src/services/inference/steps.py` | YENİ — transform adımları (şimdilik ImputeStep). |
| `src/services/inference/orchestrator.py` | YENİ — PIPELINE/fit_pipeline/transform_pipeline + **üretim-kanıtı guard'ı**. |
| `tests/test_layer1_impute_parity.py` | YENİ — parity + guard (57 test, her adımda büyüyecek). |
| `scripts/layer1_coverage_inventory.py` | YENİ — kapsam envanteri ("kod söylesin"). Her port turundan sonra koş; **liste boşalınca Katman 1 kapsam borcu kapanır**. |
| `scripts/offline_report.py` | LLM'siz deterministik rapor/pipeline harness'ı (build_state + run_pipeline). |

**Not:** A/B ölçüm scripti scratchpad'de (throwaway, commit'lenmez); sonucu §3'te.
