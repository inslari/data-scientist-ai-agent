# Proje (Project) API Dökümantasyonu

Bu dökümantasyon, kullanıcıların yaptığı veri analizlerini, yazdıkları query'leri ve seçtikleri veri gruplarını (DataGroup) bir araya getirip kaydetmelerini sağlayan **Project** API uç noktalarını açıklamaktadır.

Ayrıca bu güncellemeyle beraber; Projeleri **Arka Planda Çalıştırma** (Background Task) ve LLM'in analizini adım adım canlı (polling) olarak çekebileceğiniz **Steps** uç noktaları da eklenmiştir.

---

## 1. Yeni Proje Oluşturma

* **URL:** `/projects`
* **Method:** `POST`

### İstek (Request) Body
```json
{
  "title": "Gelir Tahmin Analizi",
  "description": "2023 verileri",
  "query": "Predict the income level...",
  "data_group_id": 2
}
```

### Yanıt (200 OK)
Dönen yanıtta yeni eklenen `status` ("idle"), `report` (null) ve `error` (null) alanlarını görebilirsiniz.
```json
{
  "id": 1,
  "title": "Gelir Tahmin Analizi",
  "status": "idle",
  "report": null,
  "error": null,
  ...
}
```

---

## 2. Projeyi Çalıştırma (Run AI Graph)

Seçili projeye ait Yapay Zeka analiz sürecini arka planda başlatır.

* **URL:** `/projects/{id}/run` (Opsiyonel olarak: `/projects/{id}/run?force_restart=true`)
* **Method:** `POST`

**Parametreler:**
* `force_restart` (boolean, opsiyonel): Eğeri `true` gönderilirse, önceden var olan tüm `steps` (adım) kayıtları ve checkpointler silinir. Proje tamamen sıfırdan, en baştan çalışmaya başlar. Varsayılan olarak `false`'tur (proje failed durumundaysa kaldığı yerden devam eder).

### Başarılı Yanıt (200 OK)
Süreci başlattığınıza dair projenin o anki durumunu (status: "running") döner. İşlem arkada asenkron olarak 1-5 dakika kadar sürebilir. İstemci kitlenmez.
```json
{
  "id": 1,
  "status": "running",
  ...
}
```

---

## 3. Adım Adım İlerleme Çekme (Polling)

Proje "running" durumundayken, her bir AI Agent nodu (SQL Node, ML Node vb.) bittikçe sonuçlarını bu uç noktadan anlık olarak saniye saniye çekebilirsiniz. Eğer proje "completed" ise eski adımların tamamını liste olarak döner.

* **URL:** `/projects/{id}/steps`
* **Method:** `GET`

### Başarılı Yanıt (200 OK)
```json
[
  {
    "id": 1,
    "project_id": 1,
    "node_name": "intent_node",
    "step_data": {
      "intent": "classification"
    },
    "created_at": 1700000005
  },
  {
    "id": 2,
    "project_id": 1,
    "node_name": "sql_node",
    "step_data": {
      "sql_query_raw": "SELECT * FROM adult_income LIMIT 10",
      "sql_query_results": [{"age": 25, "income": "<=50K"}]
    },
    "created_at": 1700000010
  }
]
```

**Tavsiye Edilen Kullanım:** `/projects/{id}/run` işlemine istek attıktan sonra frontend tarafında `setInterval` ile her saniye `/projects/{id}/steps` uç noktasına istek atarak ekrandaki "Chat" akışını güncelleyebilirsiniz. Proje `status`'ü "completed" veya "failed" olduğunda interval'ı durdurabilirsiniz. (Bunun için bir defa da `/projects/{id}` çekip durumu kontrol etmeniz gerekebilir).

---

## 4. Tüm Projeleri Listeleme / Detay Getirme / Silme

* **Tümünü Listele:** `GET /projects`
* **Tekil Detay:** `GET /projects/{id}`
* **Silme:** `DELETE /projects/{id}` (Eğer proje silinirse ona ait `steps` kayıtları da CASCADE mantığıyla otomatik olarak silinir).
