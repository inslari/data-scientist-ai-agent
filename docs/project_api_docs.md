# Proje (Project) API Dökümantasyonu

Bu dökümantasyon, kullanıcıların yaptığı veri analizlerini, yazdıkları query'leri ve seçtikleri veri gruplarını (DataGroup) bir araya getirip kaydetmelerini sağlayan **Project** API uç noktalarını açıklamaktadır.

---

## 1. Yeni Proje Oluşturma

Yeni bir proje kaydı yaratır.

* **URL:** `/projects`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek (Request) Body
Tüm alanlar `title` hariç isteğe bağlıdır (opsiyonel).
```json
{
  "title": "Gelir Tahmin Analizi",
  "description": "2023 verileriyle yapılan denemeler",
  "query": "Predict the income level of individuals in this dataset...",
  "data_group_id": 2
}
```

### Başarılı Yanıt (200 OK)
```json
{
  "title": "Gelir Tahmin Analizi",
  "description": "2023 verileriyle yapılan denemeler",
  "query": "Predict the income level of individuals in this dataset...",
  "data_group_id": 2,
  "id": 1,
  "created_at": 1700000000,
  "updated_at": 1700000000,
  "data_group": null
}
```

---

## 2. Tüm Projeleri Listeleme

Sistemdeki kayıtlı tüm projeleri getirir.

* **URL:** `/projects`
* **Method:** `GET`

### Başarılı Yanıt (200 OK)
Dizi (array) halinde tüm projeleri döner.
```json
[
  {
    "id": 1,
    "title": "Gelir Tahmin Analizi",
    "description": "...",
    "query": "...",
    "data_group_id": 2,
    "created_at": 1700000000,
    "updated_at": 1700000000,
    "data_group": null
  }
]
```

---

## 3. Tekil Proje Detayı Getirme

Belirli bir projenin tüm detaylarını (varsa `data_group` içeriğiyle beraber) getirir.

* **URL:** `/projects/{id}`
* **Method:** `GET`

### Başarılı Yanıt (200 OK)
JSON Objesi döner.

### Hata Yanıtı (404 Not Found)
* Belirtilen ID'ye sahip bir proje bulunamazsa döner.

---

## 4. Proje Güncelleme

Var olan bir projenin bilgilerini günceller. Sadece değiştirmek istediğiniz alanları göndermeniz yeterlidir.

* **URL:** `/projects/{id}`
* **Method:** `PUT`
* **Content-Type:** `application/json`

### İstek (Request) Body
Tüm alanlar opsiyoneldir. Sadece güncellenecek olanları gönderin.
```json
{
  "query": "Bu sefer yeni bir analiz metni..."
}
```

### Başarılı Yanıt (200 OK)
Güncellenmiş projenin JSON detayını döner.
```json
{
  "id": 1,
  "title": "Gelir Tahmin Analizi",
  "description": "2023 verileriyle yapılan denemeler",
  "query": "Bu sefer yeni bir analiz metni...",
  "data_group_id": 2,
  "created_at": 1700000000,
  "updated_at": 1700000150,
  "data_group": null
}
```

---

## 5. Proje Silme

Belirli bir projeyi veritabanından kalıcı olarak siler.

* **URL:** `/projects/{id}`
* **Method:** `DELETE`

### Başarılı Yanıt (200 OK)
```json
{
  "status": "success",
  "message": "Project with id 1 deleted successfully"
}
```

### Hata Yanıtı (404 Not Found)
* Belirtilen ID'ye sahip bir proje bulunamazsa döner.

---

> [!NOTE]
> Projenin bağlı olduğu `DataGroup` silinirse (örneğin MCP bağlantısı veya CSV silinirse), projeler **silinmez**. Sadece projenin içerisindeki `data_group_id` alanı otomatik olarak `null` değerine düşer (SET NULL kuralı). Böylece yaptığınız analizler ve girdiğiniz query metinleri kaybolmaz.
