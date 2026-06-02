# CSV API Dökümantasyonu

Bu dökümantasyon, Data Scientist AI Agent sistemine CSV dosyalarını yüklemek, listelemek, silmek ve dosyaların analizlerini (şema, örnek veri, kolon istatistikleri) almak için oluşturulan API uç noktalarını açıklar. Frontend entegrasyonu için hazırlanmıştır.

## 1. CSV Dosyası Yükleme

Kullanıcıdan alınan bir veya birden fazla CSV dosyasını sunucuya yükler. Dosyalar sisteme kaydedilirken arka planda Pandas kütüphanesi kullanılarak satır sayıları, dosya boyutları ve kolon veri tipleri otomatik olarak çıkarılır ve veritabanına kaydedilir.

* **URL:** `/csv/upload`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`

### İstek (Request)

Form verisi (FormData) içinde `files` anahtarı ile (liste şeklinde) birden fazla `.csv` uzantılı dosya gönderilebilir.

### Başarılı Yanıt (200 OK)
Dosya bilgilerini ve otomatik çıkarılan ön şemaları bir liste halinde döner. Ayrıca başarısız yüklemeler (parse hataları vs.) varsa `errors` dizisinde raporlanır.
```json
{
  "status": "success",
  "message": "Successfully uploaded 1 files. (0 errors)",
  "data": [
    {
      "id": 1,
      "filename": "data.csv",
      "file_path": "/path/to/data/csvs/uuid_data.csv",
      "file_size": 10245,
      "row_count": 150,
      "columns_schema": [
        {"name": "id", "type": "int64"},
        {"name": "name", "type": "object"},
        {"name": "score", "type": "float64"}
      ],
      "created_at": 1700000000
    }
  ],
  "errors": []
}
```

### Hata Yanıtları
* `400 Bad Request`: Yüklenen dosya `.csv` uzantılı değilse veya Pandas dosyayı okurken parse hatası (bozuk format vb.) alırsa döner.

---

## 2. Tüm CSV Dosyalarını Listeleme

Sisteme daha önceden yüklenmiş tüm CSV dosyalarının temel bilgilerini getirir.

* **URL:** `/csv/files`
* **Method:** `GET`

### Başarılı Yanıt (200 OK)
```json
[
  {
    "id": 1,
    "filename": "data.csv",
    "file_path": "/path/to/data/csvs/uuid_data.csv",
    "file_size": 10245,
    "row_count": 150,
    "columns_schema": [...],
    "created_at": 1700000000
  }
]
```

---

## 3. Belirli Bir CSV Dosyasının Bilgilerini Getirme

ID'si belirtilen CSV dosyasının detaylarını döner.

* **URL:** `/csv/{file_id}`
* **Method:** `GET`

### Başarılı Yanıt (200 OK)
İlgili dosyanın JSON nesnesini döner.

### Hata Yanıtları
* `404 Not Found`: Verilen `{file_id}` veritabanında bulunamazsa döner.

---

## 4. CSV Dosyasını Silme

Belirtilen CSV dosyasının kaydını veritabanından siler ve aynı zamanda fiziksel dosyayı diskten (sunucudan) tamamen kaldırır.

* **URL:** `/csv/{file_id}`
* **Method:** `DELETE`

### Başarılı Yanıt (200 OK)
```json
{
  "status": "success",
  "message": "CSV file with id 1 deleted successfully"
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `{file_id}` veritabanında bulunamazsa döner.

---

## 5. CSV Şemasını Getirme

Dosya yüklenirken analiz edilen ve veritabanına kaydedilen kolon isimlerini ve veri tiplerini döner.

* **URL:** `/csv/{file_id}/schema`
* **Method:** `POST`
* **Content-Type:** `application/json`

*(Herhangi bir Request Body parametresi göndermeye gerek yoktur)*

### Başarılı Yanıt (200 OK)
```json
{
  "status": "success",
  "data": [
    {"name": "id", "type": "int64"},
    {"name": "name", "type": "object"},
    {"name": "score", "type": "float64"}
  ],
  "message": null
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `{file_id}` veritabanında bulunamazsa döner.

---

## 6. CSV'den Örnek Veri Çekme

Pandas kütüphanesi kullanarak dosyadan baştan itibaren belirtilen limit kadar (`nrows`) satır okur ve JSON formatında döner. Arayüzde veri tablosunu önizlemek için kullanılır.

* **URL:** `/csv/{file_id}/sample-data`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)
```json
{
  "limit": 10   // (Opsiyonel) Çekilecek maksimum satır sayısı (Varsayılan: 10)
}
```

### Başarılı Yanıt (200 OK)
`columns` dizisi kolon adlarını, `rows` dizisi ise veri objelerini içerir.
```json
{
  "status": "success",
  "data": {
    "columns": ["id", "name", "score"],
    "rows": [
      {"id": 1, "name": "Ali", "score": 95.5},
      {"id": 2, "name": "Veli", "score": 80.0}
    ]
  },
  "message": null
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `{file_id}` veritabanında bulunamazsa döner.
* `200 OK` (Error Status): Dosya okunurken veya parse edilirken hata olursa `"status": "error"` ve `message` alanı ile yanıt döner.

---

## 7. Kolon İstatistiklerini Getirme

Pandas kütüphanesinin `.describe(include='all')` fonksiyonunu kullanarak dosyadaki tüm kolonlar hakkında veri analizi istatistiklerini (min, max, ortalama, standart sapma, null frekansı vb.) hesaplar ve döner.

* **URL:** `/csv/{file_id}/column-stats`
* **Method:** `POST`
* **Content-Type:** `application/json`

*(Herhangi bir Request Body parametresi göndermeye gerek yoktur)*

### Başarılı Yanıt (200 OK)
Her bir kolon için hesaplanan istatistikler obje listesi olarak `data` içerisinde döner. Boş olan (hesaplanamayan) alanlar `null` gelir.
```json
{
  "status": "success",
  "data": [
    {
      "column_name": "id",
      "count": 150.0,
      "mean": 75.5,
      "std": 43.44,
      "min": 1.0,
      "25%": 38.25,
      "50%": 75.5,
      "75%": 112.75,
      "max": 150.0,
      "null_count": 0,
      "dtype": "int64",
      "unique": null,
      "top": null,
      "freq": null
    },
    {
      "column_name": "name",
      "count": 150,
      "unique": 140,
      "top": "Ali",
      "freq": 3,
      "mean": null,
      "std": null,
      "min": null,
      "25%": null,
      "50%": null,
      "75%": null,
      "max": null,
      "null_count": 0,
      "dtype": "object"
    }
  ],
  "message": null
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `{file_id}` veritabanında bulunamazsa döner.
* `200 OK` (Error Status): İstatistikler hesaplanırken bellek (memory) aşımı vb. bir hata olursa `"status": "error"` ve `message` alanı ile döner.
