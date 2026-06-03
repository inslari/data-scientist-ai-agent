# Data Group (Veri Grupları) API Dökümantasyonu

Bu dökümantasyon, tüm veri kaynaklarını (MCP veritabanı bağlantıları veya toplu yüklenen CSV dosyaları) tek bir çatı altında toplayan `DataGroup` mimarisi için oluşturulan API uç noktalarını açıklar. Frontend uygulamasında sol menüde veya liste ekranlarında tüm veri kaynaklarını bir arada göstermek için bu yapıyı kullanmalısınız.

## Mimari Mantığı (Kısa Özet)
* Bir kullanıcı yeni bir MCP bağlantısı eklediğinde veya CSV yüklediğinde, API arka planda otomatik olarak bir **DataGroup** nesnesi oluşturur.
* Bu nesnenin `source_type` alanı bağlantının tipini belirler: `"mcp"` veya `"csv"`.
* Veri grubunun içeriği tipe göre değişkenlik gösterir:
  * `source_type == "mcp"` ise `mcp_config` alanı doludur.
  * `source_type == "csv"` ise `csv_files` alanı (dizi olarak) doludur.

---

## 1. Tüm Veri Kaynaklarını (Data Groups) Listeleme

Sisteme kayıtlı tüm veri gruplarını, içerdikleri alt dosyalar veya MCP konfigürasyonları ile birlikte döner.

* **URL:** `/data-groups`
* **Method:** `GET`

### Başarılı Yanıt (200 OK)
Tüm kaynakları tek bir dizide getirir. Frontend bu diziyi dönerken `source_type`'a bakarak farklı ikonlar (örn: veritabanı ikonu veya excel ikonu) gösterebilir.

```json
[
  {
    "name": "Production Postgres DB",
    "source_type": "mcp",
    "id": 1,
    "created_at": 1700000100,
    "mcp_config": {
      "name": "Production Postgres DB",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://user:pass@localhost/db"],
      "url": null,
      "env": null,
      "id": 1,
      "data_group_id": 1
    },
    "csv_files": []
  },
  {
    "name": "CSV Upload - 2026-06-02 14:30:00",
    "source_type": "csv",
    "id": 2,
    "created_at": 1700000500,
    "mcp_config": null,
    "csv_files": [
      {
        "id": 1,
        "filename": "sales_q1.csv",
        "file_path": ".../data/csvs/uuid_sales_q1.csv",
        "file_size": 15000,
        "row_count": 300,
        "columns_schema": [...],
        "created_at": 1700000500,
        "data_group_id": 2
      },
      {
        "id": 2,
        "filename": "sales_q2.csv",
        "file_path": ".../data/csvs/uuid_sales_q2.csv",
        "file_size": 12000,
        "row_count": 250,
        "columns_schema": [...],
        "created_at": 1700000500,
        "data_group_id": 2
      }
    ]
  }
]
```

---

## 2. Veri Grubunu Silme (Cascade Delete)

Belirtilen `group_id`'ye sahip veri kaynağını tamamen siler. Eğer bu bir MCP ise altındaki konfigürasyonu, CSV ise altındaki tüm fiziksel dosyaları (`.csv`) ve veritabanı kayıtlarını otomatik ve güvenli bir şekilde (Cascade) siler. **(Frontend'de silme işlemi yaparken alt dosyaların ID'si yerine bu uç noktayı kullanmalısınız.)**

* **URL:** `/data-groups/{group_id}`
* **Method:** `DELETE`

### Başarılı Yanıt (200 OK)
```json
{
  "status": "success",
  "message": "Data group with id 2 deleted successfully"
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `{group_id}` bulunamazsa.

---

## CSV Yükleme Uç Noktasında Yapılan Değişiklikler

`POST /csv/upload` uç noktası artık isteğe bağlı bir `group_name` parametresi (form data içinde string olarak) almaktadır. 
* Birden fazla dosya yüklerseniz, hepsi oluşturulan o grup adı altında toplanır.
* Eğer `group_name` göndermezseniz, API otomatik olarak `"CSV Upload - YYYY-MM-DD HH:MM:SS"` şeklinde bir grup ismi verir.

**Örnek FormData (Toplu CSV Yükleme):**
* `files`: [dosya1.csv, dosya2.csv]
* `group_name`: "Q1 Satış Raporları"

---

## MCP Ekleme Uç Noktasında Yapılan Değişiklikler
`POST /mcp/config` uç noktasının request (istek) body'si değişmemiştir. Ancak sunucu, gelen kaydı önce yeni bir `DataGroup` nesnesine bağlar. Kaydettiğiniz isme (`name`) sahip bir veri kaynağı otomatik olarak liste ekranında (`/data-groups`) görünecektir.
