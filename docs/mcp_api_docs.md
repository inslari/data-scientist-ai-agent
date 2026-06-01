# MCP API Dökümantasyonu

Bu dökümantasyon, Data Scientist AI Agent sistemine Model Context Protocol (MCP) bağlantılarını eklemek, listelemek ve test etmek için oluşturulan API uç noktalarını açıklar. Frontend entegrasyonu için hazırlanmıştır.

## 1. Yeni MCP Konfigürasyonu Ekleme

Kullanıcıdan alınan yeni bir MCP bağlantı bilgisini sisteme kaydeder. Bağlantı türü lokal komut (`stdio`) veya uzak sunucu (`sse`) olabilir.

* **URL:** `/mcp/config`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)

```json
{
  "name": "string",       // (Zorunlu) Bağlantı için benzersiz bir isim (Örn: "local-db-mcp")
  "type": "string",       // (Zorunlu) Sadece "stdio" veya "sse" değerlerini alabilir
  "command": "string",    // (Opsiyonel) type "stdio" ise zorunlu. Çalıştırılacak ana komut (Örn: "uv", "npx", "python")
  "args": ["string"],     // (Opsiyonel) Komuta geçilecek argüman listesi (Örn: ["run", "server.py"])
  "url": "string",        // (Opsiyonel) type "sse" ise zorunlu. Uzak MCP sunucusunun tam URL'i
  "env": {                // (Opsiyonel) Sunucuya geçilecek çevresel (environment) değişkenler
    "API_KEY": "secret"
  }
}
```

### Başarılı Yanıt (200 OK)
Veritabanına kaydedilen veriyi, oluşan `id` değeri ile birlikte döner.
```json
{
  "name": "local-db-mcp",
  "type": "stdio",
  "command": "python",
  "args": ["server.py"],
  "url": null,
  "env": null,
  "id": 1
}
```

### Hata Yanıtları
* `400 Bad Request`: Aynı isimde ("name") başka bir kayıt zaten varsa döner.

---

## 2. Tüm MCP Konfigürasyonlarını Listeleme

Sisteme daha önceden kaydedilmiş tüm MCP bağlantı bilgilerini getirir.

* **URL:** `/mcp/config`
* **Method:** `GET`

### Başarılı Yanıt (200 OK)
```json
[
  {
    "name": "local-db-mcp",
    "type": "stdio",
    "command": "python",
    "args": ["server.py"],
    "url": null,
    "env": null,
    "id": 1
  },
  {
    "name": "remote-cloud-mcp",
    "type": "sse",
    "command": null,
    "args": null,
    "url": "https://mcp.example.com/sse",
    "env": {"TOKEN": "1234"},
    "id": 2
  }
]
```

---

## 3. MCP Konfigürasyonunu Güncelleme

Sisteme kaydedilmiş belirli bir MCP konfigürasyonunu (ID'ye göre) günceller. İstekte sadece değiştirilmek istenen alanların gönderilmesi yeterlidir.

* **URL:** `/mcp/config/{config_id}`
* **Method:** `PUT`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)
Sadece güncellenmesi istenen alanlar gönderilebilir. Örnek:
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-postgres"]
}
```

### Başarılı Yanıt (200 OK)
Güncellenmiş veriyi döner.

### Hata Yanıtları
* `404 Not Found`: Verilen `{config_id}` veritabanında bulunamazsa döner.
* `400 Bad Request`: Güncellenmek istenen yeni `name` başka bir kayıtta zaten kullanılıyorsa döner.

---

## 4. MCP Konfigürasyonunu Silme

Sisteme kaydedilmiş belirli bir MCP konfigürasyonunu siler.

* **URL:** `/mcp/config/{config_id}`
* **Method:** `DELETE`

### Başarılı Yanıt (200 OK)
```json
{
  "status": "success",
  "message": "MCP config with id 1 deleted successfully"
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `{config_id}` veritabanında bulunamazsa döner.

---

## 5. MCP Bağlantısını Test Etme

Sisteme kaydedilmiş belirli bir MCP konfigürasyonuna (stdio veya sse) **gerçek bir MCP istemcisi** ile bağlanmayı dener. Başarılı olursa, sunucunun sağladığı **tüm araçların (tools) listesini** ve şemalarını döner.

* **URL:** `/mcp/test-connection`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)

```json
{
  "config_id": 1   // (Zorunlu) Test edilecek bağlantının ID değeri
}
```

### Başarılı Yanıt (200 OK)
Başarılı bağlantılarda `tools` dizisi içinde her bir aracın ismi, açıklaması ve beklediği parametreler (`inputSchema`) yer alır.
```json
{
  "status": "success",
  "message": "Successfully connected to MCP server. Found 2 tools.",
  "tools": [
    {
      "name": "run_sql_query",
      "description": "Execute a SELECT query...",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        }
      }
    }
  ]
}
```

### Başarısız Test Yanıtı (200 OK ama Hata Durumu)
Uç noktaya ulaşılamazsa, komut bulunamazsa veya **sunucu doğru araçlara (get_schema, get_column_stats, run_sql_query) sahip değilse** uygulama hata fırlatmaz (500 vb. dönmez), bağlantının başarısız olduğu bilgisini verir:
```json
{
  "status": "error",
  "message": "Connected to MCP server, but it is NOT the correct server. Missing required tools: get_column_stats",
  "tools": [
    {
      "name": "get_schema",
      "description": "...",
      "inputSchema": {}
    }
  ]
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `config_id` veritabanında bulunamazsa döner.
* `400 Bad Request`: İlgili bağlantı tipinin gerektirdiği parametre (stdio için `command`, sse için `url`) eksik kaydedilmişse döner.

---

## 6. Veritabanı Şemasını Getirme

Kayıtlı bir MCP sunucusuna bağlanır ve `get_schema` aracını çalıştırarak bağlı veritabanındaki **tüm tabloların yapısını (şema / DDL bilgisi)** döner. Frontend'de kullanıcıya hangi tablo ve kolonların mevcut olduğunu göstermek için kullanılır.

* **URL:** `/mcp/schema`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)

```json
{
  "config_id": 1   // (Zorunlu) Kullanılacak MCP bağlantısının ID değeri
}
```

### Başarılı Yanıt (200 OK)

`data` alanı, MCP sunucusunun döndürdüğü ham şema metnini içerir. Her tablo için `CREATE TABLE` DDL ifadesi yer alır.

```json
{
  "status": "success",
  "data": "Database: 'default'\nDatabase Schema:\n- Table 'adult_income': CREATE TABLE \"adult_income\" (\n\"age\" INTEGER,\n  \"workclass\" TEXT,\n  \"fnlwgt\" INTEGER,\n  \"education\" TEXT,\n  \"education.num\" INTEGER,\n  \"marital.status\" TEXT,\n  \"occupation\" TEXT,\n  \"relationship\" TEXT,\n  \"race\" TEXT,\n  \"sex\" TEXT,\n  \"capital.gain\" INTEGER,\n  \"capital.loss\" INTEGER,\n  \"hours.per.week\" INTEGER,\n  \"native.country\" TEXT,\n  \"income\" TEXT,\n  \"income_binary\" INTEGER\n)\n- Table 'steam_game_clean': CREATE TABLE steam_game_clean (\n\tappid BIGINT,\n\tname TEXT,\n\ttype TEXT,\n\treleasedate TEXT,\n\tgenres TEXT,\n\tdevelopers TEXT,\n\tpublishers TEXT,\n\tdescription TEXT,\n\tprice TEXT,\n\tthumbnail TEXT,\n\ttags TEXT,\n\treviewscore BIGINT,\n\tpositivereview BIGINT,\n\tnegativereview BIGINT,\n\tosrequirement TEXT,\n\tmemoryrequirement TEXT,\n\tcpurequirement TEXT,\n\trank BIGINT\n)\n",
  "message": null
}
```

### Hata Yanıtı (200 OK ama Hata Durumu)

```json
{
  "status": "error",
  "message": "Failed to get schema: Connection timed out."
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `config_id` veritabanında bulunamazsa döner.

---

## 7. Kolon İstatistiklerini Getirme

Kayıtlı bir MCP sunucusuna bağlanır ve `get_column_stats` aracını çalıştırarak bağlı veritabanındaki **tüm tabloların kolon bazında istatistiklerini** döner.

- Sayısal kolonlar için: min, max, ortalama (avg) ve boş değer (null) sayısı
- Metin kolonları için: en sık görülen 10 farklı değer ve kaç kez geçtiği

Frontend'de kullanıcıya veri hakkında istatistiksel bir özet sunmak için kullanılır.

* **URL:** `/mcp/column-stats`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)

```json
{
  "config_id": 1   // (Zorunlu) Kullanılacak MCP bağlantısının ID değeri
}
```

### Başarılı Yanıt (200 OK)

`data` alanı, MCP sunucusunun döndürdürdüğü ham istatistik metnini içerir.
- **Sayısal kolonlar** için: `min`, `max`, `avg`, `distinct` değer sayısı, `nulls` oranı
- **Metin kolonlar** için: `distinct` değer sayısı, `nulls` oranı ve en sık geçen 10 değer ile frekansları

```json
{
  "status": "success",
  "data": "Column Statistics:\n\n=== Database: 'default' ===\n\nTable: adult_income\n  Total rows: 32,561\n  \"age\" [numeric]: min=17, max=90, avg=38.5816, distinct=73, nulls=0 (0.0%)\n  \"workclass\" [text]: distinct=8, nulls=1836 (5.6%), top values=[\"Private\"(22696), \"Self-emp-not-inc\"(2541), \"Local-gov\"(2093), ...]\n  \"education\" [text]: distinct=16, nulls=0 (0.0%), top values=[\"HS-grad\"(10501), \"Some-college\"(7291), \"Bachelors\"(5355), ...]\n  \"income\" [text]: distinct=2, nulls=0 (0.0%), top values=[\"<=50K\"(24720), \">50K\"(7841)]\n\nTable: steam_game_clean\n  Total rows: 29,931\n  \"appid\" [numeric]: min=10, max=4460980, avg=1722531.1656, distinct=29789, nulls=0 (0.0%)\n  \"type\" [text]: distinct=6, nulls=0 (0.0%), top values=[\"game\"(22713), \"dlc\"(6586), \"music\"(600), ...]\n  \"price\" [text]: distinct=452, nulls=0 (0.0%), top values=[\"9.99$\"(3451), \"19.99$\"(2765), \"4.99$\"(2726), ...]\n  \"reviewscore\" [numeric]: min=0, max=9, avg=6.2371, distinct=10, nulls=0 (0.0%)\n",
  "message": null
}
```

### Hata Yanıtı (200 OK ama Hata Durumu)

```json
{
  "status": "error",
  "message": "Failed to get column stats: MCP tool call timed out."
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `config_id` veritabanında bulunamazsa döner.

---

## 8. Veritabanı Önbelleğini (Cache) Yenileme

Kayıtlı bir MCP sunucusuna bağlanarak şema ve kolon istatistiklerini zorla (force) yeniden çeker. Veritabanındaki kayıtlı önbelleği (cache) bu yeni verilerle ezerek günceller. Veritabanında tablo, veri veya kolon değişikliği yapıldığında arayüzün güncel veriyi görebilmesi için (örneğin bir "Verileri Güncelle" butonuna basıldığında) çağrılmalıdır.

* **URL:** `/mcp/refresh-stats`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)

```json
{
  "config_id": 1   // (Zorunlu) Kullanılacak MCP bağlantısının ID değeri
}
```

### Başarılı Yanıt (200 OK)

Verilerin başarılı bir şekilde çekilip veritabanına kaydedildiğini belirtir.

```json
{
  "status": "success",
  "message": "Cache refreshed successfully."
}
```

### Hata Yanıtı (200 OK ama Hata Durumu)

```json
{
  "status": "error",
  "message": "MCP tool call timed out during refresh."
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `config_id` veritabanında bulunamazsa döner.

---

## 9. Tablodan Örnek Veri Çekme

Kayıtlı bir MCP sunucusuna bağlanarak `run_sql_query` aracı yardımıyla belirtilen tablodan örnek veri (sample data) çeker. Veritabanının diline uygun olan (LIMIT veya TOP) SQL komutları otomatik denenir. Frontend'de tablo yapısını ve içerik formatını önizlemek için kullanılır.

* **URL:** `/mcp/sample-data`
* **Method:** `POST`
* **Content-Type:** `application/json`

### İstek Gövdesi (Request Body)

```json
{
  "config_id": 1,         // (Zorunlu) Kullanılacak MCP bağlantısının ID değeri
  "table_name": "users",  // (Zorunlu) Örnek veri çekilecek tablonun adı
  "limit": 10             // (Opsiyonel) Çekilecek maksimum satır sayısı (Varsayılan: 10)
}
```

### Başarılı Yanıt (200 OK)

Veriler ayrııştırılarak JSON formatında (`columns` ve `rows`) döner.

```json
{
  "status": "success",
  "data": {
    "columns": ["id", "name", "email", "created_at"],
    "rows": [
      {"id": 1, "name": "Ali", "email": "ali@example.com", "created_at": "2024-01-01"},
      {"id": 2, "name": "Veli", "email": "veli@example.com", "created_at": "2024-01-02"}
    ]
  }
}
```

### Hata Yanıtı (200 OK ama Hata Durumu)

```json
{
  "status": "error",
  "message": "SQL Error: no such table: users"
}
```

### Hata Yanıtları
* `404 Not Found`: Verilen `config_id` veritabanında bulunamazsa döner.
