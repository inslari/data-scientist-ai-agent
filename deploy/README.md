# Dağıtım (Deployment) Rehberi

Bu klasör (`deploy/`), projenizi özellikle Dokploy gibi ortamlara veya herhangi bir Docker sunucusuna kurarken ihtiyacınız olabilecek farklı senaryoları destekleyen dosyaları içerir. 

Ana dizindeki projenizde hem bir **API sunucusu** (FastAPI) hem de **Arkaplan Görev Yöneticisi** (Worker) bulunmaktadır.

## Senaryo 1: İkisini Birlikte Çalıştırmak (Docker Compose - Önerilen)

Eğer Dokploy üzerinde veya sunucunuzda API ve Worker'ı tek bir yerden birlikte yönetmek istiyorsanız bu yöntemi seçin.

- **Dokploy Kurulum Tipi:** `Docker Compose`
- **Compose File Path (Dosya Yolu):** `./deploy/docker-compose.prod.yml`
- **Açıklama:** Bu dosya bir üst dizini (`..`) kaynak (context) olarak alır ve hem API'yi hem de Worker uygulamanızı ayağa kaldırıp `.env` değişkenlerine ve `/app/data` kalıcı depolama alanına otomatik bağlar.

## Senaryo 2: Ayrı Ayrı Dağıtım (Ayrı Dokploy Uygulamaları)

Eğer Dokploy üzerinde API için bir uygulama oluşturduysanız ve yanına **SADECE** Worker'ı ekstra bir uygulama (Application) olarak kurmak istiyorsanız bu yöntemi seçmelisiniz.

**Worker Kurulum Adımları:**
1. **Dokploy Kurulum Tipi:** `Dockerfile`
2. **Context Path (Kaynak Klasörü):** `/` *(Çok önemli! Kurulumun ana proje dizininde çalışması gerekir)*
3. **Dockerfile Path:** `./deploy/Dockerfile.worker`
4. **Environment Variables (.env):** API ile birebir aynı değerleri girmeyi unutmayın (Veritabanı URL'si vb. dahil).

Bu sayede API projeniz standart ana dizindeki `Dockerfile`'ı kullanırken, yeni açtığınız uygulamanız `./deploy/Dockerfile.worker` sayesinde arka planda işleri yürüten Worker olarak çalışacaktır.
