"""
Sentetik Türkçe müşteri geri bildirimi üreteci (metin analizi özelliğini test için).
==================================================================================
Çıktı (data/sample_text_analysis/ altına, gitignored):
  - sample_feedback_tr.txt : "YYYY-MM-DD - <metin>" (satır başına bir geri bildirim)
                              → Metin Analizi ekranına DOĞRUDAN yapıştırılabilir.
  - sample_feedback_tr.csv : date,sector,text  (gelecekteki CSV ingestion için)

Özellikler:
  - 3 sektör (e-ticaret / banka / telekom), karışık sentiment + kategori + aspect
  - Tarih aralığı 2026-01..2026-06; aylara göre bilinçli trend (Şub-Mar kötü,
    May-Haz toparlanma) → Zaman Trendi grafiğini anlamlı kılar.
  - Ilımlı tekrar (dedup/cache yolunu da test eder).
  - Deterministik (RANDOM_SEED) → tekrar çalıştırınca aynı set.

Çalıştırma:
    uv run python scripts/generate_sample_feedback.py
    uv run python scripts/generate_sample_feedback.py --rows 300
"""
import argparse
import csv
import os
import random
from datetime import date, timedelta

SEED = 42

# Aylara göre "olumsuzluk olasılığı" — trend sinyali (Şub-Mar pik, May-Haz iyileşme)
MONTH_NEG_WEIGHT = {1: 0.50, 2: 0.78, 3: 0.82, 4: 0.55, 5: 0.35, 6: 0.28}

# Sektör → polariteye göre şablonlar (aspect'ler metne gömülü)
TEMPLATES = {
    "e-ticaret": {
        "neg": [
            "Kargom {n} gündür gelmedi, takip numarası da güncellenmiyor, çok mağdurum.",
            "Teslimat yine gecikti, kurye bir türlü ulaşmadı, sipariş no {no}.",
            "Yanlış ürün gönderilmiş, iade sürecini başlatmak istiyorum.",
            "Ürün kutusu ezik ve hasarlı geldi, bu kalite kabul edilemez.",
            "Ödeme ekranında uygulama sürekli hata veriyor, sipariş veremiyorum.",
            "İade ettiğim ürünün parası {n} gündür hesabıma geçmedi.",
            "Müşteri hizmetlerine bağlanamıyorum, canlı destek hep meşgul.",
            "Fiyatlar son dönemde çok arttı, aynı ürün geçen ay çok daha ucuzdu.",
            "Sipariş ettiğim beden yanlış geldi, değişim talep ediyorum.",
            "Kampanya kodu çalışmıyor, indirim sepete yansımadı.",
        ],
        "pos": [
            "Ürün kalitesi beklediğimden çok daha iyi, teşekkür ederim.",
            "Kargo aynı gün elime ulaştı, hız konusunda harikasınız.",
            "Müşteri temsilcisi çok ilgiliydi, sorunumu hemen çözdü.",
            "Paketleme özenliydi, ürün sapasağlam geldi, çok memnunum.",
            "İade sürecim sorunsuz ve hızlı tamamlandı, teşekkürler.",
        ],
        "neu": [
            "Siparişimin ne zaman kargoya verileceğini öğrenebilir miyim?",
            "Bu ürünün stoğa ne zaman gireceğini merak ediyorum.",
            "Taksit seçenekleri hangi bankalar için geçerli?",
            "Faturamı e-posta olarak tekrar gönderebilir misiniz?",
        ],
    },
    "banka": {
        "neg": [
            "Hesabımdan açıklamasız {n} TL kesinti yapılmış, derhal açıklama istiyorum.",
            "Mobil bankacılık uygulaması güncellemeden sonra hiç açılmıyor.",
            "Kredi başvurum sebepsiz reddedildi, gerekçe verilmiyor.",
            "Kart aidatı tekrar yansıtılmış, oysa muaf olmam gerekiyordu.",
            "EFT işlemim {n} saattir gerçekleşmedi, para havada kaldı.",
            "Çağrı merkezinde {n} dakika bekledim, kimse açmadı.",
            "Şifremi sıfırlayamıyorum, SMS doğrulama kodu gelmiyor.",
            "Faiz oranları habersiz değişti, sözleşmede bu yoktu.",
        ],
        "pos": [
            "Yeni mobil uygulama arayüzü çok pratik olmuş, elinize sağlık.",
            "Kredi başvurum dakikalar içinde onaylandı, çok memnun kaldım.",
            "Şubedeki görevli son derece yardımcı oldu, teşekkürler.",
            "Müşteri temsilciniz sorunumu büyük bir nezaketle çözdü.",
        ],
        "neu": [
            "Konut kredisi faiz oranlarını öğrenebilir miyim?",
            "Kartımın limitini nasıl artırabilirim?",
            "Hesap işletim ücreti ne zaman tahsil ediliyor?",
        ],
    },
    "telekom": {
        "neg": [
            "İnternetim {n} gündür çok yavaş, akşamları hiç bağlanamıyorum.",
            "Faturama tanımsız bir paket ücreti eklenmiş, iptal istiyorum.",
            "Hat sürekli kopuyor, teknik ekip hâlâ gelmedi.",
            "Taahhüt bitti ama hâlâ eski tarifeden ücretlendiriliyorum.",
            "Numara taşıma işlemim {n} gündür tamamlanmadı.",
            "Müşteri hizmetleri beni sürekli farklı birime aktarıyor, çözüm yok.",
            "Mobil veri paketim bitmeden tükendi, haksız tüketim var.",
        ],
        "pos": [
            "Fiber bağlantı hızı muhteşem, kesinti yaşamadım, teşekkürler.",
            "Tarife değişikliğim anında yapıldı, çok hızlısınız.",
            "Teknik servis aynı gün geldi ve sorunu çözdü, memnunum.",
        ],
        "neu": [
            "Yurt dışı paketleri hakkında bilgi alabilir miyim?",
            "Taahhüdüm ne zaman doluyor öğrenebilir miyim?",
            "Yeni tarifeye nasıl geçebilirim?",
        ],
    },
}


def _random_date(rng: random.Random) -> date:
    month = rng.choices(list(MONTH_NEG_WEIGHT.keys()), k=1)[0]
    # Haziran'da bugüne (22) kadar; diğer aylar tam.
    max_day = 22 if month == 6 else 28
    day = rng.randint(1, max_day)
    return date(2026, month, day)


def _pick_text(rng: random.Random, sector: str, polarity: str) -> str:
    tpl = rng.choice(TEMPLATES[sector][polarity])
    return tpl.format(n=rng.randint(2, 9), no=rng.randint(100000, 999999))


def generate(rows: int) -> list[tuple[date, str, str]]:
    rng = random.Random(SEED)
    sectors = list(TEMPLATES.keys())
    out: list[tuple[date, str, str]] = []
    for _ in range(rows):
        d = _random_date(rng)
        sector = rng.choice(sectors)
        neg_w = MONTH_NEG_WEIGHT[d.month]
        if rng.random() < neg_w:
            polarity = "neg"
        else:
            # kalan olasılık olumlu/nötr arasında
            polarity = "pos" if rng.random() < 0.7 else "neu"
        out.append((d, sector, _pick_text(rng, sector, polarity)))
    out.sort(key=lambda r: r[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=180)
    args = ap.parse_args()

    rows = generate(args.rows)

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "sample_text_analysis")
    os.makedirs(base, exist_ok=True)
    txt_path = os.path.join(base, "sample_feedback_tr.txt")
    csv_path = os.path.join(base, "sample_feedback_tr.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        for d, _sector, text in rows:
            f.write(f"{d.isoformat()} - {text}\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "sector", "text"])
        for d, sector, text in rows:
            w.writerow([d.isoformat(), sector, text])

    print(f"{len(rows)} satır üretildi.")
    print(f"  TXT (yapıştırmak için): {txt_path}")
    print(f"  CSV (ileride ingestion): {csv_path}")


if __name__ == "__main__":
    main()
