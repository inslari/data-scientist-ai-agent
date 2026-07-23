"""Katman 1 — inference artifact.

Eğitim sonunda öğrenilen fitted-state'i (transform parametreleri + model + eşik +
şema imzası) tek bir taşınabilir Artifact'e toplar; prodüksiyonda tek ham satırı,
eğitimle BİREBİR AYNI kod yolundan (``transform``) geçirerek modele hazır hale getirir.

Temel ilke (training/serving skew'i yapısal olarak imkânsız kılmak): ``fit`` yalnız
X_train görür ve parametre öğrenir; ``transform`` yalnız parametre alır ve uygular —
istatistik hesaplamaz. Eğitim ``fit`` + ``transform`` çağırır, inference yalnız
``transform``; ikisi de AYNI ``transform`` fonksiyonunu paylaşır (bkz. protocol.py).
"""
