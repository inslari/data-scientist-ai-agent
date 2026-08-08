"""ServingCache — per-key load-lock (eşzamanlılık), hot-path reuse, LRU eviction.

En kritik: iki eşzamanlı istek AYNI soğuk artifact'i isterse TEK yükleme olmalı (147MB'ı iki kez
yüklemek RAM'i patlatır). loader injection ile sayaç + gecikme kullanılıp bu çitlenir."""
import threading
import time

from api.core.serving_cache import ServingCache


def test_per_key_single_load_under_concurrency():
    calls = {"n": 0}
    clock = threading.Lock()

    def slow_loader(aid, base_dir):
        with clock:
            calls["n"] += 1
        time.sleep(0.3)                       # 147MB yükleme simülasyonu
        return (f"art_{aid}", f"model_{aid}")

    c = ServingCache(max_artifacts=2, loader=slow_loader)
    results = []

    def worker():
        results.append(c.get("A"))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1                     # 6 eşzamanlı istek → TEK yükleme
    assert all(r == ("art_A", "model_A") for r in results)


def test_different_keys_load_in_parallel_not_serialized():
    """Farklı anahtarlar birbirini BLOKLAMAZ (per-key kilit, global değil)."""
    order = []
    clock = threading.Lock()

    def loader(aid, base_dir):
        time.sleep(0.2)
        with clock:
            order.append(aid)
        return (aid, aid)

    c = ServingCache(max_artifacts=4, loader=loader)
    t0 = time.time()
    ts = [threading.Thread(target=c.get, args=(k,)) for k in ("A", "B", "C")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # Seri olsaydı ~0.6s; paralel ~0.2s. Gevşek tavan: <0.5s.
    assert (time.time() - t0) < 0.5
    assert set(order) == {"A", "B", "C"}


def test_hot_path_no_reload():
    calls = {"n": 0}

    def loader(aid, base_dir):
        calls["n"] += 1
        return (aid, aid)

    c = ServingCache(2, loader)
    c.get("A"); c.get("A"); c.get("A")
    assert calls["n"] == 1                      # sıcak → tek yükleme


def test_lru_eviction():
    loaded = []

    def loader(aid, base_dir):
        loaded.append(aid)
        return (aid, aid)

    c = ServingCache(max_artifacts=2, loader=loader)
    c.get("A"); c.get("B")
    assert set(c.keys()) == {"A", "B"}
    c.get("C")                                  # kapasite 2 → LRU (A) düşer
    assert set(c.keys()) == {"B", "C"}
    c.get("A")                                  # A yeniden yüklenir
    assert loaded.count("A") == 2


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
