# =============================================================
# utils/performance.py — Ayrıntılı gecikme ve FPS ölçümü
# =============================================================
#
# Ölçülen süreler:
#   capture_ms  : cap.read() — kamera/video okuma süresi
#   process_ms  : tracker.update() — tüm algoritma süresi
#   render_ms   : çizim + imshow hazırlık süresi
#   total_ms    : capture + process + render toplamı
#                 (gerçek uçtan uca gecikme)
#
# FPS hesabı total_ms üzerinden yapılır.
# =============================================================

import time
import collections


class PerformanceMonitor:

    def __init__(self, window: int = 30):
        self.window = window

        self._times = {
            "capture" : collections.deque(maxlen=window),
            "process" : collections.deque(maxlen=window),
            "render"  : collections.deque(maxlen=window),
            "total"   : collections.deque(maxlen=window),
        }
        self._t     = {}   # aktif zamanlayıcılar

    # ── Zamanlayıcı API ──────────────────────────────────────

    def start(self, key: str):
        self._t[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        """ms cinsinden süreyi döndürür ve kaydeder."""
        if key not in self._t:
            return 0.0
        elapsed = (time.perf_counter() - self._t.pop(key)) * 1000
        if key in self._times:
            self._times[key].append(elapsed)
        return elapsed

    def record_total(self):
        """capture + process + render toplamını hesaplar."""
        total = (self._last("capture") +
                 self._last("process") +
                 self._last("render"))
        self._times["total"].append(total)

    def _last(self, key: str) -> float:
        q = self._times.get(key)
        return q[-1] if q else 0.0

    # ── Ortalama değerler ────────────────────────────────────

    def avg(self, key: str) -> float:
        q = self._times.get(key)
        if not q:
            return 0.0
        return sum(q) / len(q)

    def last(self, key: str) -> float:
        return self._last(key)

    # ── FPS (total_ms üzerinden) ─────────────────────────────

    @property
    def fps(self) -> float:
        avg_total = self.avg("total")
        return 1000.0 / avg_total if avg_total > 0 else 0.0

    # ── Geriye dönük uyumluluk (display.py avg_ms çağırıyor) ─

    @property
    def avg_ms(self) -> float:
        return self.avg("process")

    @property
    def last_ms(self) -> float:
        return self._last("process")

    # ── Terminal çıktısı ─────────────────────────────────────

    def terminal_line(self, n_points: int, layers: dict,
                      frame_conf: float = 1.0) -> str:
        aktif    = [k for k, v in layers.items() if v]
        aktif_str = " | ".join(aktif) if aktif else "yok"

        conf_sym = ("✓" if frame_conf >= 0.45
                    else "~" if frame_conf >= 0.25
                    else "✗")

        return (
            f"FPS:{self.fps:5.1f}  "
            f"cap:{self.avg('capture'):4.1f}ms  "
            f"proc:{self.avg('process'):5.1f}ms  "
            f"rend:{self.avg('render'):4.1f}ms  "
            f"toplam:{self.avg('total'):5.1f}ms  "
            f"nokta:{n_points}  "
            f"C̄:{frame_conf:.2f}{conf_sym}  "
            f"[{aktif_str}]"
        )

    def summary(self) -> dict:
        """Tüm ortalama değerleri sözlük olarak döndürür."""
        return {k: round(self.avg(k), 2) for k in self._times}