# =============================================================
# utils/logger.py — Olay kaydedici + terminal log paneli
# =============================================================
# Olayları hem bellek içinde tutar (UI için)
# hem de dosyaya yazar (kalıcı kayıt için).
# =============================================================

import os
import time
import collections
from datetime import datetime


# ── Olay türleri ve renkleri (BGR) ───────────────────────────
EVENT_COLORS = {
    "info"     : (180, 180, 180),   # gri
    "tracking" : (0,   200, 110),   # yeşil
    "drifting" : (0,   165, 255),   # turuncu
    "lost"     : (60,   60, 230),   # kırmızı
    "redet"    : (200, 130,   0),   # mavi-yeşil
    "warn"     : (0,   165, 255),   # turuncu
    "system"   : (160, 100, 200),   # mor
}

EVENT_PREFIX = {
    "info"     : "[BİLGİ]  ",
    "tracking" : "[TAKİP]  ",
    "drifting" : "[SÜRÜK]  ",
    "lost"     : "[KAYIP]  ",
    "redet"    : "[REDET]  ",
    "warn"     : "[UYARI]  ",
    "system"   : "[SİSTEM] ",
}

REDET_METHOD = {
    1: "Histogram+MeanShift",
    2: "NCC Şablon Eşleme",
    3: "ORB-BRIEF+RANSAC",
}


class EventLogger:
    """
    Sistem olaylarını kaydeder.
    - Bellek içinde son N olayı tutar (UI paneli için)
    - Dosyaya tam log yazar
    """

    def __init__(self, log_dir: str = "logs",
                 max_display: int = 12):
        self.max_display = max_display
        self.events: collections.deque = collections.deque(
            maxlen=max_display
        )
        self._session_start = datetime.now()

        # Log klasörü ve dosyası
        os.makedirs(log_dir, exist_ok=True)
        ts = self._session_start.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"tracker_{ts}.log")

        self._file = open(self.log_path, "w", encoding="utf-8")
        self._write_file(
            "system",
            f"Oturum başladı — {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ── Genel log metodu ─────────────────────────────────────

    def log(self, event_type: str, message: str):
        ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        text = f"{ts}  {EVENT_PREFIX.get(event_type, '')} {message}"
        entry = {
            "time"  : ts,
            "type"  : event_type,
            "msg"   : message,
            "text"  : text,
            "color" : EVENT_COLORS.get(event_type, (180, 180, 180)),
        }
        self.events.append(entry)
        self._write_file(event_type, message)

    # ── Yardımcı olay metodları ──────────────────────────────

    def point_added(self, pid: int, x: int, y: int):
        self.log("system", f"Nokta #{pid} eklendi  ({x}, {y})")

    def point_lost(self, pid: int):
        self.log("lost", f"Nokta #{pid} kaybedildi")

    def point_drifting(self, pid: int, d_k: float, e_a: float):
        self.log("drifting",
                 f"Nokta #{pid} sürükleniyor  "
                 f"dK={d_k:.1f}  eA={e_a:.1f}px")

    def point_tracking(self, pid: int):
        self.log("tracking", f"Nokta #{pid} takip ediliyor")

    def redet_searching(self, pid: int):
        self.log("redet", f"Nokta #{pid} aranıyor...")

    def redet_found(self, pid: int, stage: int,
                    x: int, y: int):
        method = REDET_METHOD.get(stage, f"Kademe {stage}")
        self.log("redet",
                 f"Nokta #{pid} bulundu  [{method}]  ({x}, {y})")

    def redet_failed(self, pid: int):
        self.log("warn",
                 f"Nokta #{pid} yeniden tespit başarısız")

    def layer_toggled(self, name: str, active: bool):
        durum = "AKTİF" if active else "DEVRE DIŞI"
        self.log("system", f"Katman '{name}' → {durum}")

    def points_cleared(self):
        self.log("system", "Tüm noktalar silindi")

    def session_end(self):
        self.log("system",
                 f"Oturum sona erdi  —  log: {self.log_path}")
        self._file.flush()
        self._file.close()

    # ── Dahili ──────────────────────────────────────────────

    def _write_file(self, event_type: str, message: str):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{ts}  {EVENT_PREFIX.get(event_type, '[?]     ')} {message}\n"
        self._file.write(line)
        self._file.flush()

    def get_display_events(self) -> list:
        """UI paneli için son olayları döndürür (yeniden eskiye)."""
        return list(reversed(self.events))