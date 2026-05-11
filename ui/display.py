# =============================================================
# ui/display.py — Yan panel tabanlı arayüz
# =============================================================
# Düzen:
#   [ Kamera görüntüsü (temiz) ] [ Bilgi paneli (sabit 300px) ]
#
# Kamera üzerinde yalnızca: nokta daireleri + iz çizgileri
# Panel: FPS, C̄, katman durumları, nokta detayları, ReID bilgisi
# =============================================================

import cv2
import numpy as np
import config

# ── Sabitler ─────────────────────────────────────────────────
PANEL_W   = 300
BG        = (18, 18, 18)
FG        = (220, 220, 220)
DIM       = (90, 90, 90)
ACCENT    = (0, 200, 110)
WARN      = (0, 165, 255)
ERR       = (60, 60, 230)
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

STATE_COLOR = {
    "tracking" : ACCENT,
    "drifting" : WARN,
    "lost"     : ERR,
    "pending"  : (0, 215, 255),   # altın sarısı
}
STATE_LABEL = {
    "tracking" : "TAKIP",
    "drifting" : "SURUKLENME",
    "lost"     : "KAYIP",
    "pending"  : "ONAY?",
}


# ── Yardımcı çizim fonksiyonları ─────────────────────────────

def _text(img, txt, x, y, color=FG, scale=0.42, thickness=1):
    cv2.putText(img, str(txt), (x, y), FONT, scale,
                color, thickness, cv2.LINE_AA)


def _bold(img, txt, x, y, color=FG, scale=0.44):
    cv2.putText(img, str(txt), (x, y), FONT_BOLD, scale,
                color, 1, cv2.LINE_AA)


def _hline(img, y, x0=8, x1=None, color=DIM):
    if x1 is None:
        x1 = img.shape[1] - 8
    cv2.line(img, (x0, y), (x1, y), color, 1)


def _bar(img, x, y, w, h, value, color, bg=(40, 40, 40)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    filled = int(w * max(0.0, min(1.0, float(value))))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)


def _badge(img, txt, x, y, color):
    (tw, th), _ = cv2.getTextSize(str(txt), FONT, 0.36, 1)
    pad = 3
    cv2.rectangle(img, (x - pad, y - th - pad),
                  (x + tw + pad, y + pad), color, -1)
    cv2.putText(img, str(txt), (x, y), FONT, 0.36,
                (255, 255, 255), 1, cv2.LINE_AA)


# ── Kamera görüntüsü (minimal çizim) ─────────────────────────

def draw_points(frame: np.ndarray, states: list) -> np.ndarray:
    out = frame.copy()
    GOLD = (0, 215, 255)

    for info in states:
        pos   = info["pos"]
        state = info["state"]
        trail = info["trail"]
        pid   = info["id"]
        color = STATE_COLOR.get(state, DIM)
        x, y  = int(pos[0]), int(pos[1])

        for i in range(1, len(trail)):
            alpha = i / len(trail)
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(out, trail[i - 1], trail[i], c, 1, cv2.LINE_AA)

        cv2.circle(out, (x, y), 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.putText(out, f"#{pid}", (x + 11, y + 5),
                    FONT, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Pending: önerilen konumu ayrıca göster
        ppos  = info.get("pending_pos")
        ptime = info.get("pending_time")
        if ppos is not None and ptime is not None:
            import time as _time
            elapsed  = _time.time() - ptime
            remaining = max(0.0, 5.0 - elapsed)
            px, py   = int(ppos[0]), int(ppos[1])

            # Yanıp sönen halka (her 0.5 sn değişir)
            blink = int(elapsed * 2) % 2 == 0
            if blink:
                cv2.circle(out, (px, py), 14, GOLD, 2, cv2.LINE_AA)
            cv2.circle(out, (px, py), 10, GOLD, -1, cv2.LINE_AA)
            cv2.putText(out, "?", (px - 5, py + 5),
                        FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            # Geri sayım etiketi
            cv2.putText(out,
                        f"Y=Onayla  N=Reddet  {remaining:.1f}s",
                        (px + 16, py - 4),
                        FONT, 0.40, GOLD, 1, cv2.LINE_AA)

    return out


# ── Panel oluşturucu ─────────────────────────────────────────

def build_panel(height: int, fps: float, avg_ms: float,
                layers: dict, states: list,
                frame_conf: float,
                adaptive_meta: dict,
                perf_summary: dict = None) -> np.ndarray:

    panel = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    y = 0

    # Başlık
    cv2.rectangle(panel, (0, 0), (PANEL_W, 32), (28, 28, 28), -1)
    _bold(panel, "LK TRACKER", 10, 22, ACCENT, 0.52)
    _text(panel, "v1.0", PANEL_W - 42, 22, DIM, 0.38)
    y = 36

    # ── Pending uyarı bandı ───────────────────────────────────
    import time as _time
    pending_list = [
        i for i in states if i.get("pending_pos") is not None
    ]
    if pending_list:
        cv2.rectangle(panel, (0, y), (PANEL_W, y + 36),
                      (0, 60, 80), -1)
        _bold(panel, "ONAY BEKLENİYOR", 8, y + 13, (0, 215, 255), 0.42)
        # En yakın süre dolan
        min_remaining = min(
            max(0.0, 5.0 - (_time.time() - i["pending_time"]))
            for i in pending_list if i.get("pending_time")
        )
        _bold(panel, f"{min_remaining:.1f}s", PANEL_W - 40,
              y + 13, (0, 215, 255), 0.48)
        _text(panel, "Y: Onayla    N: Reddet",
              8, y + 28, (180, 180, 180), 0.36)
        y += 40
    _hline(panel, y);  y += 12
    _bold(panel, "PERFORMANS", 8, y, FG, 0.40);  y += 16

    fps_color = ACCENT if fps >= 28 else (WARN if fps >= 15 else ERR)
    _text(panel, "FPS",   8,  y, DIM, 0.38)
    _bold(panel, f"{fps:.1f}", 44, y, fps_color, 0.52)
    _text(panel, "nokta", 130, y, DIM, 0.38)
    _bold(panel, str(len(states)), 178, y, FG, 0.50)
    y += 16

    if perf_summary:
        rows = [
            ("Kamera okuma",   "capture"),
            ("Algoritma",      "process"),
            ("Render",         "render"),
            ("Toplam gecikme", "total"),
        ]
        for label, key in rows:
            val       = perf_summary.get(key, 0.0)
            is_total  = key == "total"
            val_color = (ERR   if val > 33 else
                         WARN  if val > 20 else
                         ACCENT)
            if is_total:
                _hline(panel, y + 1, color=(50, 50, 50))
                y += 4
                _bold(panel, label, 8, y + 11, FG, 0.38)
                _bold(panel, f"{val:.1f}ms", PANEL_W - 52,
                      y + 11, val_color, 0.40)
            else:
                _text(panel, label, 14, y + 11, DIM, 0.35)
                _text(panel, f"{val:.1f}ms", PANEL_W - 50,
                      y + 11, val_color, 0.36)
            y += 14
    else:
        _text(panel, f"proc: {avg_ms:.1f}ms", 8, y, DIM, 0.38)
        y += 16

    # ── Kare Güven Skoru ─────────────────────────────────────
    _hline(panel, y);  y += 12
    _bold(panel, "KARE GUVEN  C_med", 8, y, FG, 0.40);  y += 18

    conf_color = (ACCENT if frame_conf >= config.CONF_DRIFT_THR
                  else WARN if frame_conf >= config.CONF_LOST_THR
                  else ERR)
    _bold(panel, f"{frame_conf:.3f}", 8, y, conf_color, 0.68)
    _bar(panel, 75, y - 12, PANEL_W - 85, 16, frame_conf, conf_color)

    # Eşik işaretçileri
    bw = PANEL_W - 85
    bx = 75
    t1 = bx + int(bw * config.CONF_LOST_THR)
    t2 = bx + int(bw * config.CONF_DRIFT_THR)
    cv2.line(panel, (t1, y - 14), (t1, y + 4), ERR,  1)
    cv2.line(panel, (t2, y - 14), (t2, y + 4), WARN, 1)
    _text(panel, f"{config.CONF_LOST_THR}", t1 - 6, y + 14, ERR, 0.30)
    _text(panel, f"{config.CONF_DRIFT_THR}", t2 - 6, y + 14, WARN, 0.30)
    y += 24

    # ── Katmanlar ────────────────────────────────────────────
    _hline(panel, y);  y += 12
    _bold(panel, "KATMANLAR", 8, y, FG, 0.40);  y += 16

    layer_cfg = [
        ("fb_error",    "F", "Ileri-Geri Hata"),
        ("confidence",  "G", "Guven Skoru"),
        ("adaptive",    "A", "Adaptif Parametre"),
        ("drift",       "D", "Suruklenme Tespiti"),
        ("redetection", "R", "Yeniden Tespit"),
    ]

    for key, kc, label in layer_cfg:
        aktif  = layers.get(key, False)
        dot_c  = ACCENT if aktif else (50, 50, 50)
        txt_c  = FG     if aktif else DIM
        status = "ON" if aktif else "OFF"

        cv2.circle(panel, (14, y - 2), 5, dot_c, -1, cv2.LINE_AA)
        _text(panel, f"[{kc}]", 22, y, txt_c, 0.38)
        _text(panel, label,     50, y, txt_c, 0.38)
        _text(panel, status, PANEL_W - 36, y,
              ACCENT if aktif else DIM, 0.36)
        y += 16

        if key == "adaptive" and aktif and adaptive_meta:
            m = adaptive_meta
            s = (f"  win={m.get('winSize','?')}  "
                 f"lv={m.get('maxLevel','?')}  "
                 f"it={m.get('iter','?')}")
            _text(panel, s, 8, y, (0, 160, 80), 0.34)
            y += 14

    # ── ReID Durumu ───────────────────────────────────────────
    _hline(panel, y);  y += 12
    _bold(panel, "YENIDEN TESPIT", 8, y, FG, 0.40);  y += 16

    redet_rows = [
        (1, "K1  Histogram+MeanShift", "2-4ms"),
        (2, "K2  NCC Sablon",          "3-6ms"),
        (3, "K3  ORB-BRIEF+RANSAC",    "4-5ms"),
    ]

    # Mevcut aktif kademeleri bul
    active_stages = {
        info.get("redet_stage", 0)
        for info in states
        if info.get("redet_stage", 0) > 0
    }

    for sno, slabel, sms in redet_rows:
        # Bu kademe şu an aktif mi arıyor?
        searching_this = any(
            i.get("state") == "lost"
            and i.get("active_kademe", 1) == sno
            for i in states
        )
        # Bu kademe en son başarılı mıydı?
        is_success = sno in active_stages
        # Bu kademe tamamlanıp geçildi mi?
        is_done = any(
            i.get("state") == "lost"
            and i.get("active_kademe", 1) > sno
            for i in states
        )

        if is_success:
            cv2.rectangle(panel, (4, y - 12),
                          (PANEL_W - 4, y + 5), (0, 55, 28), -1)
            _bold(panel, slabel, 10, y, ACCENT, 0.38)
            _text(panel, sms, PANEL_W - 42, y, ACCENT, 0.34)
            cv2.circle(panel, (PANEL_W - 10, y - 4),
                       4, ACCENT, -1, cv2.LINE_AA)
        elif searching_this and layers.get("redetection"):
            cv2.rectangle(panel, (4, y - 12),
                          (PANEL_W - 4, y + 5), (40, 30, 0), -1)
            kf = max(
                (i.get("kademe_frames", 0) for i in states
                 if i.get("state") == "lost"
                 and i.get("active_kademe", 1) == sno),
                default=0
            )
            limit = (config.REDET_K1_MAX_FRAMES if sno == 1
                     else config.REDET_K2_MAX_FRAMES if sno == 2
                     else config.REDET_K3_MAX_FRAMES)
            _bold(panel, slabel, 10, y, WARN, 0.38)
            _text(panel, f"{kf}/{limit}", PANEL_W - 52, y, WARN, 0.34)
            cv2.circle(panel, (PANEL_W - 10, y - 4),
                       4, WARN, -1, cv2.LINE_AA)
        elif is_done:
            _text(panel, slabel, 10, y, (60, 60, 60), 0.38)
            _text(panel, "gecildi", PANEL_W - 52, y, (60, 60, 60), 0.34)
        else:
            _text(panel, slabel, 10, y, DIM, 0.38)
            _text(panel, sms, PANEL_W - 42, y, DIM, 0.34)
        y += 18

    if not layers.get("redetection"):
        _text(panel, "Katman kapali", 10, y, DIM, 0.36)
        y += 14

    # ── Nokta Detayları ───────────────────────────────────────
    _hline(panel, y);  y += 12
    _bold(panel, "NOKTA DETAYLARI", 8, y, FG, 0.40);  y += 16

    for info in states:
        if y + 80 > height - 80:
            _text(panel, f"... +{len(states)} nokta", 8, y, DIM, 0.36)
            break

        pid   = info["id"]
        state = info["state"]
        conf  = info["conf"]
        fb    = info.get("fb_error", 0.0)
        dk    = info.get("d_k", 0.0)
        ea    = info.get("e_anchor", 0.0)
        redet = info.get("redet_stage", 0)
        comps = info.get("conf_comps", {})
        color = STATE_COLOR.get(state, DIM)
        sl    = STATE_LABEL.get(state, state)

        # Başlık
        cv2.rectangle(panel, (4, y - 12),
                      (PANEL_W - 4, y + 4), (30, 30, 30), -1)
        _bold(panel, f"#{pid}", 8, y, color, 0.44)
        _badge(panel, sl, 38, y, color)
        if redet > 0:
            _badge(panel, f"K{redet}", PANEL_W - 32, y, (0, 100, 180))
        y += 18

        # Güven çubuğu
        conf_c = (ACCENT if conf >= config.CONF_DRIFT_THR
                  else WARN if conf >= config.CONF_LOST_THR else ERR)
        _text(panel, f"C={conf:.2f}", 8, y, FG, 0.36)
        _bar(panel, 58, y - 10, PANEL_W - 68, 12, conf, conf_c)
        y += 14

        # Bileşenler
        if comps:
            _text(panel,
                  f"ncc={comps.get('ncc',0):.2f}  "
                  f"fb={comps.get('fb',0):.2f}  "
                  f"eig={comps.get('eig',0):.2f}  "
                  f"grd={comps.get('grad',0):.2f}",
                  8, y, DIM, 0.32)
            y += 13

        _text(panel,
              f"FB={fb:.1f}px  dK={dk:.1f}  eA={ea:.1f}",
              8, y, DIM, 0.34)
        y += 14

        # Kayıp durumunda kademe ve ilerleme bilgisi
        lost_frames   = info.get("lost_frames", 0)
        active_kademe = info.get("active_kademe", 1)
        kademe_frames = info.get("kademe_frames", 0)

        if state == "lost" and lost_frames > 0:
            limit = (config.REDET_K1_MAX_FRAMES if active_kademe == 1
                     else config.REDET_K2_MAX_FRAMES if active_kademe == 2
                     else config.REDET_K3_MAX_FRAMES)
            warn_color = ERR if active_kademe == 3 else WARN

            steps     = lost_frames // config.SEARCH_GROWTH_EVERY
            sigma_eff = min(
                config.SEARCH_SIGMA + steps * config.SEARCH_GROWTH_RATE,
                config.SEARCH_SIGMA_MAX
            )
            _text(panel,
                  f"K{active_kademe}: {kademe_frames}/{limit}  "
                  f"Toplam: {lost_frames}  σ={sigma_eff:.1f}",
                  8, y, warn_color, 0.33)
            y += 12
            # Kademe ilerleme çubuğu
            if isinstance(limit, int) and limit > 0:
                ratio = kademe_frames / limit
                _bar(panel, 8, y, PANEL_W - 18, 5,
                     ratio, warn_color, bg=(40, 40, 40))
                y += 8
        _hline(panel, y, color=(35, 35, 35));  y += 6

    # ── Klavye kısayolları ────────────────────────────────────
    ky = height - 72
    _hline(panel, ky);  ky += 10
    _bold(panel, "KISAYOLLAR", 8, ky, DIM, 0.36);  ky += 14
    for s in ["Sol tik: nokta ekle",
               "C: temizle  Q/ESC: cikis",
               "Y: onayla  N: reddet",
               "F G A D R: katman"]:
        _text(panel, s, 8, ky, DIM, 0.33)
        ky += 13

    return panel


# ── Log Paneli ───────────────────────────────────────────────

LOG_PANEL_H = 140   # log panelinin piksel yüksekliği

def build_log_panel(width: int, events: list) -> np.ndarray:
    """
    Alt kısma yapışan yatay log şeridi oluşturur.
    events: logger.get_display_events() çıktısı (yeniden eskiye)
    """
    panel = np.full((LOG_PANEL_H, width, 3), (12, 12, 12), dtype=np.uint8)

    # Üst çizgi
    cv2.line(panel, (0, 0), (width, 0), (50, 50, 50), 1)

    # Başlık
    _bold(panel, "OLAY KAYITLARI", 8, 14, DIM, 0.38)
    _hline(panel, 18, color=(40, 40, 40))

    if not events:
        _text(panel, "Henüz olay yok...", 8, 38, DIM, 0.36)
        return panel

    y = 32
    for entry in events:
        if y > LOG_PANEL_H - 8:
            break
        color = entry.get("color", FG)
        ts    = entry.get("time", "")
        msg   = entry.get("msg", "")
        etype = entry.get("type", "info")
        prefix = {
            "lost"     : "✗",
            "redet"    : "↺",
            "drifting" : "~",
            "tracking" : "✓",
            "system"   : "·",
            "warn"     : "!",
            "info"     : " ",
        }.get(etype, " ")

        _text(panel, ts,              8,   y, DIM,   0.33)
        _text(panel, prefix,          82,  y, color, 0.38)
        _text(panel, msg,             96,  y, color, 0.36)
        y += 16

    return panel


# ── Ana birleştirici (güncellendi) ───────────────────────────

def compose(frame: np.ndarray, states: list,
            fps: float, avg_ms: float,
            layers: dict, frame_conf: float,
            adaptive_meta: dict,
            log_events: list = None,
            perf_summary: dict = None) -> np.ndarray:
    """
    Kamera görüntüsü + bilgi paneli + log şeridi birleştirir.

    Düzen:
        ┌─────────────────┬──────────┐
        │  Kamera         │  Panel   │
        ├─────────────────┴──────────┤
        │  Log şeridi (tam genişlik) │
        └────────────────────────────┘
    """
    video = draw_points(frame, states)
    panel = build_panel(
        height        = video.shape[0],
        fps           = fps,
        avg_ms        = avg_ms,
        layers        = layers,
        states        = states,
        frame_conf    = frame_conf,
        adaptive_meta = adaptive_meta,
        perf_summary  = perf_summary,
    )

    top = np.hstack([video, panel])
    log_panel = build_log_panel(
        width  = top.shape[1],
        events = log_events or [],
    )
    return np.vstack([top, log_panel])