# =============================================================
# core/redetection.py — Yeniden Tespit Fonksiyonları
# =============================================================
# Bu modül üç tespit fonksiyonu ve referans verilerini tutan
# RedetectionManager sınıfını içerir.
#
# Kademe geçiş mantığı tracker.py → _run_redetection'dadır.
# Bu modül yalnızca tek bir kademeyi çalıştırır ve sonucu döner.
#
# Kademe 1: detect_by_histogram  — Renk histogramı + MeanShift
# Kademe 2: detect_by_template   — NCC şablon eşleme
# Kademe 3: detect_by_orb        — ORB-BRIEF + RANSAC
# =============================================================

import cv2
import numpy as np
import config


# ── Kademe 1: Renk Histogramı + MeanShift ───────────────────

def detect_by_histogram(frame: np.ndarray,
                        ref_hist: np.ndarray,
                        roi: tuple,
                        threshold: float = None) -> tuple | None:
    """
    H-S renk histogramı ile arka yansıtma + MeanShift.

    ref_hist L1 normalize edilmiş olmalıdır.
    Bhattacharyya benzerlik skoru eşiği aşarsa başarılı.

    Döndürür: (cx, cy, score) veya None
    """
    thr = threshold or config.HIST_THRESHOLD
    x1, y1, x2, y2 = roi

    search = frame[y1:y2, x1:x2]
    if search.size == 0:
        return None

    hsv_search = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)

    # Arka yansıtma
    back = cv2.calcBackProject(
        [hsv_search], [0, 1],
        ref_hist,
        [0, 180, 0, 256], 1
    )

    # Gürültü azaltma
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(back, -1, kernel, back)

    # MeanShift
    init_window = (0, 0, search.shape[1], search.shape[0])
    criteria    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
    _, window   = cv2.meanShift(back, init_window, criteria)

    wx, wy, ww, wh = window
    cx = x1 + wx + ww // 2
    cy = y1 + wy + wh // 2

    # Bulunan bölgeyi L1 normalize ile karşılaştır
    found_roi = frame[y1 + wy: y1 + wy + wh, x1 + wx: x1 + wx + ww]
    if found_roi.size == 0:
        return None

    hsv_found  = cv2.cvtColor(found_roi, cv2.COLOR_BGR2HSV)
    found_hist = cv2.calcHist([hsv_found], [0, 1], None,
                               [18, 32], [0, 180, 0, 256])
    cv2.normalize(found_hist, found_hist,
                  alpha=1, beta=0, norm_type=cv2.NORM_L1)

    dist  = cv2.compareHist(ref_hist, found_hist, cv2.HISTCMP_BHATTACHARYYA)
    score = float(max(0.0, 1.0 - dist))

    if score >= thr:
        return (cx, cy, score)
    return None


# ── Kademe 2: NCC Şablon Eşleme ─────────────────────────────

def detect_by_template(frame_gray: np.ndarray,
                       template: np.ndarray,
                       roi: tuple,
                       threshold: float = None) -> tuple | None:
    """
    CV_TM_CCOEFF_NORMED ile şablon eşleme.
    En yüksek korelasyon skoru eşiği aşarsa başarılı.

    Döndürür: (cx, cy, score) veya None
    """
    thr = threshold or config.MATCH_THRESHOLD
    x1, y1, x2, y2 = roi

    search = frame_gray[y1:y2, x1:x2]
    if (search.shape[0] < template.shape[0] or
            search.shape[1] < template.shape[1]):
        return None

    result               = cv2.matchTemplate(
        search, template, cv2.TM_CCOEFF_NORMED
    )
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= thr:
        th, tw = template.shape[:2]
        cx = x1 + max_loc[0] + tw // 2
        cy = y1 + max_loc[1] + th // 2
        return (cx, cy, float(max_val))
    return None


# ── Kademe 3: ORB-BRIEF + RANSAC ────────────────────────────

def detect_by_orb(frame_gray: np.ndarray,
                  ref_gray: np.ndarray,
                  ref_kp: list,
                  ref_des: np.ndarray,
                  roi: tuple,
                  orb: cv2.ORB) -> tuple | None:
    """
    ORB özellik eşleme + RANSAC ile yeniden tespit.
    Homografi inlier oranı eşiği aşarsa başarılı.

    Döndürür: (cx, cy, inlier_ratio) veya None
    """
    x1, y1, x2, y2 = roi
    search = frame_gray[y1:y2, x1:x2]
    if search.size == 0 or ref_des is None:
        return None

    kp2, des2 = orb.detectAndCompute(search, None)
    if des2 is None or len(kp2) < config.ORB_MIN_MATCHES:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(ref_des, des2)
    if len(matches) < config.ORB_MIN_MATCHES:
        return None

    matches = sorted(matches, key=lambda m: m.distance)
    good    = matches[:min(50, len(matches))]

    src_pts = np.float32(
        [ref_kp[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None

    inlier_ratio = float(mask.sum()) / len(mask)
    if inlier_ratio < config.ORB_INLIER_RATIO:
        return None

    rh, rw = ref_gray.shape[:2]
    center  = np.array([[[rw / 2, rh / 2]]], dtype=np.float32)
    mapped  = cv2.perspectiveTransform(center, H)[0][0]

    cx = int(x1 + mapped[0])
    cy = int(y1 + mapped[1])
    return (cx, cy, inlier_ratio)


# ── Referans Verisi Yöneticisi ───────────────────────────────

class RedetectionManager:
    """
    Takip noktası seçildiğinde referans verilerini kaydeder.

    Tutulan veriler:
        ref_hist  : L1 normalize H-S renk histogramı (K1)
        template  : gri görüntü yaması                (K2)
        ref_gray  : ORB için referans bölge           (K3)
        ref_kp    : referans anahtar noktaları        (K3)
        ref_des   : referans tanımlayıcılar           (K3)
        orb       : ORB dedektör nesnesi              (K3)

    Kademe geçiş mantığı bu sınıfta değil,
    tracker.py → _run_redetection metodundadır.
    """

    def __init__(self, frame: np.ndarray, frame_gray: np.ndarray,
                 x: int, y: int):
        # K1/K2 için küçük yama, K3 için daha büyük bölge
        s_small = config.NCC_PATCH_SIZE * 2
        s_orb   = config.NCC_PATCH_SIZE * config.ORB_PATCH_SCALE
        h, w    = frame.shape[:2]

        # K1 — Renk histogramı (küçük yama, L1 normalize)
        x1s = max(0, x - s_small // 2);  x2s = min(w, x + s_small // 2)
        y1s = max(0, y - s_small // 2);  y2s = min(h, y + s_small // 2)
        roi_bgr       = frame[y1s:y2s, x1s:x2s]
        roi_hsv       = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        self.ref_hist = cv2.calcHist(
            [roi_hsv], [0, 1], None, [18, 32], [0, 180, 0, 256]
        )
        cv2.normalize(self.ref_hist, self.ref_hist,
                      alpha=1, beta=0, norm_type=cv2.NORM_L1)

        # K2 — NCC şablonu (küçük yama)
        self.template = frame_gray[y1s:y2s, x1s:x2s].copy()

        # K3 — ORB referansı (daha büyük bölge)
        x1o = max(0, x - s_orb // 2);  x2o = min(w, x + s_orb // 2)
        y1o = max(0, y - s_orb // 2);  y2o = min(h, y + s_orb // 2)
        self.orb      = cv2.ORB_create(nfeatures=config.ORB_N_FEATURES)
        self.ref_gray = frame_gray[y1o:y2o, x1o:x2o].copy()
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(
            self.ref_gray, None
        )
        if self.ref_des is not None:
            print(f"[ORB] Referans: {len(self.ref_kp)} keypoint, "
                  f"bölge {self.ref_gray.shape}")