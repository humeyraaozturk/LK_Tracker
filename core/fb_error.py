# =============================================================
# core/fb_error.py — İleri-Geri Hata Analizi
# =============================================================
# Bir noktayı t → t+1 ileri, t+1 → t geri takip eder.
# Başlangıç konumuna dönüş mesafesi eşiği aşarsa nokta elenir.
# Referans: Kalal vd. (2010)
# =============================================================

import cv2
import numpy as np
import config


def compute_fb_error(
    prev_gray  : np.ndarray,
    curr_gray  : np.ndarray,
    prev_pts   : np.ndarray,   # (N, 1, 2) float32
    next_pts   : np.ndarray,   # (N, 1, 2) float32  — LK ileri çıktısı
    lk_params  : dict,
) -> np.ndarray:
    """
    Her nokta için ileri-geri hata mesafesini hesaplar.

    Döndürür:
        fb_errors : (N,) float32 — piksel cinsinden hata mesafesi
    """
    if len(prev_pts) == 0:
        return np.array([], dtype=np.float32)

    # Geri yön LK: curr_gray → prev_gray, next_pts'ten prev_pts'e
    back_pts, status_back, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, next_pts, None, **lk_params
    )

    # ||p_t - LK_bwd(LK_fwd(p_t))||_2
    diff      = prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2)
    fb_errors = np.linalg.norm(diff, axis=1).astype(np.float32)

    # Geri takip başarısız olan noktaları büyük hata ile işaretle
    for i, s in enumerate(status_back):
        if s[0] == 0:
            fb_errors[i] = 999.0

    return fb_errors


def filter_by_fb(
    fb_errors : np.ndarray,
    threshold : float = None,
) -> np.ndarray:
    """
    Eşiği aşan noktalar için False döndüren maske üretir.

    Döndürür:
        mask : (N,) bool — True → geçerli, False → elendi
    """
    thr = threshold if threshold is not None else config.FB_THRESHOLD
    return fb_errors <= thr