# =============================================================
# core/adaptive.py — Güven Güdümlü Adaptif Parametre Kontrolü
# =============================================================
# C̄ düştükçe parametreler büyür (daha dikkatli takip),
# C̄ yükseldikçe parametreler küçülür (daha hızlı işlem).
#
# winSize  = winSize_min  + (1 - C̄) * (winSize_max  - winSize_min)
# maxLevel = L_min        + floor((1 - C̄) * (L_max - L_min))
# iter     = iter_min     + floor((1 - C̄) * (iter_max - iter_min))
# eps      = EPS_max - (1 - C̄) * (EPS_max - EPS_min)
# =============================================================

import math
import cv2
import config


def compute_adaptive_params(frame_conf: float) -> dict:
    """
    Kare düzeyinde medyan güven skoruna göre LK parametrelerini hesaplar.

    Parametreler:
        frame_conf : float — C̄ ∈ [0, 1]

    Döndürür:
        lk_params : dict — cv2.calcOpticalFlowPyrLK'ya doğrudan verilebilir
        meta      : dict — insan okunabilir parametre özeti
    """
    c = float(max(0.0, min(1.0, frame_conf)))  # [0,1] sınırla
    inv = 1.0 - c                               # düşük güven → büyük inv

    # winSize: tek sayı olmalı (OpenCV gereksinimi)
    raw_win = config.WIN_SIZE_MIN + inv * (config.WIN_SIZE_MAX - config.WIN_SIZE_MIN)
    win     = int(raw_win)
    if win % 2 == 0:          # çift ise bir artır
        win += 1
    win = max(config.WIN_SIZE_MIN, min(config.WIN_SIZE_MAX, win))

    # maxLevel
    level = config.MAX_LEVEL_MIN + math.floor(
        inv * (config.MAX_LEVEL_MAX - config.MAX_LEVEL_MIN)
    )
    level = max(config.MAX_LEVEL_MIN, min(config.MAX_LEVEL_MAX, level))

    # iterasyon
    n_iter = config.ITER_MIN + math.floor(
        inv * (config.ITER_MAX - config.ITER_MIN)
    )
    n_iter = max(config.ITER_MIN, min(config.ITER_MAX, n_iter))

    # epsilon (yüksek güvende büyük eps → erken dur, az iterasyon)
    eps = config.EPS_MAX - inv * (config.EPS_MAX - config.EPS_MIN)
    eps = max(config.EPS_MIN, min(config.EPS_MAX, eps))

    lk_params = dict(
        winSize          = (win, win),
        maxLevel         = level,
        criteria         = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            n_iter,
            eps,
        ),
        minEigThreshold  = config.LK_MIN_EIG,
    )

    meta = {
        "winSize"  : win,
        "maxLevel" : level,
        "iter"     : n_iter,
        "eps"      : round(eps, 4),
    }
    return lk_params, meta