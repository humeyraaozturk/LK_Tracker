# =============================================================
# core/confidence.py — Bileşik Güven Skoru
# =============================================================
# C(p) = w1*C_NCC + w2*C_FB + w3*C_eig + w4*C_grad
# C̄   = medyan(tüm noktaların C(p) değerleri)
# =============================================================

import cv2
import numpy as np
import config


# ── Bileşen hesaplayıcılar ───────────────────────────────────

def c_ncc(patch_ref: np.ndarray, patch_cur: np.ndarray) -> float:
    """
    Normalize çapraz korelasyon skoru.
    İki yama arasındaki benzerliği [0, 1] aralığında döndürür.
    """
    if patch_ref is None or patch_cur is None:
        return 0.0

    # Boyut eşitleme
    h = min(patch_ref.shape[0], patch_cur.shape[0])
    w = min(patch_ref.shape[1], patch_cur.shape[1])
    if h < 3 or w < 3:
        return 0.0

    r = patch_ref[:h, :w].astype(np.float32)
    c = patch_cur[:h, :w].astype(np.float32)

    r -= r.mean();  c -= c.mean()
    denom = np.std(r) * np.std(c)
    if denom < 1e-6:
        return 0.0

    ncc_val = float(np.sum(r * c) / (r.size * denom))
    return float(max(0.0, ncc_val))


def c_fb(fb_error: float, norm: float = 5.0) -> float:
    """
    İleri-geri hata değerini [0,1] güven skoruna dönüştürür.
    Hata 0 → skor 1.0,  hata ≥ norm → skor 0.0
    """
    return float(max(0.0, 1.0 - fb_error / norm))


def c_eig(gray: np.ndarray, x: int, y: int,
          patch_size: int = None) -> float:
    """
    Hessian matrisinin minimum özdeğerini normalize eder.
    Düz bölge → 0,  zengin doku → 1'e yakın.
    """
    s  = patch_size or config.NCC_PATCH_SIZE
    h, w = gray.shape
    x1 = max(0, x - s // 2);  x2 = min(w, x + s // 2 + 1)
    y1 = max(0, y - s // 2);  y2 = min(h, y + s // 2 + 1)
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    if patch.size < 9:
        return 0.0

    # Shi-Tomasi minimum özdeğer
    corners = cv2.goodFeaturesToTrack(
        patch,
        maxCorners  = 1,
        qualityLevel= 0.01,
        minDistance = 1,
        blockSize   = 5,
    )
    if corners is None:
        return 0.0

    # Hessian özdeğer hesabı
    Ix = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = (Ix * Ix).mean()
    Ixy = (Ix * Iy).mean()
    Iyy = (Iy * Iy).mean()

    trace    = Ixx + Iyy
    det      = Ixx * Iyy - Ixy ** 2
    disc     = max(0.0, (trace / 2) ** 2 - det)
    lambda_min = trace / 2 - np.sqrt(disc)

    # normalize: 0–1 arasına sıkıştır (log ölçeği daha dengeli)
    val = float(np.clip(lambda_min / 1000.0, 0.0, 1.0))
    return val


def c_grad(gray: np.ndarray, x: int, y: int,
           patch_size: int = None) -> float:
    """
    Yerel gradyan büyüklüğünü normalize eder.
    Düz bölge → 0,  keskin kenar → 1'e yakın.
    """
    s  = patch_size or config.NCC_PATCH_SIZE
    h, w = gray.shape
    x1 = max(0, x - s // 2);  x2 = min(w, x + s // 2 + 1)
    y1 = max(0, y - s // 2);  y2 = min(h, y + s // 2 + 1)
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    if patch.size < 9:
        return 0.0

    Ix   = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    Iy   = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(Ix**2 + Iy**2).mean()

    # normalize: tipik değer aralığı 0–200
    return float(np.clip(mag / 200.0, 0.0, 1.0))


# ── Ana fonksiyon ────────────────────────────────────────────

def compute_confidence(
    gray       : np.ndarray,
    x          : int,
    y          : int,
    fb_error   : float,
    patch_ref  : np.ndarray,
) -> tuple[float, dict]:
    """
    Tek bir nokta için bileşik güven skoru hesaplar.

    Döndürür:
        score      : float  — C(p) ∈ [0, 1]
        components : dict   — her bileşenin ayrı değeri
    """
    # Mevcut konumdaki yamayı kes
    s  = config.NCC_PATCH_SIZE
    h, w = gray.shape
    x1 = max(0, x - s // 2);  x2 = min(w, x + s // 2 + 1)
    y1 = max(0, y - s // 2);  y2 = min(h, y + s // 2 + 1)
    patch_cur = gray[y1:y2, x1:x2]

    # Bileşenler
    cncc  = c_ncc(patch_ref, patch_cur)
    cfb   = c_fb(fb_error)
    ceig  = c_eig(gray, x, y)
    cgrad = c_grad(gray, x, y)

    # Ağırlıklı toplam
    score = (
        config.W_NCC  * cncc  +
        config.W_FB   * cfb   +
        config.W_EIG  * ceig  +
        config.W_GRAD * cgrad
    )
    score = float(np.clip(score, 0.0, 1.0))

    components = {
        "ncc"  : round(cncc,  3),
        "fb"   : round(cfb,   3),
        "eig"  : round(ceig,  3),
        "grad" : round(cgrad, 3),
    }
    return score, components


def frame_confidence(scores: list[float]) -> float:
    """
    Tüm noktaların güven skorlarının medyanını döndürür.
    Boş liste için 0.0 döner.
    """
    if not scores:
        return 0.0
    return float(np.median(scores))