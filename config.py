# =============================================================
# config.py — Sistem genelinde kullanılan sabit parametreler
# =============================================================

# ── LK Temel Parametreleri ───────────────────────────────────
LK_WIN_SIZE      = (15, 15)   # Başlangıç pencere boyutu
LK_MAX_LEVEL     = 3          # Piramit seviyesi
LK_MAX_ITER      = 20         # Maksimum iterasyon
LK_EPSILON       = 0.03       # Durdurma kriteri
LK_MIN_EIG       = 1e-6       # minEigThreshold (textureless eleme)

# ── Adaptif Parametre Aralıkları ─────────────────────────────
WIN_SIZE_MIN     = 7
WIN_SIZE_MAX     = 21
MAX_LEVEL_MIN    = 1
MAX_LEVEL_MAX    = 4
ITER_MIN         = 5
ITER_MAX         = 30
EPS_MIN          = 0.01
EPS_MAX          = 0.03

# ── İleri-Geri Hata Eşiği ───────────────────────────────────
FB_THRESHOLD     = 3.0        # piksel cinsinden

# ── NCC Eşiği ───────────────────────────────────────────────
NCC_THRESHOLD    = 0.60
NCC_PATCH_SIZE   = 15         # şablon yama boyutu (piksel)

# ── Güven Skoru Ağırlıkları ──────────────────────────────────
W_NCC            = 0.35
W_FB             = 0.30
W_EIG            = 0.20
W_GRAD           = 0.15

# ── Güven Skoru Eşikleri ─────────────────────────────────────
CONF_DRIFT_THR   = 0.15       # altında → sürüklenme riski
CONF_LOST_THR    = 0.05       # altında → kayıp ilan et

# ── Sürüklenme Tespiti ───────────────────────────────────────
ANCHOR_COUNT     = 10         # bağlantı noktası sayısı
ANCHOR_THRESHOLD = 25.0        # piksel cinsinden e_anchor eşiği
KALMAN_THRESHOLD = 15.0        # Mahalanobis eşiği (χ² %95)

# ── Yeniden Tespit ───────────────────────────────────────────
HIST_THRESHOLD      = 0.35    # Bhattacharyya benzerlik skoru (1-dist)
MATCH_THRESHOLD     = 0.60    # NCC şablon eşleme skoru eşiği
ORB_N_FEATURES      = 500     # ORB özellik sayısı
SEARCH_SIGMA        = 2.0     # Kalman başlangıç σ çarpanı
SEARCH_SIGMA_MAX    = 5.0     # maksimum σ çarpanı
SEARCH_GROWTH_EVERY = 10      # her N kayıp karede bir genişle
SEARCH_GROWTH_RATE  = 0.3     # her adımda σ'ya eklenen miktar

# Kademe bazlı arama süresi (kare cinsinden)
# K1 bu kadar kare başarısız olursa K2'ye geçilir
# K2 bu kadar kare başarısız olursa K3'e geçilir
# K3 bu kadar kare başarısız olursa nokta kaldırılır
REDET_DELAY_FRAMES  = 30   # kayip sonrasi bu kadar bekle sonra ara
REDET_K1_MAX_FRAMES = 120     # ~4 sn @30fps
REDET_K2_MAX_FRAMES = 120     # ~4 sn @30fps
REDET_K3_MAX_FRAMES = 180     # ~6 sn @30fps

# ORB kalitesi
ORB_MIN_MATCHES     = 4      # RANSAC için minimum eşleşme sayısı
ORB_PATCH_SCALE     = 8      # ref_gray boyutu: NCC_PATCH_SIZE × bu değer
ORB_INLIER_RATIO    = 0.20   # minimum inlier oranı (düşürüldü: 0.30 → 0.25)

# ── Görüntü ──────────────────────────────────────────────────
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480

# ── Shi-Tomasi (nokta seçimi) ────────────────────────────────
ST_MAX_CORNERS   = 1          # başlangıçta her seçim için 1 merkez
ST_QUALITY       = 0.3
ST_MIN_DIST      = 7.0
ST_BLOCK_SIZE    = 7

# ── Katman Açma/Kapama Varsayılanları ────────────────────────
# True → aktif, False → devre dışı
LAYERS = {
    "fb_error"    : True,   # İleri-geri hata filtresi
    "confidence"  : True,   # Bileşik güven skoru
    "adaptive"    : True,   # Adaptif parametre kontrolü
    "drift"       : True,   # Sürüklenme tespiti
    "redetection" : True,   # Kademeli yeniden tespit
}
# Yeniden tespit onay mekanizması
REID_CONFIRM_REQUIRED = True  # False: otomatik onayla

