# =============================================================
# core/drift.py � Sürüklenme Tespiti
# =============================================================
# Ýki baðýmsýz mekanizma birlikte çalýþýr:
#
# Mekanizma 1 � Baðlantý Noktasý Geometri Testi:
#   Baþlangýçta seçilen k anchor noktasýnýn mevcut konumu
#   afin dönüþümle beklenen konumdan sapýyorsa sürüklenme.
#   e_anchor = ||p_current - H_est * p_anchor||_2
#
# Mekanizma 2 � Kalman Tutarsýzlýk Testi:
#   Kalman öngörüsü ile LK ölçümü arasýndaki Mahalanobis
#   mesafesi eþiði aþarsa sürüklenme.
#   d_K = sqrt((z - H*x^)^T * S^-1 * (z - H*x^))
#
# Karar: C¯ < CONF_DRIFT_THR VE (anchor VEYA Kalman) � sürüklenme
# =============================================================

import cv2
import numpy as np
import config


# ¦¦ Kalman Filtresi ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦

class PointKalman:
    """
    Tek bir takip noktasý için sabit hýz modeli Kalman filtresi.
    Durum: x^ = [x, y, vx, vy]^T
    """

    def __init__(self, x: float, y: float):
        self.kf = cv2.KalmanFilter(4, 2)   # 4 durum, 2 ölçüm

        # Geçiþ matrisi (sabit hýz)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Ölçüm matrisi (yalnýzca konum)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Gürültü kovaryanslarý
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Baþlangýç durumu
        self.kf.statePost = np.array(
            [[x], [y], [0.0], [0.0]], dtype=np.float32
        )

        self.initialized = True

    def predict(self) -> np.ndarray:
        """Bir sonraki konumu tahmin eder. (2,) array döner."""
        pred = self.kf.predict()
        return pred[:2].flatten()

    def update(self, x: float, y: float) -> float:
        """
        LK ölçümüyle Kalman durumunu günceller.
        Mahalanobis mesafesini (d_K) döndürür.
        """
        measurement = np.array([[x], [y]], dtype=np.float32)

        # Ýnovasyon: z - H*x^
        pred_state  = self.kf.statePre
        H           = self.kf.measurementMatrix
        innov       = measurement - H @ pred_state      # (2,1)

        # Ýnovasyon kovaryansý S = H*P*H^T + R
        P = self.kf.errorCovPre
        R = self.kf.measurementNoiseCov
        S = H @ P @ H.T + R                             # (2,2)

        # Mahalanobis mesafesi
        try:
            S_inv = np.linalg.inv(S)
            val   = innov.T @ S_inv @ innov
            d_k   = float(np.sqrt(float(np.squeeze(val))))
        except (np.linalg.LinAlgError, ValueError):
            d_k = 0.0

        self.kf.correct(measurement)
        return d_k

    def get_state(self) -> np.ndarray:
        """Mevcut durum tahminini (x, y) döndürür."""
        return self.kf.statePost[:2].flatten()


# ¦¦ Anchor (Baðlantý Noktasý) Yöneticisi ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦

class AnchorManager:
    """
    Baþlangýç karesinde hedef bölgesinden seçilen anchor
    noktalarýnýn mevcut konumlarýný takip eder ve geometrik
    tutarlýlýðý denetler.
    """

    def __init__(self, frame_gray: np.ndarray,
                 cx: int, cy: int, roi_size: int = 40):
        """
        cx, cy: takip noktasýnýn merkezi
        roi_size: anchor noktalarýnýn seçileceði bölge boyutu
        """
        h, w = frame_gray.shape
        x1 = max(0, cx - roi_size // 2)
        y1 = max(0, cy - roi_size // 2)
        x2 = min(w, cx + roi_size // 2)
        y2 = min(h, cy + roi_size // 2)

        roi = frame_gray[y1:y2, x1:x2]

        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners   = config.ANCHOR_COUNT,
            qualityLevel = 0.1,
            minDistance  = 5,
            blockSize    = 5,
        )

        if corners is not None and len(corners) > 0:
            # ROI koordinatlarýndan tam görüntü koordinatlarýna çevir
            self.anchors = corners.reshape(-1, 2) + np.array([x1, y1],
                                                              dtype=np.float32)
        else:
            # Köþe bulunamazsa merkezi anchor olarak kullan
            self.anchors = np.array([[cx, cy]], dtype=np.float32)

        self.ref_anchors = self.anchors.copy()   # baþlangýç referansý

    def update(self, prev_gray: np.ndarray,
               curr_gray: np.ndarray,
               lk_params: dict) -> float:
        """
        Anchor noktalarýný LK ile takip eder ve
        afin dönüþümden beklenen konumla karþýlaþtýrýr.
        Ortalama e_anchor hatasýný döndürür.
        """
        if len(self.anchors) == 0:
            return 0.0

        pts = self.anchors.reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **lk_params
        )

        good_cur = []
        good_ref = []
        for i, s in enumerate(status):
            if s[0] == 1:
                good_cur.append(next_pts[i][0])
                good_ref.append(self.ref_anchors[i])

        if len(good_cur) < 3:
            return config.ANCHOR_THRESHOLD + 1.0   # yetersiz nokta � uyarý

        cur_arr = np.array(good_cur, dtype=np.float32)
        ref_arr = np.array(good_ref, dtype=np.float32)

        # Afin dönüþüm kestirimi
        H, inliers = cv2.estimateAffinePartial2D(
            ref_arr, cur_arr, method=cv2.RANSAC
        )

        if H is None:
            return config.ANCHOR_THRESHOLD + 1.0

        # Beklenen konumlarý hesapla
        ref_h  = np.hstack([ref_arr,
                             np.ones((len(ref_arr), 1))]).T   # (3, N)
        expect = (H @ ref_h).T                                # (N, 2)

        # Hata
        errors = np.linalg.norm(cur_arr - expect, axis=1)
        self.anchors = cur_arr   # anchor'larý güncelle

        return float(np.mean(errors))


# ¦¦ Ana Sürüklenme Dedektörü ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦

class DriftDetector:
    """
    Tek bir TrackedPoint için anchor + Kalman tabanlý
    sürüklenme dedektörü.
    """

    def __init__(self, frame_gray: np.ndarray, x: float, y: float):
        self.kalman  = PointKalman(x, y)
        self.anchor  = AnchorManager(frame_gray, int(x), int(y))
        self.d_k     = 0.0
        self.e_anchor = 0.0

    def update(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
               x: float, y: float, lk_params: dict) -> bool:
        """
        Sürüklenme testi çalýþtýrýr.

        Döndürür:
            is_drifting : bool
        """
        # Kalman güncelle
        self.kalman.predict()
        self.d_k = self.kalman.update(x, y)

        # Anchor testi
        self.e_anchor = self.anchor.update(prev_gray, curr_gray, lk_params)

        # Karar: en az bir test eþiði aþmalý
        kalman_alarm = self.d_k     > config.KALMAN_THRESHOLD
        anchor_alarm = self.e_anchor > config.ANCHOR_THRESHOLD

        return bool(kalman_alarm and anchor_alarm)
