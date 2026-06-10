# =============================================================
# core/tracker.py — Temel Lucas-Kanade takip motoru
# Aşama 1: Çoklu nokta seçimi + temel LK
# =============================================================

import cv2
import numpy as np
import time
import config
from core.fb_error     import compute_fb_error, filter_by_fb
from core.confidence   import compute_confidence, frame_confidence
from core.adaptive     import compute_adaptive_params
from core.drift        import DriftDetector
from core.redetection  import RedetectionManager


class PointTracker:
    """
    Kullanıcının fare ile seçtiği noktaları Lucas-Kanade
    optik akış yöntemiyle takip eder.

    Her nokta bağımsız bir TrackedPoint nesnesi olarak tutulur.
    Katmanlar config.LAYERS sözlüğünden açılıp kapatılabilir.
    """

    def __init__(self, layers: dict = None, logger=None):
        self.layers = dict(config.LAYERS)
        if layers:
            self.layers.update(layers)

        self.points: list[TrackedPoint] = []
        self.logger = logger   # EventLogger — None ise log atılmaz

        self.lk_params = dict(
            winSize      = config.LK_WIN_SIZE,
            maxLevel     = config.LK_MAX_LEVEL,
            criteria     = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.LK_MAX_ITER,
                config.LK_EPSILON,
            ),
            minEigThreshold = config.LK_MIN_EIG,
        )

        self.prev_gray     = None
        self.frame_conf    = 1.0
        self.adaptive_meta = {}

    # ── Dışarıdan çağrılan ana metodlar ─────────────────────

    def add_point(self, x: int, y: int, frame_gray: np.ndarray,
                  frame_bgr: np.ndarray = None):
        """Kullanıcı tıklamasıyla yeni bir takip noktası ekler."""
        pt = TrackedPoint(x, y, frame_gray, point_id=len(self.points))
        if frame_bgr is not None:
            pt.redet_manager = RedetectionManager(
                frame_bgr, frame_gray, x, y
            )
        self.points.append(pt)
        if self.logger:
            self.logger.point_added(pt.id, x, y)

    def remove_all(self):
        """Tüm noktaları siler."""
        self.points.clear()
        if self.logger:
            self.logger.points_cleared()

    def update(self, frame: np.ndarray) -> list:
        """Her kare için çağrılır."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or len(self.points) == 0:
            self.prev_gray = gray
            return self._get_states()

        # Duruma göre ayır
        # pending noktalar ne LK'ya ne redet'e girer, onay bekler
        active  = [p for p in self.points
                   if p.state in ("tracking", "drifting")]
        lost    = [p for p in self.points if p.state == "lost"]
        pending = [p for p in self.points if p.state == "pending"]

        # ── Lost noktalar: Kalman predict ile arama bölgesini güncelle ──
        for pt in lost:
            pt.drift_detector.kalman.predict()

        # ── Pending noktalar: önerilen konumu LK ile takip et ────────────
        if pending and self.prev_gray is not None:
            pend_pts = np.array(
                [p.pending_pos for p in pending], dtype=np.float32
            ).reshape(-1, 1, 2)

            # Sabit parametrelerle hafif LK (pending için adaptif gerekmez)
            lk_pend = dict(
                winSize         = (15, 15),
                maxLevel        = 2,
                criteria        = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    10, 0.03,
                ),
                minEigThreshold = config.LK_MIN_EIG,
            )
            next_pend, status_pend, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pend_pts, None, **lk_pend
            )
            for i, pt in enumerate(pending):
                if status_pend[i][0] == 1:
                    pt.pending_pos = next_pend[i][0]

        # ── Aktif noktalar varsa LK çalıştır ─────────────────────────
        if active:
            prev_pts = np.array(
                [p.position for p in active], dtype=np.float32
            ).reshape(-1, 1, 2)

            # Adaptif parametre
            if self.layers["adaptive"]:
                new_params, self.adaptive_meta = compute_adaptive_params(
                    self.frame_conf
                )
                self.lk_params = new_params
            else:
                self.lk_params = dict(
                    winSize         = config.LK_WIN_SIZE,
                    maxLevel        = config.LK_MAX_LEVEL,
                    criteria        = (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        config.LK_MAX_ITER,
                        config.LK_EPSILON,
                    ),
                    minEigThreshold = config.LK_MIN_EIG,
                )
                self.adaptive_meta = {
                    "winSize"  : config.LK_WIN_SIZE[0],
                    "maxLevel" : config.LK_MAX_LEVEL,
                    "iter"     : config.LK_MAX_ITER,
                    "eps"      : config.LK_EPSILON,
                }

            # LK ileri takip
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None, **self.lk_params
            )

            # İleri-Geri hata
            if self.layers["fb_error"]:
                fb_errors = compute_fb_error(
                    self.prev_gray, gray,
                    prev_pts, next_pts, self.lk_params,
                )
                fb_mask = filter_by_fb(fb_errors)
                
                # Global hareket tespiti: aktif noktaların %70'inden fazlası
                # FB'den başarısız olduysa kamera sarsıntısı varsay,
                # bu karede FB filtresini atla
                if len(active) >= 2:
                    fail_ratio = 1.0 - (fb_mask.sum() / len(fb_mask))
                    if fail_ratio >= 0.50:
                        fb_mask = np.ones(len(active), dtype=bool)
            else:
                fb_errors = np.zeros(len(active), dtype=np.float32)
                fb_mask   = np.ones(len(active), dtype=bool)

            for i, pt in enumerate(active):
                prev_state = pt.state
                lk_ok = status[i][0] == 1
                fb_ok = bool(fb_mask[i])

                in_warmup = pt.warmup_frames > 0
                if in_warmup:
                    pt.warmup_frames -= 1

                if lk_ok and fb_ok:
                    new_pos = next_pts[i][0]
                    fb_err  = float(fb_errors[i])
                    pt.update_position(
                        new_pos, gray, self.layers, fb_err,
                        prev_gray=self.prev_gray,
                        lk_params=self.lk_params,
                    )
                elif lk_ok and not fb_ok and in_warmup:
                    # Warmup'ta FB başarısız ama LK buldu:
                    # FB eşiğini 2 katına çıkararak daha toleranslı ol
                    fb_err = float(fb_errors[i])
                    if fb_err < config.FB_THRESHOLD * 2.0:
                        new_pos = next_pts[i][0]
                        pt.update_position(
                            new_pos, gray, self.layers, fb_err,
                            prev_gray=self.prev_gray,
                            lk_params=self.lk_params,
                        )
                    else:
                        pt.state    = "lost"
                        pt.fb_error = fb_err
                        pt.searching_logged = False
                else:
                    pt.state    = "lost"
                    pt.fb_error = float(fb_errors[i]) if lk_ok else 999.0
                    pt.searching_logged = False

                # Durum değişikliği logu (tracking/drifting → lost)
                if self.logger and pt.state != prev_state:
                    if pt.state == "lost":
                        self.logger.point_lost(pt.id)
                    elif pt.state == "drifting" and prev_state == "tracking":
                        self.logger.point_drifting(
                            pt.id, pt.d_k, pt.e_anchor
                        )

            # Güven skoru
            scores = []
            for pt in active:
                if pt.state == "tracking":
                    if self.layers["confidence"]:
                        score, comps = compute_confidence(
                            gray,
                            int(pt.position[0]),
                            int(pt.position[1]),
                            pt.fb_error,
                            pt.template,
                        )
                        pt.conf       = score
                        pt.conf_comps = comps

                        # Warmup'ta eşikler daha toleranslı
                        in_warmup   = pt.warmup_frames > 0
                        lost_thr    = (config.CONF_LOST_THR * 0.5
                                       if in_warmup
                                       else config.CONF_LOST_THR)
                        drift_thr   = (config.CONF_DRIFT_THR * 0.7
                                       if in_warmup
                                       else config.CONF_DRIFT_THR)

                        prev_state = pt.state
                        if score < lost_thr:
                            pt.state = "lost"
                            pt.searching_logged = False
                            if self.logger:
                                self.logger.point_lost(pt.id)
                        elif score < drift_thr and self.layers.get("drift"):
                            if prev_state == "tracking" and self.logger:
                                self.logger.point_drifting(
                                    pt.id, pt.d_k, pt.e_anchor
                                )
                            pt.state = "drifting"
                            #if self.layers.get("drift"):
                            #    pt.state = "drifting"
                        else:
                            pt.state = "tracking"
                    else:
                        pt.conf = 1.0
                    scores.append(pt.conf)

            self.frame_conf = frame_confidence(scores)

        # ── Yeniden Tespit ────────────────────────────────────────────
        self._run_redetection(frame, gray)

        self.prev_gray = gray
        return self._get_states()

    def _run_redetection(self, frame: np.ndarray, gray: np.ndarray):
        """
        Kayıp noktalar için kademe bazlı yeniden tespit.

        Her kare yalnızca aktif kademe çalışır:
          K1 → REDET_K1_MAX_FRAMES başarısız → K2'ye geç
          K2 → REDET_K2_MAX_FRAMES başarısız → K3'e geç
          K3 → başarı bulana kadar süresiz devam
        """
        if not self.layers["redetection"]:
            return

        PENDING_TIMEOUT = 5.0   # saniye
        to_remove = []

        for pt in self.points:
            # ── Pending: onay bekleniyor ──────────────────────
            if pt.state == "pending":
                elapsed = time.time() - pt.pending_time
                if elapsed >= PENDING_TIMEOUT:
                    self._confirm_redet(pt)
                    if self.logger:
                        self.logger.log(
                            "redet",
                            f"Nokta #{pt.id} otomatik onaylandı "
                            f"(süre doldu)"
                        )
                continue

            if pt.state != "lost" or pt.redet_manager is None:
                if pt.state == "tracking":
                    pt.redet_stage   = 0
                    pt.lost_frames   = 0
                    pt.active_kademe = 1
                    pt.kademe_frames = 0
                continue

            pt.lost_frames   += 1
            pt.kademe_frames += 1
            if pt.lost_frames < config.REDET_DELAY_FRAMES:
                continue

            if self.logger and not pt.searching_logged:
                self.logger.log(
                    "redet",
                    f"Nokta #{pt.id} aranıyor  "
                    f"[Kademe {pt.active_kademe}]  "
                    f"(kare {pt.kademe_frames})"
                )
                pt.searching_logged = True

            kalman_pos = pt.drift_detector.kalman.get_state()
            roi        = self._compute_roi(frame, kalman_pos, pt.lost_frames)
            result     = None

            if pt.active_kademe == 1:
                result = self._try_k1(frame, gray, roi, pt)
                if result is None:
                    if pt.kademe_frames >= config.REDET_K1_MAX_FRAMES:
                        self._advance_kademe(pt, from_k=1)

            elif pt.active_kademe == 2:
                result = self._try_k2(gray, roi, pt)
                if result is None:
                    if pt.kademe_frames >= config.REDET_K2_MAX_FRAMES:
                        self._advance_kademe(pt, from_k=2)

            elif pt.active_kademe == 3:
                result = self._try_k3(gray, roi, pt)
                if result is None:
                    if pt.kademe_frames >= config.REDET_K3_MAX_FRAMES:
                        to_remove.append(pt.id)
                        if self.logger:
                            self.logger.log(
                                "warn",
                                f"Nokta #{pt.id} kaldırıldı  "
                                f"(tüm kademeler başarısız, "
                                f"{pt.lost_frames} kare)"
                            )
                        continue

            if result is not None:
                cx, cy, stage = result
                if config.REID_CONFIRM_REQUIRED:
                    pt.pending_pos   = np.array([cx, cy], dtype=np.float32)
                    pt.pending_stage = stage
                    pt.pending_time  = time.time()
                    pt.state         = "pending"
                    if self.logger:
                        self.logger.log(
                            "redet",
                            f"Nokta #{pt.id} bulundu [K{stage}]  "
                            f"({cx},{cy})  — onay bekleniyor (Y/N)"
                        )
                else:
                    # Otomatik onayla
                    self._confirm_redet_direct(pt, cx, cy, stage)
                if self.logger:
                    self.logger.log(
                        "redet",
                        f"Nokta #{pt.id} bulundu [K{stage}]  "
                        f"({cx},{cy})  — onay bekleniyor (Y/N)"
                    )

        if to_remove:
            self.points = [p for p in self.points
                           if p.id not in to_remove]

    def _confirm_redet(self, pt):
        """Pending noktayı onaylar, tracking'e alır."""
        cx, cy = int(pt.pending_pos[0]), int(pt.pending_pos[1])
        pt.position         = pt.pending_pos.copy()
        pt.redet_stage      = pt.pending_stage
        pt.state            = "tracking"
        pt.conf             = 0.6
        pt.warmup_frames    = 15
        pt.lost_frames      = 0
        pt.kademe_frames    = 0
        pt.active_kademe    = 1
        pt.searching_logged = False
        pt.pending_pos      = None
        pt.pending_stage    = 0
        pt.pending_time     = None
        pt.pending_skipped_stages.clear()
        pt.trail.append((cx, cy))
        pt.drift_detector.kalman.kf.statePost = np.array(
            [[float(cx)], [float(cy)], [0.0], [0.0]],
            dtype=np.float32
        )
        if self.logger:
            self.logger.redet_found(pt.id, pt.redet_stage, cx, cy)

    def _confirm_redet_direct(self, pt, cx: int, cy: int, stage: int):
        """Onay beklemeden doğrudan tracking'e al."""
        pt.position         = np.array([cx, cy], dtype=np.float32)
        pt.redet_stage      = stage
        pt.state            = "tracking"
        pt.conf             = 0.6
        pt.warmup_frames    = 15
        pt.lost_frames      = 0
        pt.kademe_frames    = 0
        pt.active_kademe    = 1
        pt.searching_logged = False
        pt.pending_pos      = None
        pt.pending_stage    = 0
        pt.pending_time     = None
        pt.pending_skipped_stages.clear()
        pt.trail.append((cx, cy))
        pt.drift_detector.kalman.kf.statePost = np.array(
            [[float(cx)], [float(cy)], [0.0], [0.0]],
            dtype=np.float32
        )
        if self.logger:
            self.logger.redet_found(pt.id, stage, cx, cy)
        
    def _reject_redet(self, pt):
        """Pending noktayı reddeder, bir sonraki kademeden devam eder."""
        rejected_stage = pt.pending_stage
        pt.pending_skipped_stages.add(rejected_stage)
        pt.state         = "lost"
        pt.pending_pos   = None
        pt.pending_stage = 0
        pt.pending_time  = None

        # Reddedilen kademeden sonraki kademeye geç
        next_k = rejected_stage + 1
        if next_k > 3:
            next_k = 3   # K3'ten öteye geçilemez, K3 süresiz devam eder

        if pt.active_kademe < next_k:
            pt.active_kademe    = next_k
            pt.kademe_frames    = 0
            pt.searching_logged = False

        if self.logger:
            self.logger.log(
                "warn",
                f"Nokta #{pt.id} K{rejected_stage} reddedildi  "
                f"→ K{pt.active_kademe}'den devam"
            )

    def confirm_pending(self, point_id: int):
        """Dışarıdan çağrılır (Y tuşu). Pending noktayı onaylar."""
        for pt in self.points:
            if pt.id == point_id and pt.state == "pending":
                self._confirm_redet(pt)
                break

    def reject_pending(self, point_id: int):
        """Dışarıdan çağrılır (N tuşu). Pending noktayı reddeder."""
        for pt in self.points:
            if pt.id == point_id and pt.state == "pending":
                self._reject_redet(pt)
                break

    def confirm_all_pending(self):
        """Tüm pending noktaları onaylar."""
        for pt in self.points:
            if pt.state == "pending":
                self._confirm_redet(pt)

    def reject_all_pending(self):
        """Tüm pending noktaları reddeder."""
        for pt in self.points:
            if pt.state == "pending":
                self._reject_redet(pt)

    def _compute_roi(self, frame: np.ndarray,
                     kalman_pos: np.ndarray,
                     lost_frames: int) -> tuple:
        """Kalman merkezli, lost_frames ile genişleyen arama bölgesi."""
        steps     = lost_frames // config.SEARCH_GROWTH_EVERY
        sigma_eff = config.SEARCH_SIGMA + steps * config.SEARCH_GROWTH_RATE
        sigma_eff = min(sigma_eff, config.SEARCH_SIGMA_MAX)
        r         = int(sigma_eff * 40)

        h, w = frame.shape[:2]
        cx   = int(kalman_pos[0])
        cy   = int(kalman_pos[1])
        x1   = max(0, cx - r);  x2 = min(w, cx + r)
        y1   = max(0, cy - r);  y2 = min(h, cy + r)
        return x1, y1, x2, y2

    def _advance_kademe(self, pt, from_k: int):
        """Bir sonraki kademeye geç, logu yaz."""
        pt.active_kademe    = from_k + 1
        pt.kademe_frames    = 0
        pt.searching_logged = False   # yeni kademe için log tetiklensin
        if self.logger:
            self.logger.log(
                "redet",
                f"Nokta #{pt.id}  K{from_k} başarısız  "
                f"→ K{from_k + 1}'e geçiliyor"
            )

    def _try_k1(self, frame, gray, roi, pt) -> tuple | None:
        from core.redetection import detect_by_histogram
        result = detect_by_histogram(
            frame, pt.redet_manager.ref_hist, roi
        )
        if result:
            return (result[0], result[1], 1)
        return None

    def _try_k2(self, gray, roi, pt) -> tuple | None:
        from core.redetection import detect_by_template
        result = detect_by_template(
            gray, pt.redet_manager.template, roi
        )
        if result:
            return (result[0], result[1], 2)
        return None

    def _try_k3(self, gray, roi, pt) -> tuple | None:
        from core.redetection import detect_by_orb
        result = detect_by_orb(
            gray,
            pt.redet_manager.ref_gray,
            pt.redet_manager.ref_kp,
            pt.redet_manager.ref_des,
            roi,
            pt.redet_manager.orb,
        )
        if result:
            return (result[0], result[1], 3)
        return None

    def toggle_layer(self, layer_name: str):
        """Katmanı açar/kapatır."""
        if layer_name in self.layers:
            self.layers[layer_name] = not self.layers[layer_name]
            if self.logger:
                self.logger.layer_toggled(
                    layer_name, self.layers[layer_name]
                )

    def set_layer(self, layer_name: str, value: bool):
        if layer_name in self.layers:
            self.layers[layer_name] = value

    # ── Yardımcı ─────────────────────────────────────────────

    def _get_states(self) -> list:
        return [p.get_info() for p in self.points]


# =============================================================
class TrackedPoint:
    """
    Takip edilen tek bir noktanın tüm durumunu tutar.
    state: "tracking" | "drifting" | "lost"
    """

    def __init__(self, x: int, y: int, frame_gray: np.ndarray, point_id: int):
        self.id             = point_id
        self.position       = np.array([x, y], dtype=np.float32)
        self.state          = "tracking"
        self.prev_state     = "tracking"
        self.conf           = 1.0
        self.conf_comps     = {}
        self.fb_error       = 0.0
        self.d_k            = 0.0
        self.e_anchor       = 0.0
        self.redet_stage    = 0
        self.warmup_frames  = 10
        self.lost_frames    = 0
        self.active_kademe  = 1
        self.kademe_frames  = 0
        self.searching_logged = False

        # Onay mekanizması
        self.pending_pos    = None   # onay bekleyen konum
        self.pending_stage  = 0      # onay bekleyen kademe
        self.pending_time   = None   # onay başlangıç zamanı (time.time())
        self.pending_skipped_stages = set()  # reddedilen kademeler

        self.trail          = [(x, y)]

        # Şablon
        self.template     = self._extract_patch(frame_gray, x, y)
        self.template_pos = np.array([x, y], dtype=np.float32)

        # Sürüklenme dedektörü
        self.drift_detector = DriftDetector(frame_gray, float(x), float(y))

        # Yeniden tespit yöneticisi
        self.redet_manager = None  # add_point sırasında frame gerekli

    def update_position(self, new_pos: np.ndarray,
                        gray: np.ndarray, layers: dict,
                        fb_err: float = 0.0,
                        prev_gray: np.ndarray = None,
                        lk_params: dict = None):
        """
        LK çıktısını alır, aktif katmanlardan geçirir,
        konumu günceller.
        """
        self.position  = new_pos
        self.fb_error  = fb_err
        self.trail.append((int(new_pos[0]), int(new_pos[1])))
        if len(self.trail) > 60:
            self.trail.pop(0)

        # ── Sürüklenme Tespiti ───────────────────────────────
        # Yalnızca güven riski varsa (drifting) VE katman aktifse
        if (layers.get("drift") and
                self.state in ("tracking", "drifting") and
                self.warmup_frames == 0 and
                prev_gray is not None and
                lk_params is not None):
            is_drifting = self.drift_detector.update(
                prev_gray, gray,
                float(new_pos[0]), float(new_pos[1]),
                lk_params,
            )
            self.d_k      = self.drift_detector.d_k
            self.e_anchor = self.drift_detector.e_anchor

            if is_drifting:
                if self.state == "tracking":
                    self.state = "drifting"   # tracking → drifting
                else:
                    self.state = "lost"       # drifting → lost
                    
            else:
                # Drift kapalı veya warmup'ta — Kalman'ı yine de güncelle
                if layers.get("drift") and lk_params is not None:
                    self.drift_detector.kalman.predict()
                    self.d_k = self.drift_detector.kalman.update(
                        float(new_pos[0]), float(new_pos[1])
                    )
                self.e_anchor = 0.0

    def get_info(self) -> dict:
        return {
            "id"            : self.id,
            "pos"           : self.position.copy(),
            "state"         : self.state,
            "conf"          : self.conf,
            "conf_comps"    : self.conf_comps,
            "fb_error"      : self.fb_error,
            "d_k"           : self.d_k,
            "e_anchor"      : self.e_anchor,
            "redet_stage"   : self.redet_stage,
            "lost_frames"   : self.lost_frames,
            "active_kademe" : self.active_kademe,
            "kademe_frames" : self.kademe_frames,
            "pending_pos"   : (self.pending_pos.copy()
                               if self.pending_pos is not None else None),
            "pending_stage" : self.pending_stage,
            "pending_time"  : self.pending_time,
            "trail"         : list(self.trail),
        }

    @staticmethod
    def _extract_patch(gray: np.ndarray, x: int, y: int,
                       size: int = None) -> np.ndarray | None:
        """Verilen konumdan kare yama keser."""
        s = size or config.NCC_PATCH_SIZE
        h, w = gray.shape
        x1 = max(0, x - s // 2)
        y1 = max(0, y - s // 2)
        x2 = min(w, x + s // 2 + 1)
        y2 = min(h, y + s // 2 + 1)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch.copy()