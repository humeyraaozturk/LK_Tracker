# =============================================================
# main.py — Giriş noktası
# =============================================================
# Klavye kısayolları:
#   Sol tık  → nokta ekle
#   C        → tüm noktaları sil
#   F        → ileri-geri hata katmanı aç/kapat
#   G        → güven skoru katmanı aç/kapat
#   A        → adaptif parametre katmanı aç/kapat
#   D        → sürüklenme tespiti katmanı aç/kapat
#   R        → yeniden tespit katmanı aç/kapat
#   Q / ESC  → çıkış
# =============================================================

import sys
import cv2

import config
from core.tracker      import PointTracker
from ui.display        import compose
from utils.performance import PerformanceMonitor
from utils.logger      import EventLogger


# ── Fare geri çağrımı ────────────────────────────────────────
def make_mouse_callback(tracker: PointTracker, frame_ref: list):
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if frame_ref[0] is not None:
                gray = cv2.cvtColor(frame_ref[0], cv2.COLOR_BGR2GRAY)
                tracker.add_point(x, y, gray, frame_bgr=frame_ref[0])
                print(f"[Nokta eklendi] id={len(tracker.points)-1}  "
                      f"konum=({x}, {y})")
    return callback


def main():
    # Kamera veya video dosyası
    source = 0                          # webcam için 0
    if len(sys.argv) > 1:
        source = sys.argv[1]            # python main.py video.mp4

    cap = cv2.VideoCapture(source)
    is_video = isinstance(source, str)

    if not is_video:
        # Webcam: çözünürlük ayarla
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[HATA] Kamera/video açılamadı.")
        sys.exit(1)

    # Video FPS'ini oku; webcam'de waitKey(1) kalır
    if is_video:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        # Her kare arasında beklenecek süre (ms)
        # İşlem süresi çıkarılacak, negatif olursa 1 ms kullanılır
        frame_delay_ms = int(1000.0 / video_fps)
        print(f"[Video] FPS: {video_fps:.1f}  →  kare gecikmesi: {frame_delay_ms} ms")
    else:
        frame_delay_ms = 1

    logger  = EventLogger(log_dir="logs")
    tracker = PointTracker(logger=logger)
    perf    = PerformanceMonitor(window=30)

    frame_ref = [None]                  # fare callback için paylaşılan referans

    WIN = "LK Tracker — Sol tık: nokta ekle"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, make_mouse_callback(tracker, frame_ref))

    print("=" * 60)
    print("  LK Tracker  —  Aşama 1")
    print("  Sol tık: nokta ekle  |  C: sil  |  Q/ESC: çıkış")
    print("  F G A D R : katman aç/kapat")
    print("=" * 60)

    while True:
        # ── Kamera okuma süresi ───────────────────────────────
        perf.start("capture")
        ret, frame = cap.read()
        perf.stop("capture")

        if not ret:
            print("[BİTTİ] Video sona erdi veya kamera hatası.")
            break

        frame_ref[0] = frame.copy()

        # ── Algoritma işlem süresi ────────────────────────────
        perf.start("process")
        states = tracker.update(frame)
        perf.stop("process")

        # ── Render süresi ─────────────────────────────────────
        perf.start("render")
        output = compose(
            frame         = frame,
            states        = states,
            fps           = perf.fps,
            avg_ms        = perf.avg_ms,
            layers        = tracker.layers,
            frame_conf    = tracker.frame_conf,
            adaptive_meta = tracker.adaptive_meta,
            log_events    = logger.get_display_events(),
            perf_summary  = perf.summary(),
        )
        perf.stop("render")

        # Toplam gecikme = capture + process + render
        perf.record_total()

        # Terminale her 15 karede bir yaz
        total_q = perf._times["total"]
        if len(total_q) % 15 == 0:
            print("\r" + perf.terminal_line(
                len(tracker.points), tracker.layers,
                tracker.frame_conf), end="", flush=True)

        cv2.imshow(WIN, output)

        # Video: hedef kare süresinden toplam işlem süresi çıkarılır
        wait = max(1, int(frame_delay_ms - perf.last("total")))
        key  = cv2.waitKey(wait) & 0xFF

        if key in (ord('q'), 27):           # Q veya ESC
            break
        elif key == ord('c'):
            tracker.remove_all()
            print("\n[Temizlendi] Tüm noktalar silindi.")
        elif key == ord('y'):
            tracker.confirm_all_pending()
        elif key == ord('n'):
            tracker.reject_all_pending()
        elif key == ord('f'):
            tracker.toggle_layer("fb_error")
        elif key == ord('g'):
            tracker.toggle_layer("confidence")
        elif key == ord('a'):
            tracker.toggle_layer("adaptive")
        elif key == ord('d'):
            tracker.toggle_layer("drift")
        elif key == ord('r'):
            tracker.toggle_layer("redetection")

    print()
    logger.session_end()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()