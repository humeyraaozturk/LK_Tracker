# =============================================================
# main_rpi.py — Raspberry Pi 5 + IMX477 HQ Kamera
# =============================================================
# Kullanım:
#   python main_rpi.py              # kamera ile
#   python main_rpi.py video.mp4   # video dosyası ile
#
# Klavye kısayolları:
#   Sol tık  → nokta ekle
#   C        → tüm noktaları sil
#   Y        → onay bekleyen noktayı onayla
#   N        → onay bekleyen noktayı reddet
#   F        → ileri-geri hata katmanı aç/kapat
#   G        → güven skoru katmanı aç/kapat
#   A        → adaptif parametre katmanı aç/kapat
#   D        → sürüklenme tespiti katmanı aç/kapat
#   R        → yeniden tespit katmanı aç/kapat
#   Q / ESC  → çıkış
# =============================================================

import sys
import cv2
import numpy as np

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
                print(f"[Nokta eklendi] id={len(tracker.points)-1} "
                      f"konum=({x}, {y})")
    return callback


def main():
    source   = sys.argv[1] if len(sys.argv) > 1 else None
    is_video = source is not None

    cap   = None
    picam = None

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[HATA] Video açılamadı.")
            sys.exit(1)
        video_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay_ms = int(1000.0 / video_fps)
        print(f"[Video] FPS: {video_fps:.1f}  gecikme: {frame_delay_ms}ms")

    else:
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("[HATA] picamera2 bulunamadı: pip install picamera2")
            sys.exit(1)

        picam = Picamera2()
        cam_cfg = picam.create_video_configuration(
            main={"size"  : (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                  "format": "BGR888"},
            controls={"FrameRate": 30},
        )
        picam.configure(cam_cfg)
        picam.start()
        frame_delay_ms = 1

        fmt = picam.camera_configuration()["main"]["format"]
        print(f"[Kamera] IMX477  {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"
              f"  30fps  format={fmt}")

        # Picamera2 bazı sistemlerde RGB verir, True yaparsanız BGR'ye çevrilir
        needs_rgb2bgr = True

    logger  = EventLogger(log_dir="logs")
    tracker = PointTracker(logger=logger)
    perf    = PerformanceMonitor(window=30)

    frame_ref         = [None]
    mouse_cb_set      = False
    mouse_cb_attempts = 0

    WIN = "LK Tracker"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, config.FRAME_WIDTH + 450, config.FRAME_HEIGHT + 300)

    print("=" * 60)
    print("  LK Tracker — Raspberry Pi")
    print("  Sol tık: nokta ekle  |  C: sil  |  Q/ESC: çıkış")
    print("  F G A D R : katman aç/kapat")
    print("  Y: onayla  N: reddet")
    print("=" * 60)

    while True:
        # ── Kamera / Video okuma ──────────────────────────────
        perf.start("capture")
        if is_video:
            ret, frame = cap.read()
            if not ret:
                print("[BİTTİ] Video sona erdi.")
                break
        else:
            frame = picam.capture_array()
            if needs_rgb2bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        perf.stop("capture")

        frame_ref[0] = frame.copy()

        # ── Algoritma ─────────────────────────────────────────
        perf.start("process")
        states = tracker.update(frame)
        perf.stop("process")

        # ── Render ────────────────────────────────────────────
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
        perf.record_total()

        # Terminal çıktısı — her 15 karede bir
        total_q = perf._times["total"]
        if len(total_q) > 0 and len(total_q) % 15 == 0:
            print("\r" + perf.terminal_line(
                len(tracker.points), tracker.layers,
                tracker.frame_conf), end="", flush=True)

        cv2.imshow(WIN, output)

        # ── Mouse callback: imshow sonrası ayarla ─────────────
        if not mouse_cb_set and mouse_cb_attempts < 30:
            mouse_cb_attempts += 1
            try:
                cv2.setMouseCallback(
                    WIN, make_mouse_callback(tracker, frame_ref)
                )
                mouse_cb_set = True
                print(f"\n[OK] Mouse callback ayarlandı "
                      f"(deneme {mouse_cb_attempts})")
            except cv2.error:
                if mouse_cb_attempts == 30:
                    print("\n[HATA] Mouse callback ayarlanamadı.")

        # ── Bekleme ve klavye ─────────────────────────────────
        wait = max(1, int(frame_delay_ms - perf.last("total")))
        key  = cv2.waitKey(wait) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            tracker.remove_all()
            print("\n[Temizlendi]")
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
    if cap is not None:
        cap.release()
    if picam is not None:
        picam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()