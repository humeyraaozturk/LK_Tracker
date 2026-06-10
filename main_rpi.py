# =============================================================
# main_rpi.py — RPi 5 + HQ Camera (IMX477) giriş noktası
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
# Gerekli ortam değişkenleri (~/.bashrc içinde tanımlı olmalı):
#   export LD_LIBRARY_PATH=/usr/local/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
#   export LIBCAMERA_IPA_MODULE_PATH=/usr/local/lib/aarch64-linux-gnu/libcamera/ipa
#   export LIBCAMERA_IPA_PROXY_PATH=/usr/local/libexec/libcamera
#   export GST_PLUGIN_PATH=/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0
# =============================================================

import sys
import os
import cv2

import config
from core.tracker      import PointTracker
from ui.display        import compose
from utils.performance import PerformanceMonitor
from utils.logger      import EventLogger


# ── GStreamer pipeline ────────────────────────────────────────
def make_gst_pipeline(width: int, height: int, fps: int = 30) -> str:
    return (
        f"libcamerasrc ! "
        f"videoconvert ! "
        f"video/x-raw,width={width},height={height},"
        f"framerate={fps}/1,format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


def open_rpi_camera(width: int, height: int, fps: int = 30) -> cv2.VideoCapture:
    """
    RPi HQ Camera'yı GStreamer + libcamerasrc üzerinden açar.
    Ortam değişkenleri ~/.bashrc'de tanımlı değilse burada da set edilir.
    """
    # Ortam değişkenlerini güvenceye al (bashrc yüklü değilse)
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/usr/local/lib/aarch64-linux-gnu"
    )
    os.environ.setdefault(
        "LIBCAMERA_IPA_MODULE_PATH",
        "/usr/local/lib/aarch64-linux-gnu/libcamera/ipa"
    )
    os.environ.setdefault(
        "LIBCAMERA_IPA_PROXY_PATH",
        "/usr/local/libexec/libcamera"
    )
    os.environ.setdefault(
        "GST_PLUGIN_PATH",
        "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0"
    )

    pipeline = make_gst_pipeline(width, height, fps)
    print(f"[RPi Kamera] GStreamer pipeline:\n  {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


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
    # Video dosyası argüman olarak verilmişse normal VideoCapture kullan
    is_video = len(sys.argv) > 1
    if is_video:
        source = sys.argv[1]               # python main_rpi.py video.mp4
        cap = cv2.VideoCapture(source)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        frame_delay_ms = int(1000.0 / video_fps)
        print(f"[Video] FPS: {video_fps:.1f}  →  kare gecikmesi: {frame_delay_ms} ms")
    else:
        # RPi HQ Camera
        cap = open_rpi_camera(
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            fps=30
        )
        frame_delay_ms = 1                 # GStreamer kendi senkronizasyonunu yapar

    if not cap.isOpened():
        print("[HATA] Kamera/video açılamadı.")
        print("  → Ortam değişkenlerini kontrol edin:")
        print("     source ~/.bashrc")
        print("  → Kameranın tanındığını doğrulayın:")
        print("     rpicam-hello --list-cameras")
        sys.exit(1)

    logger  = EventLogger(log_dir="logs")
    tracker = PointTracker(logger=logger)
    perf    = PerformanceMonitor(window=30)

    frame_ref = [None]                     # fare callback için paylaşılan referans

    WIN = "LK Tracker  [RPi HQ Camera]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, config.FRAME_WIDTH + 450, config.FRAME_HEIGHT + 300)
    cv2.setMouseCallback(WIN, make_mouse_callback(tracker, frame_ref))

    print("=" * 60)
    print("  LK Tracker  —  RPi 5 + HQ Camera (IMX477)")
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

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
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

        if key in (ord('q'), 27):          # Q veya ESC
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