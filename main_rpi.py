# =============================================================
# main_rpi.py � Raspberry Pi 5 + IMX477 HQ Kamera giriþ noktasý
# =============================================================
# Picamera2 ile CSI kamera desteði.
# Bilgisayarda test için main.py kullanýn.
#
# Kurulum:
#   pip install picamera2
#
# Çalýþtýrma:
#   python main_rpi.py              # kamera ile
#   python main_rpi.py video.mp4   # video dosyasý ile
# =============================================================

import sys
import cv2
import numpy as np

import config
from core.tracker      import PointTracker
from ui.display        import compose
from utils.performance import PerformanceMonitor
from utils.logger      import EventLogger


# ¦¦ Fare geri çaðrýmý ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
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
    source   = sys.argv[1] if len(sys.argv) > 1 else None
    is_video = source is not None

    # ¦¦ Kamera baþlatma ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    cap     = None
    picam   = None

    if is_video:
        # Video dosyasý � standart OpenCV
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[HATA] Video açýlamadý.")
            sys.exit(1)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay_ms = int(1000.0 / video_fps)
        print(f"[Video] FPS: {video_fps:.1f}  gecikme: {frame_delay_ms}ms")

    else:
        # CSI kamera � Picamera2
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("[HATA] picamera2 bulunamadý: pip install picamera2")
            sys.exit(1)

        picam = Picamera2()
        # 1332x990 @120fps veya 2028x1520 @40fps
        # 640x480 crop ile 30+ fps için en uygun mod: 1332x990
        cam_cfg = picam.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                  "format": "BGR888"},
            controls={"FrameRate": 30},
        )
        picam.configure(cam_cfg)
        picam.start()
        frame_delay_ms = 1   # Picamera2 zaten FPS'i kontrol ediyor
        print(f"[Kamera] IMX477  {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}  30fps")

    logger  = EventLogger(log_dir="logs")
    tracker = PointTracker(logger=logger)
    perf    = PerformanceMonitor(window=30)

    frame_ref = [None]                  # fare callback için paylaþýlan referans

    WIN = "LK Tracker � Sol tik: nokta ekle"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 940, 760)

    mouse_callback_set = False

    print("=" * 60)
    print("  LK Tracker  �  Aþama 1")
    print("  Sol týk: nokta ekle  |  C: sil  |  Q/ESC: çýkýþ")
    print("  F G A D R : katman aç/kapat")
    print("=" * 60)

    while True:
        # ¦¦ Kamera okuma ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        perf.start("capture")
        if is_video:
            ret, frame = cap.read()
            if not ret:
                print("[BÝTTÝ] Video sona erdi.")
                break
        else:
            # Picamera2: capture_array direkt numpy array döner
            frame = picam.capture_array()
            # Picamera2 BGR888 formatýnda veriyor, dönüþüm gerekmez
        perf.stop("capture")

        frame_ref[0] = frame.copy()

        # ¦¦ Algoritma iþlem süresi ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        perf.start("process")
        states = tracker.update(frame)
        perf.stop("process")

        # ¦¦ Render süresi ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
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

        # Ýlk kare gösterildikten sonra mouse callback'i ayarla
        if not mouse_callback_set:
            try:
                cv2.setMouseCallback(WIN, make_mouse_callback(tracker, frame_ref))
                mouse_callback_set = True
            except cv2.error:
                pass

        # Video: hedef kare süresinden toplam iþlem süresi çýkarýlýr
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
    if cap is not None:
        cap.release()
    if picam is not None:
        picam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
