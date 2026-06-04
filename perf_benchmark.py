# =============================================================
# perf_benchmark.py — Raspberry Pi 5 Performans Ölçümü
# =============================================================
# Kullanım:
#   python perf_benchmark.py --video video.mp4
#   python perf_benchmark.py --video video.mp4 --frames 300
# =============================================================

import argparse
import time
import statistics
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.tracker import PointTracker
import config


def benchmark(video_path: str, n_frames: int = 300):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        sys.exit(1)

    # İlk kareyi al ve noktayı ekle
    ret, frame = cap.read()
    if not ret:
        print("[HATA] Video okunamadı.")
        sys.exit(1)

    frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Kare ortasına nokta ekle
    cx = config.FRAME_WIDTH  // 2
    cy = config.FRAME_HEIGHT // 2

    tracker = PointTracker()
    tracker.add_point(cx, cy, gray, frame_bgr=frame)

    # ── Isınma turları ────────────────────────────────────────
    print("[Isınma] 30 kare...")
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = cap.read()
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        tracker.update(frame)

    # ── Ölçüm ─────────────────────────────────────────────────
    print(f"[Ölçüm] {n_frames} kare ölçülüyor...")

    cap_times   = []
    proc_times  = []
    total_times = []

    for i in range(n_frames):
        # Kare okuma
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = cap.read()
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        t1 = time.perf_counter()

        # Algoritma
        tracker.update(frame)
        t2 = time.perf_counter()

        cap_times.append((t1 - t0) * 1000)
        proc_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_frames}", end="\r")

    cap.release()
    print()

    # ── Sonuçlar ──────────────────────────────────────────────
    def stats(data):
        return {
            "ort"  : statistics.mean(data),
            "med"  : statistics.median(data),
            "std"  : statistics.stdev(data),
            "min"  : min(data),
            "maks" : max(data),
            "p95"  : sorted(data)[int(len(data) * 0.95)],
        }

    cap_s   = stats(cap_times)
    proc_s  = stats(proc_times)
    total_s = stats(total_times)
    fps     = 1000.0 / total_s["ort"]

    print(f"\n{'='*55}")
    print(f"  Raspberry Pi 5 Performans Ölçümü")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Çözünürlük: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f"  Ölçülen kare: {n_frames}")
    print(f"{'='*55}")
    print(f"  {'Aşama':<22} {'Ort':>7} {'Med':>7} {'Std':>7} {'P95':>7}")
    print(f"  {'-'*50}")
    print(f"  {'Kare okuma (ms)':<22} "
          f"{cap_s['ort']:>7.2f} {cap_s['med']:>7.2f} "
          f"{cap_s['std']:>7.2f} {cap_s['p95']:>7.2f}")
    print(f"  {'Algoritma (ms)':<22} "
          f"{proc_s['ort']:>7.2f} {proc_s['med']:>7.2f} "
          f"{proc_s['std']:>7.2f} {proc_s['p95']:>7.2f}")
    print(f"  {'Toplam (ms)':<22} "
          f"{total_s['ort']:>7.2f} {total_s['med']:>7.2f} "
          f"{total_s['std']:>7.2f} {total_s['p95']:>7.2f}")
    print(f"{'='*55}")
    print(f"  Ortalama FPS : {fps:.1f}")
    print(f"  Hedef (≥30)  : {'✓ KARŞILANDI' if fps >= 30 else '✗ KARŞILANMADI'}")
    print(f"  Hedef (<50ms): {'✓ KARŞILANDI' if total_s['ort'] < 50 else '✗ KARŞILANMADI'}")
    print(f"{'='*55}")

    # CSV kaydet
    out = f"results/perf_{os.path.splitext(os.path.basename(video_path))[0]}.csv"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        f.write("asama,ort_ms,med_ms,std_ms,min_ms,maks_ms,p95_ms\n")
        f.write(f"kare_okuma,{cap_s['ort']:.3f},{cap_s['med']:.3f},"
                f"{cap_s['std']:.3f},{cap_s['min']:.3f},"
                f"{cap_s['maks']:.3f},{cap_s['p95']:.3f}\n")
        f.write(f"algoritma,{proc_s['ort']:.3f},{proc_s['med']:.3f},"
                f"{proc_s['std']:.3f},{proc_s['min']:.3f},"
                f"{proc_s['maks']:.3f},{proc_s['p95']:.3f}\n")
        f.write(f"toplam,{total_s['ort']:.3f},{total_s['med']:.3f},"
                f"{total_s['std']:.3f},{total_s['min']:.3f},"
                f"{total_s['maks']:.3f},{total_s['p95']:.3f}\n")
        f.write(f"fps,{fps:.2f},,,,,\n")
    print(f"  Kaydedildi: {out}")

    return total_s, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    benchmark(args.video, args.frames)