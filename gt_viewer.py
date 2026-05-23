# =============================================================
# gt_viewer.py — GT Etiketlerini Video Üzerinde Görüntüle
# =============================================================
# Kullanım:
#   python gt_viewer.py --video video.mp4 --gt video_gt.csv
#
# Klavye:
#   Boşluk      → oynat / duraklat
#   N / →       → sonraki kare
#   P / ←       → önceki kare
#   Q / ESC     → çıkış
# =============================================================

import argparse
import csv
import os
import sys

import cv2
import numpy as np

FONT   = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (0,   220, 100)
YELLOW = (0,   215, 255)
WHITE  = (255, 255, 255)
DARK   = (20,   20,  20)
GRAY   = (100, 100, 100)


def load_gt(csv_path: str) -> dict:
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi     = int(row["frame"])
            x      = float(row["x"])
            y      = float(row["y"])
            manual = row.get("manual", "False").strip().lower() == "true"
            gt[fi] = (x, y, manual)
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt",    required=True)
    parser.add_argument("--trail", type=int, default=15,
                        help="Gösterilecek iz uzunluğu (kare)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {args.video}")
        sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gt = load_gt(args.gt)
    manual_count = sum(1 for _, _, m in gt.values() if m)
    print(f"[GT] {len(gt)} kare yüklendi  "
          f"(manuel: {manual_count}  otomatik: {len(gt)-manual_count})")

    # Tüm kareleri belleğe yükle
    print(f"[Video] {total} kare yükleniyor...")
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total):
        ret, frame = cap.read()
        frames.append(frame if ret else np.zeros(
            (height, width, 3), dtype=np.uint8))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}", end="\r")
    print(f"  {total} kare yüklendi.    ")
    cap.release()

    current = 0
    playing = False
    delay   = max(1, int(1000 / fps))

    WIN = "GT Viewer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(width, 1280), min(height + 60, 800))

    while True:
        frame = frames[current]
        vis   = frame.copy()

        # İz — son N kare
        trail = []
        for i in range(max(0, current - args.trail), current + 1):
            if i in gt:
                trail.append((i, gt[i]))

        for i in range(1, len(trail)):
            _, (x0, y0, m0) = trail[i - 1]
            _, (x1, y1, m1) = trail[i]
            alpha = i / len(trail)
            col   = GREEN if m1 else YELLOW
            c     = tuple(int(ch * alpha) for ch in col)
            cv2.line(vis,
                     (int(x0), int(y0)),
                     (int(x1), int(y1)), c, 2, cv2.LINE_AA)

        # Mevcut kare GT noktası
        if current in gt:
            x, y, manual = gt[current]
            col   = GREEN if manual else YELLOW
            label = "Manuel" if manual else "LK-otomatik"
            cv2.circle(vis, (int(x), int(y)), 10, col, 2, cv2.LINE_AA)
            cv2.circle(vis, (int(x), int(y)),  3, col, -1, cv2.LINE_AA)
            cv2.putText(vis, f"{label} ({int(x)},{int(y)})",
                        (int(x) + 12, int(y) - 8),
                        FONT, 0.42, col, 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "GT YOK", (width // 2 - 30, 70),
                        FONT, 0.7, (60, 60, 230), 2, cv2.LINE_AA)

        # İlerleme çubuğu
        bh   = height - 14
        bw   = width  - 20
        prog = int(bw * current / max(1, total - 1))
        cv2.rectangle(vis, (10, bh), (10 + bw, bh + 8), (35, 35, 35), -1)
        cv2.rectangle(vis, (10, bh), (10 + prog, bh + 8),
                      (100, 200, 100), -1)
        for fi in gt:
            bx  = 10 + int(bw * fi / max(1, total - 1))
            _, _, m = gt[fi]
            cv2.line(vis, (bx, bh - 1), (bx, bh + 9),
                     GREEN if m else YELLOW, 1)

        # Üst bilgi
        ov = vis.copy()
        cv2.rectangle(ov, (0, 0), (width, 42), DARK, -1)
        cv2.addWeighted(ov, 0.65, vis, 0.35, 0, vis)
        durum = "▶" if playing else "⏸"
        cv2.putText(vis,
                    f"{durum}  Kare {current}/{total-1}  |  "
                    f"GT: {len(gt)} kare  |  "
                    f"Yeşil=manuel  Sarı=otomatik  |  "
                    f"Boşluk: oynat  N/P: kare  Q: çıkış",
                    (8, 26), FONT, 0.38, WHITE, 1, cv2.LINE_AA)

        cv2.imshow(WIN, vis)

        wait = delay if playing else 20
        key  = cv2.waitKey(wait) & 0xFF

        if playing:
            if current < total - 1:
                current += 1
            else:
                playing = False

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            playing = not playing
        elif key in (ord('n'), 83):
            playing = False
            if current < total - 1:
                current += 1
        elif key in (ord('p'), 81):
            playing = False
            if current > 0:
                current -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()