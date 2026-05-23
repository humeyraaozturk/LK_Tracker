# =============================================================
# gt_annotator.py — Manuel Ground Truth İşaretleme Aracı
# =============================================================
# Kullanım:
#   python gt_annotator.py --video video.mp4
#   python gt_annotator.py --video video.mp4 --resume video_gt.csv
#
# Klavye kısayolları:
#   Sol tık        → bu karedeki GT koordinatını işaretle
#   N / Sağ ok     → sonraki kare
#   P / Sol ok     → önceki kare
#   Boşluk         → oynat/duraklat
#   Z              → bu karenin işaretini sil
#   S              → kaydet
#   Q / ESC        → kaydet ve çıkış
# =============================================================

import argparse
import csv
import os
import sys

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.tracker import PointTracker


FONT   = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (0,   220, 100)
RED    = (60,   60, 230)
YELLOW = (0,   215, 255)
WHITE  = (255, 255, 255)
GRAY   = (100, 100, 100)
DARK   = (20,   20,  20)


class GTAnnotator:
    def __init__(self, video_path: str, out_path: str):
        self.video_path = video_path
        self.out_path   = out_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"[HATA] Video açılamadı: {video_path}")
            sys.exit(1)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Video] {os.path.basename(video_path)}")
        print(f"  {self.total_frames} kare  |  "
              f"{self.fps:.1f} fps  |  {self.width}x{self.height}")

        # GT verisi: frame_idx → (x, y, manual)
        # manual=True  → kullanıcı tıkladı
        # manual=False → LK tahmininden otomatik
        self.gt: dict[int, tuple] = {}

        self.current_frame = 0
        self.playing       = False
        self.tracker       = None   # LK tracker (isteğe bağlı)
        self.tracker_pos   = None   # tracker'ın mevcut tahmini

        # Tüm kareleri belleğe yükle
        self._load_frames()

    # ── Kare yükleme ─────────────────────────────────────────

    def _load_frames(self):
        print(f"[Yükleniyor] {self.total_frames} kare...")
        self.frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.total_frames):
            ret, frame = self.cap.read()
            if ret:
                self.frames.append(frame)
            else:
                # Eksik kare için öncekini kopyala
                prev = self.frames[-1] if self.frames else \
                       np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.frames.append(prev.copy())
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{self.total_frames}", end="\r")
        print(f"  {len(self.frames)} kare yüklendi.    ")

    # ── LK Tracker başlat ────────────────────────────────────

    def _start_tracker(self, x: int, y: int):
        """İlk işaret konumundan LK takibini başlat."""
        frame = self.frames[self.current_frame]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracker = PointTracker()
        self.tracker.add_point(x, y, gray, frame_bgr=frame)
        self.tracker_pos = (x, y)
        print(f"\n[Tracker] ({x},{y}) konumundan LK başlatıldı.")

    def _update_tracker(self, to_frame: int):
        """Tracker'ı mevcut konumdan to_frame'e kadar çalıştır."""
        if self.tracker is None:
            return
        start = self.current_frame
        step  = 1 if to_frame > start else -1
        for fi in range(start + step, to_frame + step, step):
            if fi < 0 or fi >= len(self.frames):
                break
            frame = self.frames[fi]
            states = self.tracker.update(frame)
            if states:
                self.tracker_pos = (
                    int(states[0]["pos"][0]),
                    int(states[0]["pos"][1])
                )

    # ── CSV işlemleri ─────────────────────────────────────────

    def load_csv(self, csv_path: str):
        if not os.path.exists(csv_path):
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fi     = int(row["frame"])
                x      = float(row["x"])
                y      = float(row["y"])
                manual = row.get("manual", "False").strip().lower() == "true"
                self.gt[fi] = (x, y, manual)
        print(f"[Resume] {len(self.gt)} kare yüklendi: {csv_path}")
        if self.gt:
            self.current_frame = min(
                max(self.gt.keys()) + 1, self.total_frames - 1
            )

    def save_csv(self):
        with open(self.out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y", "manual"])
            for fi in sorted(self.gt.keys()):
                x, y, manual = self.gt[fi]
                writer.writerow([fi, f"{x:.2f}", f"{y:.2f}",
                                  "True" if manual else "False"])
        manual_count = sum(1 for _, _, m in self.gt.values() if m)
        auto_count   = len(self.gt) - manual_count
        print(f"\n[Kaydedildi] {len(self.gt)} kare → {self.out_path}")
        print(f"  Manuel: {manual_count}  |  Otomatik (LK): {auto_count}")

    # ── Çizim ────────────────────────────────────────────────

    def _draw(self) -> np.ndarray:
        frame = self.frames[self.current_frame]
        vis   = frame.copy()

        # Son 8 karedeki işaretleri iz olarak göster
        # Manuel → yeşil, otomatik (LK) → sarı
        trail_pts = []
        for i in range(max(0, self.current_frame - 8),
                       self.current_frame + 1):
            if i in self.gt:
                x, y, manual = self.gt[i]
                trail_pts.append((x, y, manual))

        for i in range(1, len(trail_pts)):
            alpha = i / len(trail_pts)
            col   = GREEN if trail_pts[i][2] else YELLOW
            c     = tuple(int(ch * alpha) for ch in col)
            p1    = (int(trail_pts[i-1][0]), int(trail_pts[i-1][1]))
            p2    = (int(trail_pts[i][0]),   int(trail_pts[i][1]))
            cv2.line(vis, p1, p2, c, 2, cv2.LINE_AA)

        # Tracker tahmini — mavi
        if self.tracker_pos is not None:
            tx, ty = self.tracker_pos
            cv2.circle(vis, (tx, ty), 8, (255, 100, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, "LK", (tx + 10, ty - 5),
                        FONT, 0.38, (255, 100, 0), 1, cv2.LINE_AA)

        # Mevcut karedeki GT
        if self.current_frame in self.gt:
            x, y, manual = self.gt[self.current_frame]
            col = GREEN if manual else YELLOW
            cv2.circle(vis, (int(x), int(y)), 9, col, 2, cv2.LINE_AA)
            cv2.circle(vis, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
            label = "manuel" if manual else "LK-otomatik"
            cv2.putText(vis, f"GT ({int(x)},{int(y)}) [{label}]",
                        (int(x) + 12, int(y) - 8),
                        FONT, 0.40, col, 1, cv2.LINE_AA)

        # İlerleme çubuğu
        bh   = self.height - 14
        bw   = self.width  - 20
        prog = int(bw * self.current_frame / max(1, self.total_frames - 1))
        cv2.rectangle(vis, (10, bh), (10 + bw, bh + 8), (35, 35, 35), -1)
        cv2.rectangle(vis, (10, bh), (10 + prog, bh + 8), GREEN, -1)
        for fi in self.gt:
            bx = 10 + int(bw * fi / max(1, self.total_frames - 1))
            cv2.line(vis, (bx, bh - 1), (bx, bh + 9), YELLOW, 1)

        # Üst bilgi şeridi
        ov = vis.copy()
        cv2.rectangle(ov, (0, 0), (self.width, 50), DARK, -1)
        cv2.addWeighted(ov, 0.65, vis, 0.35, 0, vis)

        durum = "▶ OYNATILIYOR" if self.playing else "⏸ DURAKLATILDI"
        d_col = YELLOW if self.playing else WHITE
        cv2.putText(vis,
                    f"Kare {self.current_frame}/{self.total_frames-1}  |  "
                    f"{self.fps:.0f}fps  |  "
                    f"İşaretli: {len(self.gt)}  |  {durum}",
                    (8, 18), FONT, 0.43, d_col, 1, cv2.LINE_AA)

        isaret = ("İŞARETLENDİ ✓" if self.current_frame in self.gt
                  else "işaretlenmedi")
        i_col  = GREEN if self.current_frame in self.gt else GRAY
        cv2.putText(vis,
                    "Sol tık: işaretle  N/P: kare  Boşluk: oynat  "
                    "Z: sil  S: kaydet  Q: çıkış    " + isaret,
                    (8, 38), FONT, 0.37, i_col, 1, cv2.LINE_AA)

        return vis

    # ── Mouse callback ────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.gt[self.current_frame] = (float(x), float(y), True)
            if self.tracker is None and self.current_frame == 0:
                self._start_tracker(x, y)
            print(f"  Kare {self.current_frame:4d} → ({x}, {y}) [manuel]",
                  end="\r")

    # ── Ana döngü ─────────────────────────────────────────────

    def run(self):
        WIN = "GT Annotator"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(self.width,  1280),
                             min(self.height + 60, 800))
        cv2.setMouseCallback(WIN, self._mouse_cb)

        print("\n── Kısayollar ───────────────────────────────")
        print("  Sol tık     → koordinat işaretle (0. karede başlarsa LK başlar)")
        print("  Boşluk      → oynat / duraklat")
        print("  N / →       → sonraki kare")
        print("  P / ←       → önceki kare")
        print("  Z           → bu karenin işaretini sil")
        print("  S           → kaydet")
        print("  Q / ESC     → kaydet ve çıkış")
        print("─────────────────────────────────────────────\n")

        delay = max(1, int(1000 / self.fps))

        while True:
            vis = self._draw()
            cv2.imshow(WIN, vis)

            wait = delay if self.playing else 20
            key  = cv2.waitKey(wait) & 0xFF

            if self.playing:
                if self.current_frame < self.total_frames - 1:
                    next_f = self.current_frame + 1
                    self._update_tracker(next_f)
                    self.current_frame = next_f

                    # LK tahmini otomatik GT olarak kaydet
                    if self.tracker_pos is not None:
                        tx, ty = self.tracker_pos
                        if self.current_frame not in self.gt:
                            self.gt[self.current_frame] = (
                                float(tx), float(ty), False
                            )
                else:
                    self.playing = False

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                self.playing = not self.playing
            elif key in (ord('n'), 83):   # N veya sağ ok
                self.playing = False
                if self.current_frame < self.total_frames - 1:
                    next_f = self.current_frame + 1
                    self._update_tracker(next_f)
                    self.current_frame = next_f
                    if self.tracker_pos is not None:
                        tx, ty = self.tracker_pos
                        if self.current_frame not in self.gt:
                            self.gt[self.current_frame] = (
                                float(tx), float(ty), False
                            )
            elif key in (ord('p'), 81):   # P veya sol ok
                self.playing = False
                if self.current_frame > 0:
                    self.current_frame -= 1
            elif key == ord('z'):
                if self.current_frame in self.gt:
                    del self.gt[self.current_frame]
                    print(f"  Kare {self.current_frame} silindi.", end="\r")
            elif key == ord('s'):
                self.save_csv()

        self.save_csv()
        cv2.destroyAllWindows()
        self.cap.release()

        if self.gt:
            print(f"\n── Özet ─────────────────────────────────────")
            print(f"  Toplam işaretlenen : {len(self.gt)} kare")
            print(f"  Kapsama            : "
                  f"{len(self.gt)/self.total_frames*100:.1f}%")
            print(f"  Çıktı              : {self.out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--out",    default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.out is None:
        base    = os.path.splitext(os.path.basename(args.video))[0]
        out_dir = os.path.dirname(args.video) or "."
        args.out = os.path.join(out_dir, f"{base}_gt.csv")

    ann = GTAnnotator(args.video, args.out)

    if args.resume:
        ann.load_csv(args.resume)
    elif os.path.exists(args.out):
        ans = input(f"  '{args.out}' mevcut. Devam? (e/h): ").strip().lower()
        if ans == "e":
            ann.load_csv(args.out)

    ann.run()

    if ann.gt:
        annotated    = len(ann.gt)
        manual_count = sum(1 for _, _, m in ann.gt.values() if m)
        print(f"\n── Özet ─────────────────────────────────────")
        print(f"  Toplam işaretlenen : {annotated} kare")
        print(f"  Manuel düzeltme    : {manual_count}")
        print(f"  Otomatik (LK)      : {annotated - manual_count}")
        print(f"  Kapsama            : "
              f"{annotated/ann.total_frames*100:.1f}%")
        print(f"  Çıktı              : {ann.out_path}")


if __name__ == "__main__":
    main()