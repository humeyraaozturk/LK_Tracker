# =============================================================
# eval_tapvid.py — TAP-Vid-DAVIS ile ablasyon değerlendirmesi
# =============================================================
# Kullanım:
#   python eval_tapvid.py --pkl tapvid_davis.pkl
#   python eval_tapvid.py --pkl tapvid_davis.pkl --video car-roundabout
#   python eval_tapvid.py --pkl tapvid_davis.pkl --video goat --point_idx 0
#   python eval_tapvid.py --pkl tapvid_davis.pkl --all   (tüm konfigürasyonlar)
#
# Çıktı:
#   results/eval_<video>_<config>.csv
#   results/summary.csv
# =============================================================

import argparse
import os
import pickle
import csv
from datetime import datetime

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.tracker import PointTracker


# ── Ablasyon konfigürasyonları ────────────────────────────────
# Her konfigürasyon bir öncekinin üzerine bir katman ekler
CONFIGS = {
    "K0_LK_only": {
        "fb_error"    : False,
        "confidence"  : False,
        "adaptive"    : False,
        "drift"       : False,
        "redetection" : False,
    },
    "K1_LK_FB": {
        "fb_error"    : True,
        "confidence"  : False,
        "adaptive"    : False,
        "drift"       : False,
        "redetection" : False,
    },
    "K2_LK_FB_Conf": {
        "fb_error"    : True,
        "confidence"  : True,
        "adaptive"    : False,
        "drift"       : False,
        "redetection" : False,
    },
    "K3_LK_FB_Conf_Adapt": {
        "fb_error"    : True,
        "confidence"  : True,
        "adaptive"    : True,
        "drift"       : False,
        "redetection" : False,
    },
    "K4_LK_FB_Conf_Adapt_Drift": {
        "fb_error"    : True,
        "confidence"  : True,
        "adaptive"    : True,
        "drift"       : True,
        "redetection" : False,
    },
    "K5_Tam_Sistem": {
        "fb_error"    : True,
        "confidence"  : True,
        "adaptive"    : True,
        "drift"       : True,
        "redetection" : True,
    },
}


def load_davis(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_pixel_coords(points: np.ndarray, H: int, W: int) -> np.ndarray:
    """Normalize koordinatları piksel koordinatına çevirir."""
    px = points[:, :, 0] * W
    py = points[:, :, 1] * H
    return np.stack([px, py], axis=-1)  # (N_nokta, N_kare, 2)


def run_evaluation(video_frames: np.ndarray,
                   gt_coords: np.ndarray,
                   gt_occluded: np.ndarray,
                   point_idx: int,
                   layers: dict,
                   show: bool = False) -> dict:
    """
    Tek bir nokta ve konfigürasyon için değerlendirme çalıştırır.

    MAE hesabı:
    - Yalnızca NOKTAnın takip edildiği (tracking/drifting) karelerde hesaplanır
    - Ground truth'un tıkalı olmadığı karelerde hesaplanır
    - Kayıp karelerdeki hata MAE'ye dahil edilmez
    """
    N_frames = video_frames.shape[0]
    H, W     = video_frames.shape[1], video_frames.shape[2]

    # Başlangıç noktasını al (normalize → piksel, yuvarlama yapma)
    start_x = float(gt_coords[point_idx, 0, 0])
    start_y = float(gt_coords[point_idx, 0, 1])

    tracker = PointTracker(layers=layers)

    first_frame = video_frames[0]
    first_gray  = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    tracker.add_point(
        round(start_x), round(start_y),
        first_gray, frame_bgr=cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    )

    tracking_errors  = []   # takip edilirken hata
    n_lost_frames    = 0    # kayıp kare sayısı
    n_tracked_frames = 0    # takip edilen kare sayısı
    n_occluded_gt    = 0    # ground truth tıkalı kare sayısı

    for t in range(1, N_frames):
        frame_rgb = video_frames[t]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        states = tracker.update(frame_bgr)

        # Pending durumundaki noktaları otomatik onayla
        # (eval ortamında kullanıcı onayı yok)
        tracker.confirm_all_pending()

        is_occluded = bool(gt_occluded[point_idx, t])
        if is_occluded:
            n_occluded_gt += 1
            continue   # tıkalı kareleri atla

        gt_x = float(gt_coords[point_idx, t, 0])
        gt_y = float(gt_coords[point_idx, t, 1])

        if not states:
            n_lost_frames += 1
            continue

        state    = states[0]
        pred_x   = float(state["pos"][0])
        pred_y   = float(state["pos"][1])
        pt_state = state["state"]

        if pt_state in ("tracking", "drifting"):
            n_tracked_frames += 1
            err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            tracking_errors.append(err)
        else:
            n_lost_frames += 1

        if show:
            vis = frame_bgr.copy()
            cv2.circle(vis, (int(gt_x), int(gt_y)), 6, (0, 255, 0), 2)
            cv2.circle(vis, (int(pred_x), int(pred_y)), 5, (0, 0, 255), -1)
            err_str = f"{tracking_errors[-1]:.1f}px" if pt_state in ("tracking","drifting") else "KAYIP"
            cv2.putText(vis, f"t={t}  {err_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("eval", vis)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    if show:
        cv2.destroyAllWindows()

    visible_frames  = N_frames - 1 - n_occluded_gt
    mae             = float(np.mean(tracking_errors)) if tracking_errors else 0.0
    track_ratio     = n_tracked_frames / visible_frames if visible_frames > 0 else 0.0

    return {
        "mae"            : round(mae, 3),
        "n_tracked"      : n_tracked_frames,
        "n_lost"         : n_lost_frames,
        "n_occluded_gt"  : n_occluded_gt,
        "track_ratio"    : round(track_ratio, 3),
        "n_visible"      : visible_frames,
        "n_frames"       : N_frames - 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",       required=True,
                        help="tapvid_davis.pkl dosya yolu")
    parser.add_argument("--video",     default=None,
                        help="Test edilecek video adı (örn: goat)")
    parser.add_argument("--point_idx", type=int, default=0,
                        help="Hangi ground truth noktası kullanılacak")
    parser.add_argument("--config",    default=None,
                        help="Tek konfigürasyon adı (örn: K5_Tam_Sistem)")
    parser.add_argument("--all",       action="store_true",
                        help="Tüm video ve konfigürasyonları çalıştır")
    parser.add_argument("--show",      action="store_true",
                        help="Görsel gösterim")
    parser.add_argument("--out_dir",   default="results",
                        help="Çıktı klasörü")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_davis(args.pkl)
    print(f"[Yüklendi] {len(data)} video")

    # Video seçimi
    if args.all:
        video_names = list(data.keys())
    elif args.video:
        if args.video not in data:
            print(f"[HATA] '{args.video}' bulunamadı. "
                  f"Mevcut: {list(data.keys())}")
            return
        video_names = [args.video]
    else:
        video_names = [list(data.keys())[0]]
        print(f"[Bilgi] Video belirtilmedi, ilk video kullanılıyor: "
              f"{video_names[0]}")

    # Konfigürasyon seçimi
    if args.config:
        configs = {args.config: CONFIGS[args.config]}
    else:
        configs = CONFIGS

    summary_rows = []

    for vname in video_names:
        sample    = data[vname]
        frames    = sample["video"]      # (N, H, W, 3) RGB
        points_n  = sample["points"]     # (P, N, 2) normalize
        occluded  = sample["occluded"]   # (P, N) bool

        H, W = frames.shape[1], frames.shape[2]

        # Piksel koordinatlarına çevir
        coords_px = get_pixel_coords(points_n, H, W)  # (P, N, 2)

        n_points = coords_px.shape[0]
        pidx     = min(args.point_idx, n_points - 1)

        print(f"\n{'='*60}")
        print(f"Video: {vname}  |  {frames.shape[0]} kare  "
              f"{W}x{H}  |  {n_points} nokta  |  nokta idx={pidx}")
        print(f"{'='*60}")

        for cfg_name, layers in configs.items():
            point_maes      = []
            point_lost      = []
            point_tracked   = []
            point_ratio     = []

            n_points_to_test = coords_px.shape[0]

            for pidx in range(n_points_to_test):
                result = run_evaluation(
                    video_frames = frames,
                    gt_coords    = coords_px,
                    gt_occluded  = occluded,
                    point_idx    = pidx,
                    layers       = layers,
                    show         = args.show and pidx == 0,
                )
                if result["n_visible"] > 0:
                    point_maes.append(result["mae"])
                    point_lost.append(result["n_lost"])
                    point_tracked.append(result["n_tracked"])
                    point_ratio.append(result["track_ratio"])

            avg_mae   = float(np.mean(point_maes))   if point_maes   else 0.0
            avg_lost  = float(np.mean(point_lost))   if point_lost   else 0.0
            avg_ratio = float(np.mean(point_ratio))  if point_ratio  else 0.0

            print(f"  [{cfg_name}]  "
                  f"MAE={avg_mae:.2f}px  "
                  f"Kayıp={avg_lost:.1f}  "
                  f"Takip={avg_ratio*100:.1f}%  "
                  f"({n_points_to_test} nokta)")

            summary_rows.append({
                "video"       : vname,
                "config"      : cfg_name,
                "n_points"    : n_points_to_test,
                "mae"         : round(avg_mae, 3),
                "n_lost"      : round(avg_lost, 1),
                "track_ratio" : round(avg_ratio, 3),
            })

    # Özet CSV kaydet
    summary_path = os.path.join(args.out_dir, "summary.csv")
    fieldnames   = ["video", "config", "n_points",
                    "mae", "n_lost", "track_ratio"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[Kaydedildi] {summary_path}")

    # Konfigürasyon bazlı ortalama
    if len(video_names) > 1:
        print(f"\n{'='*60}")
        print("Tüm videolar üzerinde konfigürasyon ortalamaları:")
        print(f"{'='*60}")
        for cfg_name in configs:
            rows = [r for r in summary_rows if r["config"] == cfg_name]
            avg_mae    = np.mean([r["mae"] for r in rows])
            avg_lost   = np.mean([r["n_lost"] for r in rows])
            avg_track  = np.mean([r["track_ratio"] for r in rows])
            print(f"  {cfg_name:35s}  "
                  f"MAE={avg_mae:.2f}px  "
                  f"Kayıp={avg_lost:.1f}  "
                  f"Takip={avg_track*100:.1f}%")


if __name__ == "__main__":
    main()