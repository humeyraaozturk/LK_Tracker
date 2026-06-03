# =============================================================
# eval_custom.py — Kendi Video + CSV GT ile Ablasyon Testi
# =============================================================
# Kullanım:
#   python eval_custom.py --video video.mp4 --gt video_gt.csv
#   python eval_custom.py --video video.mp4 --gt video_gt.csv --show
#   python eval_custom.py --video video.mp4 --gt video_gt.csv --config K5_Tam_Sistem --show
#
# Çıktı:
#   results/custom_<video_adı>.csv  — her konfigürasyon için metrikler
# =============================================================

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.tracker import PointTracker


# ── Ablasyon konfigürasyonları ────────────────────────────────
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


# ── GT yükleme ────────────────────────────────────────────────

def load_gt(csv_path: str, scale_x: float = 1.0, scale_y: float = 1.0) -> dict:
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi     = int(row["frame"])
            x      = float(row["x"]) * scale_x
            y      = float(row["y"]) * scale_y
            manual = row.get("manual", "False").strip().lower() == "true"
            gt[fi] = (x, y, manual)
    return gt


# ── Video yükleme ─────────────────────────────────────────────

def load_video(video_path: str) -> list:
    cap    = cv2.VideoCapture(video_path)
    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {total} kare yükleniyor...", end="", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)
    cap.release()
    print(f" {len(frames)} kare.")
    return frames, orig_w, orig_h


# Yeniden tespit onay eşiği (piksel)
# Önerilen konum GT'ye bu mesafeden yakınsa onayla, uzaksa reddet
REID_ACCEPT_THRESHOLD = 30.0


# ── Tek konfigürasyon değerlendirmesi ────────────────────────

def run_evaluation(frames: list,
                   gt: dict,
                   layers: dict,
                   show: bool = False,
                   win_name: str = "eval") -> dict:
    """
    Verilen konfigürasyonla tüm video boyunca takip çalıştırır.

    MAE: yalnızca takip edilen karelerde hesaplanır.
    Kayıp kare: state == lost veya pending olan kareler.

    Yeniden tespit onayı:
    - GT bu karede mevcutsa: önerilen konum GT'ye REID_ACCEPT_THRESHOLD
      piksel içindeyse onayla, değilse reddet (kullanıcı onayını simüle eder)
    - GT bu karede yoksa: otomatik onayla
    """
    if not gt:
        return {}

    sorted_frames = sorted(gt.keys())
    start_frame   = sorted_frames[0]
    sx, sy, _     = gt[start_frame]

    tracker = PointTracker(layers=layers)

    first_bgr  = frames[start_frame]
    first_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    tracker.add_point(int(sx), int(sy), first_gray, frame_bgr=first_bgr)

    tracking_errors  = []
    n_lost           = 0
    n_tracked        = 0
    n_reid_accepted  = 0   # GT tabanlı onaylanan yeniden tespit sayısı
    n_reid_rejected  = 0   # GT tabanlı reddedilen yeniden tespit sayısı
    per_frame        = {}

    for fi in range(start_frame + 1, len(frames)):
        frame_bgr = frames[fi]
        states    = tracker.update(frame_bgr)

        # ── Pending: GT mesafesine göre onayla veya reddet ───
        pending_pts = [p for p in tracker.points
                       if p.state == "pending"]
        for pt in pending_pts:
            if pt.pending_pos is None:
                tracker.confirm_pending(pt.id)
                continue
            px = float(pt.pending_pos[0])
            py = float(pt.pending_pos[1])
            if fi in gt:
                gt_x, gt_y, _ = gt[fi]
                dist = float(np.sqrt((px - gt_x)**2 + (py - gt_y)**2))
                if dist <= REID_ACCEPT_THRESHOLD:
                    tracker.confirm_pending(pt.id)
                    n_reid_accepted += 1
                else:
                    tracker.reject_pending(pt.id)
                    n_reid_rejected += 1
            else:
                # GT yok → otomatik onayla
                tracker.confirm_pending(pt.id)
                n_reid_accepted += 1

        states = tracker._get_states()

        if not states:
            # Nokta tamamen kaldırıldı (K3 timeout)
            n_lost += 1
            per_frame[fi] = {"state": "removed", "error": None,
                             "pred": None}
            continue

        state    = states[0]
        pred_x   = float(state["pos"][0])
        pred_y   = float(state["pos"][1])
        pt_state = state["state"]

        # Bu karede GT var mı?
        if fi in gt:
            gt_x, gt_y, _ = gt[fi]
            error = float(np.sqrt((pred_x - gt_x)**2 +
                                  (pred_y - gt_y)**2))
        else:
            error = None

        if pt_state in ("tracking", "drifting"):
            n_tracked += 1
            if error is not None:
                tracking_errors.append(error)
        else:
            n_lost += 1

        per_frame[fi] = {
            "state" : pt_state,
            "error" : error,
            "pred"  : (pred_x, pred_y),
        }

        # Görsel gösterim
        if show:
            vis = frame_bgr.copy()

            # GT noktası — yeşil
            if fi in gt:
                gx, gy, _ = gt[fi]
                cv2.circle(vis, (int(gx), int(gy)), 8,
                           (0, 220, 100), 2, cv2.LINE_AA)
                cv2.putText(vis, "GT", (int(gx)+10, int(gy)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 220, 100), 1)

            # Tahmin — kırmızı/yeşil
            col = (0, 200, 80) if pt_state == "tracking" else \
                  (0, 165, 255) if pt_state == "drifting" else \
                  (60, 60, 230)
            cv2.circle(vis, (int(pred_x), int(pred_y)), 6,
                       col, -1, cv2.LINE_AA)

            err_str = f"{error:.1f}px" if error is not None else "GT yok"
            cv2.putText(vis,
                        f"t={fi}  {pt_state}  hata={err_str}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
            cv2.imshow(win_name, vis)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                show = False

    if show:
        cv2.destroyWindow(win_name)

    gt_frames   = len([fi for fi in per_frame if fi in gt])
    mae         = float(np.mean(tracking_errors)) if tracking_errors else 0.0
    median_err  = float(np.median(tracking_errors)) if tracking_errors else 0.0
    max_err     = float(np.max(tracking_errors))    if tracking_errors else 0.0
    total_eval  = n_tracked + n_lost

    return {
        "mae"           : round(mae,        3),
        "median_err"    : round(median_err, 3),
        "max_err"       : round(max_err,    3),
        "n_tracked"     : n_tracked,
        "n_lost"        : n_lost,
        "track_ratio"   : round(n_tracked / total_eval, 3) if total_eval > 0
                          else 0.0,
        "n_gt_frames"   : gt_frames,
        "n_errors"      : len(tracking_errors),
        "n_reid_accepted": n_reid_accepted,
        "n_reid_rejected": n_reid_rejected,
        "per_frame"     : per_frame,
    }


# ── Ana fonksiyon ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True,
                        help="Video dosyası")
    parser.add_argument("--gt",     required=True,
                        help="GT CSV dosyası")
    parser.add_argument("--config", default=None,
                        help="Tek konfigürasyon (örn: K5_Tam_Sistem)")
    parser.add_argument("--show",   action="store_true",
                        help="Görsel gösterim")
    parser.add_argument("--out_dir", default="results",
                        help="Çıktı klasörü")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frames, orig_w, orig_h = load_video(args.video)
    scale_x = 640 / orig_w
    scale_y = 480 / orig_h
    gt = load_gt(args.gt, scale_x=scale_x, scale_y=scale_y)

    manual_c = sum(1 for _, _, m in gt.values() if m)
    print(f"[GT] {len(gt)} kare  "
          f"(manuel: {manual_c}  otomatik: {len(gt)-manual_c})")

    configs = ({args.config: CONFIGS[args.config]}
               if args.config else CONFIGS)

    video_name = os.path.splitext(os.path.basename(args.gt))[0]

    print(f"\n{'='*62}")
    print(f"Video: {video_name}  |  {len(frames)} kare  "
          f"|  GT: {len(gt)} nokta")
    print(f"{'='*62}")

    summary_rows = []

    for cfg_name, layers in configs.items():
        print(f"\n  [{cfg_name}]")

        result = run_evaluation(
            frames   = frames,
            gt       = gt,
            layers   = layers,
            show     = args.show,
            win_name = cfg_name,
        )

        print(f"    MAE          : {result['mae']:.2f} px")
        print(f"    Medyan hata  : {result['median_err']:.2f} px")
        print(f"    Maks hata    : {result['max_err']:.2f} px")
        print(f"    Takip oranı  : {result['track_ratio']*100:.1f}%  "
              f"({result['n_tracked']} / "
              f"{result['n_tracked']+result['n_lost']} kare)")
        print(f"    Kayıp kare   : {result['n_lost']}")
        if result['n_reid_accepted'] + result['n_reid_rejected'] > 0:
            print(f"    ReID onay    : {result['n_reid_accepted']} kabul  "
                  f"{result['n_reid_rejected']} red")

        summary_rows.append({
            "config"          : cfg_name,
            "mae"             : result["mae"],
            "median_err"      : result["median_err"],
            "max_err"         : result["max_err"],
            "track_ratio"     : result["track_ratio"],
            "n_tracked"       : result["n_tracked"],
            "n_lost"          : result["n_lost"],
            "n_gt_frames"     : result["n_gt_frames"],
            "n_reid_accepted" : result["n_reid_accepted"],
            "n_reid_rejected" : result["n_reid_rejected"],
        })

    # Özet tablosu
    print(f"\n{'='*62}")
    print(f"{'Konfigürasyon':<30} {'MAE':>8} {'Medyan':>8} "
          f"{'Takip%':>8} {'Kayıp':>7}")
    print(f"{'-'*62}")
    for r in summary_rows:
        print(f"  {r['config']:<28} "
              f"{r['mae']:>7.2f}px "
              f"{r['median_err']:>7.2f}px "
              f"{r['track_ratio']*100:>7.1f}% "
              f"{r['n_lost']:>6}")
    print(f"{'='*62}")

    # CSV kaydet
    out_path   = os.path.join(args.out_dir, f"custom_{video_name}.csv")
    fieldnames = ["config", "mae", "median_err", "max_err",
                  "track_ratio", "n_tracked", "n_lost", "n_gt_frames",
                  "n_reid_accepted", "n_reid_rejected"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[Kaydedildi] {out_path}")


if __name__ == "__main__":
    main()