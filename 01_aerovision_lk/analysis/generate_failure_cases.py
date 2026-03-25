"""
Generate 5 SAHI failure case images with GT vs prediction overlays.
Identifies failures by category: density, bicycle, low-contrast, truck/van, people/pedestrian.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time

# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "yolov8s_baseline.pt"
VAL_IMAGES = ROOT / "data" / "VisDrone_Dataset" / "VisDrone2019-DET-val" / "images"
VAL_LABELS = ROOT / "data" / "VisDrone_Dataset" / "VisDrone2019-DET-val" / "labels"
OUT_DIR = ROOT / "analysis" / "failure_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "three_wheeler", "bus", "motor",
]
BICYCLE_ID = 2
TRUCK_ID = 5
VAN_ID = 4
PEDESTRIAN_ID = 0
PEOPLE_ID = 1

# SAHI config (best from grid search)
SLICE_SIZE = 512
OVERLAP = 0.1
CONF = 0.25


def load_gt(label_path, img_w, img_h):
    """Load YOLO-format GT labels -> list of (class_id, x1, y1, x2, y2)."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cid = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((cid, x1, y1, x2, y2))
    return boxes


def iou(box_a, box_b):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def match_preds_to_gt(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Match predictions to GT. Returns:
    - matched: list of (gt_idx, pred_idx, iou_val)
    - missed_gt: list of gt indices with no match
    - false_pos: list of pred indices with no match
    """
    used_pred = set()
    matched = []
    for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
        best_iou, best_pi = 0, -1
        for pi, (pc, px1, py1, px2, py2, pconf) in enumerate(pred_boxes):
            if pi in used_pred:
                continue
            v = iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            if v > best_iou:
                best_iou, best_pi = v, pi
        if best_iou >= iou_thresh and best_pi >= 0:
            matched.append((gi, best_pi, best_iou))
            used_pred.add(best_pi)

    missed_gt = [i for i in range(len(gt_boxes)) if i not in {m[0] for m in matched}]
    false_pos = [i for i in range(len(pred_boxes)) if i not in used_pred]
    return matched, missed_gt, false_pos


def draw_failure(img_path, gt_boxes, pred_boxes, title):
    """Draw GT (green dashed) and predictions (red solid) on image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Draw GT boxes — green
    for cid, x1, y1, x2, y2 in gt_boxes:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        # Dashed effect via dotted line segments
        cv2.rectangle(img, pt1, pt2, (0, 200, 0), 1, cv2.LINE_AA)
        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        cv2.putText(img, f"GT:{label}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1, cv2.LINE_AA)

    # Draw predictions — red
    for cid, x1, y1, x2, y2, conf in pred_boxes:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        cv2.putText(img, f"{label} {conf:.2f}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    # Title bar
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 1, cv2.LINE_AA)

    return img


def main():
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    print("Loading SAHI model...")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(WEIGHTS),
        confidence_threshold=CONF,
        device="cuda:0",
    )

    img_paths = sorted(VAL_IMAGES.glob("*.jpg"))
    print(f"Running SAHI on {len(img_paths)} val images...")

    # ── Collect stats per image ──────────────────────────────────────────────
    stats = []
    t0 = time.time()

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT
        label_path = VAL_LABELS / (img_path.stem + ".txt")
        gt = load_gt(label_path, w, h)

        # SAHI prediction
        result = get_sliced_prediction(
            str(img_path), sahi_model,
            slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
            verbose=0,
        )

        preds = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            preds.append((
                obj.category.id,
                bbox.minx, bbox.miny, bbox.maxx, bbox.maxy,
                obj.score.value,
            ))

        # Match
        matched, missed_gt, false_pos = match_preds_to_gt(gt, preds)

        # Per-class missed counts
        missed_by_class = {}
        for gi in missed_gt:
            cid = gt[gi][0]
            missed_by_class[cid] = missed_by_class.get(cid, 0) + 1

        # Count class confusions in matched pairs
        confusions = {}
        for gi, pi, _ in matched:
            gc = gt[gi][0]
            pc = preds[pi][0]
            if gc != pc:
                key = (gc, pc)
                confusions[key] = confusions.get(key, 0) + 1

        # Average prediction confidence
        avg_conf = np.mean([p[5] for p in preds]) if preds else 0

        # Count GT by class
        gt_by_class = {}
        for cid, *_ in gt:
            gt_by_class[cid] = gt_by_class.get(cid, 0) + 1

        stats.append({
            "path": img_path,
            "gt_count": len(gt),
            "pred_count": len(preds),
            "missed_count": len(missed_gt),
            "miss_rate": len(missed_gt) / len(gt) if gt else 0,
            "avg_conf": avg_conf,
            "missed_by_class": missed_by_class,
            "confusions": confusions,
            "gt_by_class": gt_by_class,
            "gt": gt,
            "preds": preds,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{len(img_paths)}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"Done. {len(stats)} images in {elapsed:.1f}s")

    # ── Select 5 failure cases ───────────────────────────────────────────────
    cases = []

    # Case 1: Extreme density — most GT objects with high miss rate
    dense = [s for s in stats if s["gt_count"] >= 100]
    dense.sort(key=lambda s: s["missed_count"], reverse=True)
    if dense:
        s = dense[0]
        cases.append(("density", s, (
            f"With {s['gt_count']} ground-truth objects, this is one of the densest scenes in the "
            f"validation set; SAHI missed {s['missed_count']} ({s['miss_rate']:.0%}). "
            f"Objects near 512px tile boundaries are split across tiles and suppressed by NMS during "
            f"the merge step -- increasing overlap beyond 10% would recover some but adds latency."
        )))

    # Case 2: Bicycle miss — most missed bicycles
    bike_miss = [(s, s["missed_by_class"].get(BICYCLE_ID, 0)) for s in stats
                 if s["gt_by_class"].get(BICYCLE_ID, 0) >= 3]
    bike_miss.sort(key=lambda x: x[1], reverse=True)
    if bike_miss:
        s = bike_miss[0][0]
        n_gt_bike = s["gt_by_class"].get(BICYCLE_ID, 0)
        n_miss_bike = s["missed_by_class"].get(BICYCLE_ID, 0)
        cases.append(("bicycle", s, (
            f"This image contains {n_gt_bike} bicycle annotations but SAHI detected only "
            f"{n_gt_bike - n_miss_bike} -- missing {n_miss_bike}. "
            f"Bicycles have the fewest training samples (3.05% of dataset) and 67.8% are under 32px, "
            f"making them the hardest class even with tiled inference; data augmentation or "
            f"class-weighted loss could help."
        )))

    # Case 3: Low contrast — lowest average confidence
    low_conf = [s for s in stats if s["pred_count"] >= 10]
    low_conf.sort(key=lambda s: s["avg_conf"])
    if low_conf:
        s = low_conf[0]
        cases.append(("low_contrast", s, (
            f"The average prediction confidence is only {s['avg_conf']:.2f} across "
            f"{s['pred_count']} detections, indicating the model is uncertain about most objects. "
            f"Low-contrast conditions (dusk, shadows, overexposure) reduce feature discriminability "
            f"and push confidence below the 0.25 threshold -- histogram equalization or contrast-"
            f"adaptive preprocessing could mitigate this."
        )))

    # Case 4: Truck/van confusion
    tv_confuse = []
    for s in stats:
        tv = s["confusions"].get((TRUCK_ID, VAN_ID), 0) + s["confusions"].get((VAN_ID, TRUCK_ID), 0)
        if tv > 0:
            tv_confuse.append((s, tv))
    tv_confuse.sort(key=lambda x: x[1], reverse=True)
    if tv_confuse:
        s, n_conf = tv_confuse[0]
        cases.append(("truck_van", s, (
            f"SAHI confused truck and van classes {n_conf} times in this image. "
            f"From an aerial perspective, trucks and vans share similar rectangular silhouettes "
            f"and are primarily distinguished by physical size -- the model lacks reliable scale "
            f"cues at altitude, so a size-aware post-processing step or explicit aspect-ratio "
            f"features could reduce these swaps."
        )))

    # Case 5: People vs pedestrian confusion
    pp_confuse = []
    for s in stats:
        pp = (s["confusions"].get((PEDESTRIAN_ID, PEOPLE_ID), 0) +
              s["confusions"].get((PEOPLE_ID, PEDESTRIAN_ID), 0))
        if pp > 0:
            pp_confuse.append((s, pp))
    pp_confuse.sort(key=lambda x: x[1], reverse=True)
    if pp_confuse:
        s, n_conf = pp_confuse[0]
        cases.append(("people_pedestrian", s, (
            f"SAHI swapped people and pedestrian labels {n_conf} times here. "
            f"These two classes overlap semantically (standing vs sitting/grouped humans) and "
            f"96.8% of 'people' annotations are under 50px -- at this scale the visual difference "
            f"is negligible, so merging the two classes or using a pose-aware head would be "
            f"more practical for deployment."
        )))

    # ── Generate outputs ─────────────────────────────────────────────────────
    print(f"\nGenerating {len(cases)} failure case images...")
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("AEROVISION LK - SAHI FAILURE ANALYSIS")
    report_lines.append("5 cases where SAHI inference still fails")
    report_lines.append("=" * 70)
    report_lines.append("")

    for i, (category, s, explanation) in enumerate(cases, 1):
        fname = f"failure_{i}_{category}.png"
        title = f"Failure {i}: {category.replace('_', ' ').title()}"

        img = draw_failure(s["path"], s["gt"], s["preds"], title)
        if img is not None:
            out_path = OUT_DIR / fname
            cv2.imwrite(str(out_path), img)
            print(f"  Saved: {fname}")

        report_lines.append(f"--- Failure Case {i}: {category.replace('_', ' ').title()} ---")
        report_lines.append(f"Image: {s['path'].name}")
        report_lines.append(f"GT objects: {s['gt_count']}")
        report_lines.append(f"SAHI detected: {s['pred_count']}")
        report_lines.append(f"Missed: {s['missed_count']} ({s['miss_rate']:.0%})")
        report_lines.append(f"")
        report_lines.append(explanation)
        report_lines.append("")
        report_lines.append("")

    report_lines.append("=" * 70)
    report_lines.append("Green boxes = Ground truth  |  Red boxes = SAHI predictions")
    report_lines.append("=" * 70)

    report_path = OUT_DIR / "failure_analysis.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
