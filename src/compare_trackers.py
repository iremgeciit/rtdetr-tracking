"""
Aynı sequence üzerinde birden fazla tracker'ı çalıştırıp karşılaştırma
için hem video hem MOT-format tahmin dosyası (.txt) üretir.
"""

import os
import glob
import cv2
import supervision as sv

from rtdetr_detector import RTDetrDetector
from trackers import ByteTrackTracker, BoTSORTTracker, SORTTracker, OCSORTTracker, CBIoUTracker

TRACKER_REGISTRY = {
    "sort": SORTTracker,
    "bytetrack": ByteTrackTracker,
    "ocsort": OCSORTTracker,
    "botsort": BoTSORTTracker,
    "cbiou": CBIoUTracker,
}


def run_one_tracker(tracker_name, frame_paths, detector, target_classes,
                     confidence_threshold, fps, sequence_name, output_root):
    tracker = TRACKER_REGISTRY[tracker_name]()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    video_path = os.path.join(output_root, "videos", f"{sequence_name}_{tracker_name}.mp4")
    txt_dir = os.path.join(output_root, "predictions", tracker_name)
    txt_path = os.path.join(txt_dir, f"{sequence_name}.txt")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    mot_lines = []

    for i, frame_path in enumerate(frame_paths):
        frame_number = i + 1  # MOT format 1'den başlar
        frame = cv2.imread(frame_path)

        detections = detector.infer(frame)

        if detections.data.get("class_name") is not None:
            mask = [name in target_classes for name in detections.data["class_name"]]
            detections = detections[mask]

        tracked = tracker.update(detections, frame=frame)

        # ---- MOT formatında satırları biriktir ----
        # format: frame,id,x,y,w,h,conf,-1,-1,-1
        for xyxy, tid, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            mot_lines.append(f"{frame_number},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")

        # ---- Görsel çıktı ----
        labels = [f"#{tid}" for tid in tracked.tracker_id]
        annotated = frame.copy()
        annotated = trace_annotator.annotate(annotated, tracked)
        annotated = box_annotator.annotate(annotated, tracked)
        annotated = label_annotator.annotate(annotated, tracked, labels=labels)
        writer.write(annotated)

        if (i + 1) % 50 == 0:
            print(f"    [{tracker_name}] {i + 1}/{len(frame_paths)} kare")

    writer.release()

    with open(txt_path, "w") as f:
        f.write("\n".join(mot_lines))

    print(f"  -> video: {video_path}")
    print(f"  -> tahmin dosyası: {txt_path}")


def main():
    # ---- AYARLAR ----
    sequence_dir = r"data\sportsmot\sportsmot\val\v_00HRwkvvjtQ_c001"
    sequence_name = "v_00HRwkvvjtQ_c001"
    output_root = "results"
    trackers_to_compare = ["sort", "bytetrack", "ocsort", "botsort", "cbiou"]
    confidence_threshold = 0.5
    target_classes = {"person"}
    fps = 25

    frames_dir = os.path.join(sequence_dir, "img1")
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"{len(frame_paths)} kare bulundu.\n")

    print("RT-DETR yükleniyor (bir kere, tüm tracker'lar için tekrar kullanılacak)...")
    detector = RTDetrDetector(confidence_threshold=confidence_threshold)

    for tracker_name in trackers_to_compare:
        print(f"\n=== {tracker_name} çalıştırılıyor ===")
        run_one_tracker(
            tracker_name, frame_paths, detector, target_classes,
            confidence_threshold, fps, sequence_name, output_root
        )

    print("\nTüm tracker'lar tamamlandı.")


if __name__ == "__main__":
    main()