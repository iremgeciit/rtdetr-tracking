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
def main():
    # ---- 1) AYARLAR (burayı sen değiştireceksin) ----
    sequence_dir = r"data\sportsmot\sportsmot\val\v_00HRwkvvjtQ_c001"
    output_video = r"results\v_00HRwkvvjtQ_c001_tracked.mp4"
    tracker_name = "bytetrack"
    confidence_threshold = 0.5
    target_classes = {"person"}
    fps = 25  # SportsMOT genelde 25fps, seqinfo.ini'de teyit edebilirsin

    # ---- 2) Kare dosyalarını bul ----
    frames_dir = os.path.join(sequence_dir, "img1")
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"{len(frame_paths)} kare bulundu: {frames_dir}")

    if len(frame_paths) == 0:
        raise RuntimeError("Kare bulunamadı, sequence_dir yolunu kontrol et.")

    # ---- 3) RT-DETR'i yükle ----
    print("RT-DETR yükleniyor...")
    detector = RTDetrDetector(confidence_threshold=confidence_threshold)

    # ---- 4) Tracker'ı hazırla ----
    tracker = TRACKER_REGISTRY[tracker_name]()

    # ---- 5) Çizim araçları ----
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    # ---- 6) Çıktı videosunu hazırla ----
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # ---- 7) Ana döngü: her kareyi sırayla işle ----
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)

        detections = detector.infer(frame)

        # Sadece istediğimiz sınıfı (person) tut
        if detections.data.get("class_name") is not None:
            mask = [name in target_classes for name in detections.data["class_name"]]
            detections = detections[mask]

        tracked = tracker.update(detections, frame=frame)

        labels = [f"#{tid}" for tid in tracked.tracker_id]

        annotated = frame.copy()
        annotated = trace_annotator.annotate(annotated, tracked)
        annotated = box_annotator.annotate(annotated, tracked)
        annotated = label_annotator.annotate(annotated, tracked, labels=labels)

        writer.write(annotated)

        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(frame_paths)} kare işlendi...")

    writer.release()
    print(f"Bitti! Çıktı: {output_video}")


if __name__ == "__main__":
    main()