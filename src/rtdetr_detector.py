import torch
import supervision as sv
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

class RTDetrDetector:
    def __init__(self, model_id="PekingU/rtdetr_r50vd_coco_o365", device=None, confidence_threshold=0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        
    @torch.no_grad()
    def infer(self, frame) -> sv.Detections:
        image = Image.fromarray(frame[:, :, ::-1])

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
        )[0]

        transformers_results = {
            "scores": results["scores"],
            "labels": results["labels"],
            "boxes": results["boxes"],
        }

        detections = sv.Detections.from_transformers(
            transformers_results=transformers_results,
            id2label=self.id2label,
        )
        return detections
    
    def infer_batch(self, frames: list) -> list[sv.Detections]:
        return [self.infer(frame) for frame in frames]