import argparse
import os
import numpy as np
import torch
import onnxruntime as nxrun
from PIL import Image
import cv2
import torchvision.transforms.functional as F
import supervision as sv

# Import utility functions from the project
from rfdetr.util.box_ops import box_cxcywh_to_xyxy

def verify_onnx_detection(model_path, image_path, output_dir, threshold=0.5):
    # 1. Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Load ONNX model
    # Note: RF-DETR exported outputs are usually [dets, labels]
    print(f"Loading ONNX model from {model_path}...")
    sess = nxrun.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # 3. Pre-processing
    # Must match RFDETRSmall logic exactly
    print(f"Processing image: {image_path}...")
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    # Default input size for RFDETRSmall is 512
    input_size = 512
    img = F.resize(image_pil, (input_size, input_size))
    img_tensor = F.to_tensor(img)
    # Standard ImageNet normalization
    img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_data = img_tensor.unsqueeze(0).numpy()

    # 4. Inference
    # Default input name is usually 'input'
    res = sess.run(None, {"input": input_data})

    # Based on export.py: res[0] is boxes (dets), res[1] is logits (labels)
    pred_boxes = torch.from_numpy(res[0])   # Shape: [1, 300, 4]
    pred_logits = torch.from_numpy(res[1])  # Shape: [1, 300, num_classes]

    # 5. Post-processing
    # Important: ONNX exports raw logits, must apply Sigmoid manually
    prob = pred_logits.sigmoid()
    num_classes = pred_logits.shape[2]

    # Get top 300 high-confidence results
    topk_values, topk_indexes = torch.topk(prob.view(1, -1), 300, dim=1)

    scores = topk_values[0]
    topk_boxes_idx = topk_indexes[0] // num_classes
    labels = topk_indexes[0] % num_classes

    # Convert cxcywh to xyxy (normalized [0, 1] coordinates)
    boxes = pred_boxes[0][topk_boxes_idx]
    boxes = box_cxcywh_to_xyxy(boxes)

    # Scale coordinates back to original image size
    scale_fct = torch.tensor([orig_w, orig_h, orig_w, orig_h])
    boxes = boxes * scale_fct

    # 6. Filter by threshold and encapsulate for Supervision
    mask = scores > threshold
    if mask.sum() == 0:
        print(f"No objects detected at threshold {threshold}. Max score: {scores.max():.4f}")
        return

    detections = sv.Detections(
        xyxy=boxes[mask].numpy(),
        confidence=scores[mask].numpy(),
        class_id=labels[mask].numpy().astype(int)
    )

    # 7. Visualization
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # Use raw class_id instead of category names
    labels_text = [
        f"ID: {class_id} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = box_annotator.annotate(
        scene=image_cv.copy(),
        detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels_text
    )

    # 8. Save result
    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"onnx_res_{image_name}")
    cv2.imwrite(save_path, annotated_image)
    print(f"Detected {len(detections)} object(s). Result saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR ONNX Inference and Visualization")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the ONNX model file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("--output_dir", type=str, default="results_onnx",
                        help="Directory to save the annotated image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")

    args = parser.parse_args()

    verify_onnx_detection(
        model_path=args.weights,
        image_path=args.image,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
