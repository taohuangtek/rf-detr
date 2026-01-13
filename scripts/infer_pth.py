import argparse
import os
import cv2
import numpy as np
import torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall

def visualize_results(model_path, image_path, output_dir, threshold=0.5):
    # 1. Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Load the model
    print(f"Loading model from {model_path}...")
    model = RFDETRSmall(pretrain_weights=model_path, num_classes=80)

    # 3. Read and process image
    print(f"Processing image: {image_path}...")
    image_pil = Image.open(image_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # 4. Perform inference
    detections = model.predict(image_pil, threshold=threshold)

    if len(detections.xyxy) > 0:
        print(f"Detected {len(detections.xyxy)} object(s).")

        # 5. Initialize annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        # 6. Generate labels using raw class_id
        # Directly use class_id instead of category names
        labels = [
            f"ID: {class_id} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        # 7. Draw annotations
        annotated_image = box_annotator.annotate(
            scene=image_cv.copy(),
            detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        # 8. Save the result
        image_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"result_{image_name}")
        cv2.imwrite(save_path, annotated_image)
        print(f"Result saved to: {save_path}")
    else:
        print("No objects detected. Try lowering the threshold value.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR Inference and Visualization")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights (.pth file)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the annotated image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")

    args = parser.parse_args()

    visualize_results(
        model_path=args.weights,
        image_path=args.image,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
