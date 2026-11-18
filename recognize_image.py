"""
Detect and Recognize Faces in Static Images.
Defaults to CPU execution.
"""
import argparse
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DEVICE, DEFAULT_CONFIDENCE, DEFAULT_SIMILARITY_THRESHOLD
from src.pipeline import FaceRecognitionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and recognize faces in static images using YOLO-Face and FaceNet."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image to analyze."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the output annotated image."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Inference device: 'cpu' (default) or 'cuda'."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD})."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Detection confidence threshold (default: {DEFAULT_CONFIDENCE})."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image in a window."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        return

    frame_bgr = cv2.imread(str(img_path))
    if frame_bgr is None:
        print(f"Error: Could not read image: {img_path}")
        return

    pipeline = FaceRecognitionPipeline(device=args.device)
    annotated, detections = pipeline.process_frame(
        frame_bgr,
        threshold=args.threshold,
        conf=args.conf,
        draw=True
    )

    print(f"\n--- Detection & Recognition Results for '{img_path.name}' ---")
    print(f"Device: {args.device.upper()} | Faces Detected: {len(detections)}")
    for i, det in enumerate(detections, 1):
        status = "RECOGNIZED" if det["is_known"] else "UNKNOWN"
        print(
            f"  Face #{i}: [{status}] Name='{det['name']}', "
            f"Similarity={det['similarity']:.2f}, DetectionConf={det['det_confidence']:.2f}, "
            f"Box={det['bbox']}"
        )

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        print(f"\nAnnotated image saved to: {save_path}")

    if args.show:
        cv2.imshow(f"Results - {img_path.name}", annotated)
        print("Press any key in the window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
