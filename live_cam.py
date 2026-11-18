"""
Real-time Camera Face Detection and Recognition.
Defaults to CPU execution with interactive OpenCV display.
"""
import argparse
import time
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DEVICE, DEFAULT_CONFIDENCE, DEFAULT_SIMILARITY_THRESHOLD
from src.pipeline import FaceRecognitionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-time camera face recognition with YOLO-Face and FaceNet."
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="0",
        help="Camera device index (e.g. 0, 1) or video stream path/URL (default: 0)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to run inference on: 'cpu' (default) or 'cuda'."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for identity match (default: {DEFAULT_SIMILARITY_THRESHOLD})."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"YOLO-Face detection confidence threshold (default: {DEFAULT_CONFIDENCE})."
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Run full recognition every N frames for maximum CPU throughput (default: 1)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera capture width (default: 640)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera capture height (default: 480)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse camera index or stream
    cam_source = int(args.cam) if args.cam.isdigit() else args.cam
    print(f"\n[LiveCam] Initializing video capture on source: {cam_source}...")
    cap = cv2.VideoCapture(cam_source)
    
    if not cap.isOpened():
        print(f"[LiveCam] Error: Could not open camera {cam_source}.")
        print("Tip: If on Windows, check if another application is using the camera or try --cam 1.")
        return

    # Set requested resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize Face Recognition Pipeline
    print(f"[LiveCam] Setting up Face Recognition on device: '{args.device}'...")
    pipeline = FaceRecognitionPipeline(device=args.device)

    print("\n" + "=" * 55)
    print(" LIVE FACE RECOGNITION RUNNING")
    print(f" Device: {args.device.upper()} | Similarity Threshold: {args.threshold}")
    print(f" Known Enrolled Identities: {len(pipeline.database.identities)}")
    print(" Controls:")
    print("   [q] / [ESC] : Quit")
    print("   [s]         : Snapshot & quick enroll face from camera")
    print("=" * 55 + "\n")

    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    last_detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[LiveCam] Failed to grab frame. Exiting stream.")
                break

            frame_count += 1
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
            prev_time = curr_time

            # Full recognition on specified interval
            if frame_count % args.skip_frames == 0:
                annotated_frame, last_detections = pipeline.process_frame(
                    frame,
                    threshold=args.threshold,
                    conf=args.conf,
                    draw=True
                )
            else:
                annotated_frame = frame.copy()
                # Redraw previous cached detections for smooth display
                for det in last_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    is_known = det["is_known"]
                    color = (0, 200, 0) if is_known else (0, 140, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{det['name']} ({det['similarity']:.2f})" if is_known else "Unknown"
                    cv2.putText(annotated_frame, label, (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Top overlay bar with status & FPS
            overlay_h = 32
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], overlay_h), (25, 25, 25), -1)
            status_text = (
                f"Device: {args.device.upper()} | "
                f"FPS: {fps:4.1f} | "
                f"Faces: {len(last_detections)} | "
                f"Enrolled: {len(pipeline.database.identities)}"
            )
            cv2.putText(
                annotated_frame,
                status_text,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("Face Recognition (CPU First) - Press 'q' to Quit", annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('s'):
                # Snapshot registration
                if last_detections:
                    name_input = input("\nEnter name for the detected face: ").strip()
                    if name_input:
                        res = pipeline.enroll_image(name_input, frame, conf=args.conf)
                        if res["success"]:
                            print(f"[LiveCam] Successfully enrolled '{name_input}'!")
                        else:
                            print(f"[LiveCam] Enrollment failed: {res.get('error')}")
                else:
                    print("\n[LiveCam] No face detected in current frame to snapshot.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[LiveCam] Camera stopped and resources released.")


if __name__ == "__main__":
    main()
