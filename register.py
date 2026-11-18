"""
CLI Tool for Enrolling Faces into the Recognition Database.
Defaults to CPU execution.
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DEVICE, DEFAULT_CONFIDENCE, KNOWN_FACES_DIR
from src.pipeline import FaceRecognitionPipeline
from src.database import FaceDatabase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enroll faces into the Face Recognition database using YOLO-Face and FaceNet."
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Full name or identifier of the person to enroll."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image file containing the person's face."
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        help="Directory containing images or subfolders named by person to batch-enroll."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device for inference: 'cpu' (default) or 'cuda'."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Face detection confidence threshold (default: 0.50)."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all enrolled identities currently in the database."
    )
    parser.add_argument(
        "--remove",
        type=str,
        help="Remove a specific identity from the database."
    )
    return parser.parse_args()


def batch_scan_directory(pipeline: FaceRecognitionPipeline, scan_path: Path, conf: float):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    enrolled_count = 0

    if not scan_path.exists():
        print(f"Directory not found: {scan_path}")
        return

    # Check for subdirectories (data/known_faces/<person_name>/<img.jpg>)
    subdirs = [p for p in scan_path.iterdir() if p.is_dir()]
    if subdirs:
        for sdir in subdirs:
            person_name = sdir.name
            img_files = [p for p in sdir.iterdir() if p.suffix.lower() in valid_exts]
            print(f"\nProcessing person '{person_name}' ({len(img_files)} images)...")
            for img_file in img_files:
                res = pipeline.enroll_image(person_name, img_file, conf=conf, save_db=False)
                if res["success"]:
                    print(f"  [+] Enrolled {img_file.name}")
                    enrolled_count += 1
                else:
                    print(f"  [-] Skipped {img_file.name}: {res.get('error')}")
    else:
        # Check files named <person_name>.jpg
        img_files = [p for p in scan_path.iterdir() if p.suffix.lower() in valid_exts]
        for img_file in img_files:
            person_name = img_file.stem
            res = pipeline.enroll_image(person_name, img_file, conf=conf, save_db=False)
            if res["success"]:
                print(f"  [+] Enrolled {person_name} from {img_file.name}")
                enrolled_count += 1
            else:
                print(f"  [-] Skipped {img_file.name}: {res.get('error')}")

    pipeline.database.save()
    print(f"\nBatch enrollment finished. Total images enrolled: {enrolled_count}")


def main():
    args = parse_args()

    # Fast management without loading heavy models
    if args.list:
        db = FaceDatabase()
        identities = db.list_identities()
        print("\n--- Enrolled Identities ---")
        if not identities:
            print("Database is currently empty.")
        else:
            for item in identities:
                print(f"• {item['name']}: {item['count']} sample(s) registered")
        print(f"Total: {len(identities)} person(s)\n")
        return

    if args.remove:
        db = FaceDatabase()
        if db.remove(args.remove):
            db.save()
            print(f"Successfully removed '{args.remove}'.")
        else:
            print(f"Identity '{args.remove}' not found in database.")
        return

    # Initialization for enrollment
    pipeline = FaceRecognitionPipeline(device=args.device)

    if args.scan_dir:
        batch_scan_directory(pipeline, Path(args.scan_dir), args.conf)
    elif args.image:
        if not args.name:
            print("Error: --name is required when enrolling with --image.")
            sys.exit(1)
        res = pipeline.enroll_image(args.name, args.image, conf=args.conf, save_db=True)
        if res["success"]:
            print(f"\n[SUCCESS] Successfully enrolled '{args.name}'!")
            print(f"Face Box: {res['bbox']}, Detection Confidence: {res['confidence']:.2f}")
            print(f"Total Enrolled Identities: {res['total_identities']}\n")
        else:
            print(f"\n[FAILED] Enrollment failed: {res.get('error')}\n")
            sys.exit(1)
    else:
        print("Please provide --image with --name, --scan-dir, --list, or --remove.")
        print("Run with -h for usage instructions.")


if __name__ == "__main__":
    main()
