import argparse
import time
import cv2

from utils import DATASET_DIR, ensure_dirs, create_face_app, get_largest_face, crop_face_with_margin


def collect_faces(name, samples, camera_index, det_size, delay):
    ensure_dirs()

    person_dir = DATASET_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    existing_images = (
        list(person_dir.glob("*.jpg")) +
        list(person_dir.glob("*.jpeg")) +
        list(person_dir.glob("*.png"))
    )

    count = len(existing_images)
    target_count = count + samples

    app = create_face_app(det_size=det_size, det_thresh=0.5)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("=" * 60)
    print(f"Collecting face images for: {name}")
    print(f"Already existing images: {count}")
    print(f"Target total images: {target_count}")
    print("Move your face: near, far, left, right, different lighting.")
    print("Press q to stop.")
    print("=" * 60)

    last_save_time = 0

    while count < target_count:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read from camera.")
            break

        faces = app.get(frame)
        face = get_largest_face(faces)

        if face is not None:
            now = time.time()

            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if now - last_save_time >= delay:
                crop = crop_face_with_margin(frame, face.bbox, margin_ratio=0.30)

                if crop.size > 0:
                    count += 1
                    image_path = person_dir / f"{count:04d}.jpg"
                    cv2.imwrite(str(image_path), crop)
                    last_save_time = now
                    print(f"Saved: {image_path}")

            cv2.putText(
                frame,
                f"Saved {count}/{target_count}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
        #cv2.namedWindow("Collect Faces - InsightFace", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Collect Faces - InsightFace", 1000, 700)
        cv2.imshow("Collect Faces - InsightFace", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("=" * 60)
    print(f"Done. Total images for {name}: {count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Person name, example: Keroles")
    parser.add_argument("--samples", type=int, default=80, help="Number of new samples to collect")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 or 1")
    parser.add_argument("--det-size", type=int, default=640, help="Detection size: 640, 960, or 1280")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between saved images in seconds")

    args = parser.parse_args()

    collect_faces(
        name=args.name,
        samples=args.samples,
        camera_index=args.camera,
        det_size=args.det_size,
        delay=args.delay
    )


if __name__ == "__main__":
    main()