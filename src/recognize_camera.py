import argparse
import time
import cv2
import numpy as np

from utils import DATABASE_PATH, create_face_app, l2_normalize


def load_database():
    if not DATABASE_PATH.exists():
        print("Error: face database not found.")
        print("Run this first:")
        print("python src/build_database.py")
        return None, None

    data = np.load(DATABASE_PATH, allow_pickle=True)
    names = data["names"]
    embeddings = data["embeddings"]

    return names, embeddings


def resize_frame(frame, width):
    if width <= 0:
        return frame

    h, w = frame.shape[:2]

    if w <= width:
        return frame

    ratio = width / w
    new_height = int(h * ratio)

    return cv2.resize(frame, (width, new_height))


def draw_box(frame, bbox, label, color):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


def recognize(
    camera_index,
    threshold,
    det_size,
    print_interval,
    process_every,
    camera_width,
    camera_height,
    display_width
):
    names, known_embeddings = load_database()

    if names is None:
        return

    app = create_face_app(det_size=det_size, det_thresh=0.5)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("=" * 60)
    print("InsightFace camera recognition started.")
    print(f"Threshold: {threshold}")
    print(f"Detection size: {det_size}")
    print(f"Process every: {process_every} frames")
    print("Press q to quit.")
    print("=" * 60)

    frame_counter = 0
    last_print_time = 0
    cached_results = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read from camera.")
            break

        frame = resize_frame(frame, display_width)

        should_process = frame_counter % process_every == 0

        if should_process:
            faces = app.get(frame)
            cached_results = []

            for face in faces:
                emb = l2_normalize(face.normed_embedding)

                similarities = np.dot(known_embeddings, emb)

                best_index = int(np.argmax(similarities))
                best_score = float(similarities[best_index])
                best_name = str(names[best_index])

                if best_score >= threshold:
                    label = f"{best_name} {best_score:.2f}"
                    color = (0, 255, 0)
                    printed_name = best_name
                else:
                    label = f"Unknown {best_score:.2f}"
                    color = (0, 0, 255)
                    printed_name = "Unknown"

                bbox = face.bbox.astype(int)

                cached_results.append(
                    {
                        "bbox": bbox,
                        "label": label,
                        "color": color,
                        "printed_name": printed_name,
                        "best_name": best_name,
                        "best_score": best_score
                    }
                )

                now = time.time()

                if now - last_print_time >= print_interval:
                    print(
                        f"Detected: {printed_name} | "
                        f"Best match: {best_name} | "
                        f"Score: {best_score:.3f}"
                    )
                    last_print_time = now

        for result in cached_results:
            draw_box(
                frame,
                result["bbox"],
                result["label"],
                result["color"]
            )

        cv2.imshow("InsightFace Name Recognition - Fast Mode", frame)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.38)

    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--print-interval", type=float, default=1.0)

    parser.add_argument("--process-every", type=int, default=3)

    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)

    parser.add_argument("--display-width", type=int, default=640)

    args = parser.parse_args()

    recognize(
        camera_index=args.camera,
        threshold=args.threshold,
        det_size=args.det_size,
        print_interval=args.print_interval,
        process_every=args.process_every,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        display_width=args.display_width
    )


if __name__ == "__main__":
    main()