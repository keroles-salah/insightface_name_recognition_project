import argparse
import cv2
import numpy as np

from utils import DATABASE_PATH, create_face_app, l2_normalize, draw_face_box


def recognize_image(image_path, threshold, det_size, output_path):
    if not DATABASE_PATH.exists():
        print("Error: face database not found.")
        print("Run: python src/build_database.py")
        return

    data = np.load(DATABASE_PATH, allow_pickle=True)
    names = data["names"]
    known_embeddings = data["embeddings"]

    app = create_face_app(det_size=det_size, det_thresh=0.5)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    faces = app.get(image)

    if not faces:
        print("No faces detected.")
        return

    for face in faces:
        emb = l2_normalize(face.normed_embedding)
        similarities = np.dot(known_embeddings, emb)

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        best_name = str(names[best_index])

        if best_score >= threshold:
            label = f"{best_name} {best_score:.2f}"
            color = (0, 255, 0)
            print(f"Detected: {best_name} | Score: {best_score:.3f}")
        else:
            label = f"Unknown {best_score:.2f}"
            color = (0, 0, 255)
            print(f"Detected: Unknown | Best match: {best_name} | Score: {best_score:.3f}")

        draw_face_box(image, face, label, color)

    cv2.imwrite(output_path, image)
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--threshold", type=float, default=0.38)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--output", default="output.jpg")

    args = parser.parse_args()
    recognize_image(args.image, args.threshold, args.det_size, args.output)


if __name__ == "__main__":
    main()