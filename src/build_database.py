import argparse
import cv2
import numpy as np

from utils import DATASET_DIR, DATABASE_PATH, ensure_dirs, create_face_app, get_largest_face, l2_normalize


def build_database(det_size):
    ensure_dirs()

    person_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])

    if not person_dirs:
        print("Error: No person folders found.")
        print("Example: dataset/Keroles/0001.jpg")
        return

    app = create_face_app(det_size=det_size, det_thresh=0.5)

    names = []
    embeddings = []
    image_counts = []

    print("=" * 60)
    print("Building face database...")
    print("=" * 60)

    for person_dir in person_dirs:
        person_name = person_dir.name

        image_files = (
            list(person_dir.glob("*.jpg")) +
            list(person_dir.glob("*.jpeg")) +
            list(person_dir.glob("*.png"))
        )

        if not image_files:
            print(f"Skipping {person_name}: no images.")
            continue

        person_embeddings = []

        print(f"\nPerson: {person_name}")
        print(f"Images found: {len(image_files)}")

        for image_path in image_files:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"  Could not read: {image_path}")
                continue

            faces = app.get(image)
            face = get_largest_face(faces)

            if face is None:
                print(f"  No face detected: {image_path.name}")
                continue

            emb = l2_normalize(face.normed_embedding)
            person_embeddings.append(emb)

        if not person_embeddings:
            print(f"  No valid embeddings for {person_name}.")
            continue

        avg_embedding = np.mean(person_embeddings, axis=0)
        avg_embedding = l2_normalize(avg_embedding)

        names.append(person_name)
        embeddings.append(avg_embedding)
        image_counts.append(len(person_embeddings))

        print(f"  Valid face images used: {len(person_embeddings)}")

    if not embeddings:
        print("Error: No embeddings were created.")
        return

    names = np.array(names)
    embeddings = np.array(embeddings, dtype=np.float32)
    image_counts = np.array(image_counts)

    np.savez(
        DATABASE_PATH,
        names=names,
        embeddings=embeddings,
        image_counts=image_counts
    )

    print("\n" + "=" * 60)
    print("Database created successfully.")
    print(f"Saved to: {DATABASE_PATH}")
    print("People:", ", ".join(names.tolist()))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-size", type=int, default=640, help="Detection size: 640, 960, or 1280")
    args = parser.parse_args()

    build_database(det_size=args.det_size)


if __name__ == "__main__":
    main()