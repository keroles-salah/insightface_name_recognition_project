from pathlib import Path
import os
import site


# ============================================================
# 1) Load CUDA / cuDNN DLL paths before importing InsightFace
# ============================================================

def add_nvidia_dll_paths():
    """
    Adds NVIDIA CUDA/cuDNN DLL folders from site-packages to Windows DLL search path.
    This helps fix errors like:
    - cublasLt64_12.dll is missing
    - cudnn_engines_tensor_ir64_9.dll is missing
    """

    possible_site_packages = []

    try:
        possible_site_packages.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        possible_site_packages.append(site.getusersitepackages())
    except Exception:
        pass

    dll_subfolders = [
        r"nvidia\cudnn\bin",
        r"nvidia\cublas\bin",
        r"nvidia\cuda_runtime\bin",
        r"nvidia\cuda_nvrtc\bin",
    ]

    for site_package_path in possible_site_packages:
        for subfolder in dll_subfolders:
            dll_path = os.path.join(site_package_path, subfolder)

            if os.path.isdir(dll_path):
                try:
                    os.add_dll_directory(dll_path)
                except Exception:
                    pass

                os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")


add_nvidia_dll_paths()


# ============================================================
# 2) Preload ONNX Runtime CUDA DLLs
# ============================================================

import onnxruntime as ort

try:
    # directory="" makes ONNX Runtime search inside NVIDIA packages in site-packages
    ort.preload_dlls(directory="")
    print("CUDA DLLs preloaded successfully.")
except Exception as e:
    print("CUDA DLL preload warning:", e)


# ============================================================
# 3) Normal project imports
# ============================================================

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# 4) Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_PATH = MODELS_DIR / "face_database.npz"


def ensure_dirs():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5) Math helpers
# ============================================================

def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def cosine_similarity(a, b):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


# ============================================================
# 6) InsightFace app
# ============================================================

def create_face_app(det_size=640, det_thresh=0.5):
    """
    Creates InsightFace app using GPU first, then CPU as fallback.
    """

    print("Available ONNX Runtime providers:", ort.get_available_providers())

    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(det_size, det_size),
        det_thresh=det_thresh
    )

    return app


# ============================================================
# 7) Face helpers
# ============================================================

def get_largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    return max(faces, key=area)


def crop_face_with_margin(image, bbox, margin_ratio=0.25):
    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox.astype(int)

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = int(box_width * margin_ratio)
    margin_y = int(box_height * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    return image[y1:y2, x1:x2]


def draw_face_box(frame, face, label, color):
    x1, y1, x2, y2 = face.bbox.astype(int)

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