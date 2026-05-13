# InsightFace Name Recognition

A local face recognition project built with **InsightFace**, **OpenCV**, and **ONNX Runtime**.

The project lets you collect face samples from a webcam, build a local face embeddings database, and recognize people either from a live camera stream or from a single image.

> This project is designed for local experimentation and learning. Do not upload real face datasets, generated face databases, or private images to public repositories.

## Features

- Collect face images from a webcam
- Build a face embeddings database from collected images
- Recognize known and unknown faces from a live camera feed
- Recognize faces in a static image
- GPU-first ONNX Runtime setup with CPU fallback
- Simple Windows `.bat` launchers for common workflows

## Tech Stack

- Python
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy

## Project Structure

```text
insightface_name_recognition_project/
|-- src/
|   |-- collect_faces.py       # Collect face samples from webcam
|   |-- build_database.py      # Build embeddings database
|   |-- recognize_camera.py    # Real-time camera recognition
|   |-- recognize_image.py     # Image-based recognition
|   `-- utils.py               # Shared helpers and InsightFace setup
|-- dataset/                   # Local face images, ignored by Git
|-- models/                    # Generated face database, ignored by Git
|-- test_images/               # Local test images, ignored by Git
|-- run_1_collect.bat          # Windows helper: collect samples
|-- run_2_build_database.bat   # Windows helper: build database
|-- run_3_recognize.bat        # Windows helper: recognize from camera
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Requirements

- Python 3.10 or 3.11 recommended
- Webcam for live recognition
- Windows recommended for the included `.bat` scripts
- Optional: NVIDIA GPU with compatible CUDA/cuDNN setup

## Installation

Clone the repository:

```bash
git clone https://github.com/keroles-salah/insightface_name_recognition_project.git
cd insightface_name_recognition_project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first run may download InsightFace model files automatically.

## Usage

The normal workflow is:

1. Collect face samples
2. Build the face database
3. Run recognition

### 1. Collect Face Samples

Collect images for one person:

```bash
python src/collect_faces.py --name PersonName --samples 80
```

Example:

```bash
python src/collect_faces.py --name PersonName --samples 80
```

Useful options:

```bash
python src/collect_faces.py --name PersonName --samples 100 --camera 0 --det-size 640 --delay 0.15
```

Tips for better accuracy:

- Use clear lighting
- Capture different angles
- Move closer and farther from the camera
- Avoid blurry images
- Collect enough samples per person

Collected images are stored locally in:

```text
dataset/PersonName/
```

### 2. Build the Face Database

After collecting samples, build the embeddings database:

```bash
python src/build_database.py
```

This creates:

```text
models/face_database.npz
```

This file is generated locally and should not be committed to GitHub.

### 3. Recognize From Camera

Run real-time recognition:

```bash
python src/recognize_camera.py
```

Useful options:

```bash
python src/recognize_camera.py --camera 0 --threshold 0.38 --det-size 320 --process-every 3
```

Press `q` to stop the camera window.

### 4. Recognize From Image

Run recognition on a single image:

```bash
python src/recognize_image.py --image test_images/photo1.jpg
```

Save the annotated output to a custom path:

```bash
python src/recognize_image.py --image test_images/photo1.jpg --output output.jpg
```

## Windows Shortcuts

You can also use the included batch files:

```text
run_1_collect.bat
run_2_build_database.bat
run_3_recognize.bat
```

These are simple wrappers around the Python commands.

## Recognition Threshold

The default threshold is:

```text
0.38
```

Lower threshold:

- More likely to recognize a person
- Higher risk of false matches

Higher threshold:

- Stricter recognition
- Higher chance of marking people as `Unknown`

Tune the threshold based on your camera, lighting, and dataset quality.

## GPU Notes

The project attempts to use:

```text
CUDAExecutionProvider
CPUExecutionProvider
```

If CUDA is available and configured correctly, ONNX Runtime can run on GPU. If not, it should fall back to CPU.

For GPU support, you may need:

- NVIDIA GPU drivers
- CUDA runtime
- cuDNN
- `onnxruntime-gpu`

If you only want CPU execution, the default `onnxruntime` package is enough.

## Privacy and GitHub Safety

Do not commit:

```text
dataset/
test_images/
models/
output.jpg
```

These files may contain personal face data or generated biometric embeddings.

The included `.gitignore` is configured to keep these files out of the repository.

## Troubleshooting

### Camera does not open

Try another camera index:

```bash
python src/recognize_camera.py --camera 1
```

or:

```bash
python src/collect_faces.py --name PersonName --samples 80 --camera 1
```

### No face detected

Try:

- Better lighting
- Facing the camera directly
- Increasing detection size:

```bash
python src/build_database.py --det-size 960
```

### Face database not found

Run:

```bash
python src/build_database.py
```

before using camera or image recognition.

### CUDA DLL or ONNX Runtime warnings

If GPU dependencies are missing, install the correct NVIDIA/CUDA/cuDNN packages or use CPU mode with the standard `onnxruntime` package.

## Recommended Workflow for New People

For each new person:

```bash
python src/collect_faces.py --name PersonName --samples 80
python src/build_database.py
python src/recognize_camera.py
```

If recognition is unstable, collect more samples and rebuild the database.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
