# Object Detection with YOLO and Voice Commands

A real-time object detection system using YOLO (You Only Look Once) with voice command support for class selection. This project allows you to train custom YOLO models and track objects in real-time via webcam with both text and voice input options.

## Features

- *Custom YOLO Model Training**: Train YOLO models on custom datasets
- **Real-time Object Tracking**: Track objects in real-time using webcam
- **Voice Commands**: Select object classes using voice recognition (French supported)
- **Text Input**: Alternative text-based class selection
- **Multi-class Detection**: Detect and track multiple object classes simultaneously
- **GPU Support**: Automatic CUDA detection for faster training and inference

## Supported Classes

The current model is trained to detect the following objects:
- `box` (boîte, boite)
- `pen` (stylo, crayon)
- `rag` (chiffon, torchon)
- `smarties` (smartie)

## Requirements

### Hardware
- Webcam (for real-time tracking)
- Microphone (for voice commands)
- NVIDIA GPU with CUDA support (recommended for training)

### Software
- Python 3.8 or higher
- CUDA-compatible GPU drivers (for GPU acceleration)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ObjectDetectionIA2
   ```

2. **Install dependencies**
   ```bash
   pip install ultralytics SpeechRecognition opencv-python keyboard pyaudio
   ```

   **Note for Windows users**: If `pyaudio` installation fails, try:
   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```
   Or download precompiled binaries from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

3. **Install PyTorch with CUDA support** (if you have an NVIDIA GPU)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

## Project Structure

```
ObjectDetectionIA2/
├── object_detection.py    # Main application file
├── dataset.yaml           # Dataset configuration file
├── yolo11n.pt             # Base YOLO model
├── train/                 # Training dataset
│   ├── images/           # Training images
│   └── labels/           # Training labels (YOLO format)
├── valid/                 # Validation dataset
│   ├── images/           # Validation images
│   └── labels/           # Validation labels
├── test/                  # Test dataset
│   ├── images/           # Test images
│   └── labels/           # Test labels
└── runs/train/            # Training results and saved models
    └── yolo11n_custom2/
        └── weights/
            └── best.pt    # Best trained model
```

## Usage

### Running the Application

```bash
python object_detection.py
```

### Main Menu Options

1. **Train the model**: Train a new YOLO model on your custom dataset
2. **Tracking**: Start real-time object tracking with webcam
3. **Exit**: Quit the application

### Training a Model

1. Select option `1` from the main menu
2. Ensure `dataset.yaml` exists in the project directory
3. The training will start automatically with the following parameters:
   - Epochs: 100
   - Image size: 1280
   - Batch size: 16
   - Data augmentation: rotation up to 180 degrees
   - Device: CUDA (if available) or CPU

### Real-time Object Tracking

1. Select option `2` from the main menu
2. Choose input method:
   - `t` for text input
   - `s` for speech input
3. **Text mode**: Enter class names separated by commas (e.g., `box, pen`)
4. **Speech mode**: Hold `Ctrl` and speak the object names in French
5. Press `q` to stop tracking and return to class selection
6. Type `back` in text mode to exit tracking

### Voice Commands

When using speech input:
- Hold the `Ctrl` key while speaking
- Release `Ctrl` to stop recording
- Speak in French (language: fr-FR)
- Supported synonyms:
  - Box: "box", "boîte", "boite"
  - Pen: "pen", "stylo", "crayon"
  - Rag: "rag", "chiffon", "torchon"
  - Smarties: "smarties", "smartie"

**Example**: Say "je cherche une box et des smarties" to detect both box and smarties.

## Configuration

### Dataset Configuration (`dataset.yaml`)

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 4
names: ['box', 'pen', 'rag', 'smarties']
```

### Model Configuration

Edit the following variables in `object_detection.py`:

```python
YOLO_CLASSES = ['box', 'pen', 'rag', 'smarties']  # Object classes
model = "yolo11n.pt"                              # Base model
model_path = r"runs\train\yolo11n_custom2\weights\best.pt"  # Trained model path
```

## Training Parameters

The default training configuration:
- **Model**: YOLO11n (nano)
- **Epochs**: 100
- **Image Size**: 1280x1280
- **Batch Size**: 16
- **Augmentation**: Rotation up to 180°
- **Device**: Auto-detected (CUDA/CPU)
- **Tracker**: ByteTrack (for object tracking)

## Dependencies

- `ultralytics`: YOLO model implementation
- `opencv-python`: Computer vision and webcam access
- `speech_recognition`: Voice command processing
- `keyboard`: Keyboard input detection
- `pyaudio`: Audio recording
- `torch`: PyTorch (with CUDA support recommended)

## Author

Clément Delporte & Ludovic Cure-Moog

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- Google Speech Recognition API for voice processing
