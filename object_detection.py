"""
Object detection module with YOLO and voice commands.

This module allows training a custom YOLO model and tracking objects in real-time
via a webcam, with the ability to select classes to detect using voice commands.

Authors: Clément Delporte && Ludovic Cure-Moog
Date: 2025
"""

from ultralytics import YOLO
import speech_recognition as sr
import cv2
import keyboard
import time

# Available YOLO classes for detection
YOLO_CLASSES = ['box', 'pen', 'rag', 'smarties']

# Base YOLO model used for training
model = "yolo11n.pt"

# Path to the trained custom model
model_path = r"runs\train\yolo11n_custom\weights\best.pt"


def train_model():
    """
    Train a YOLO model on a custom dataset.
    
    This function loads a base YOLO model and trains it on a custom dataset
    defined in the dataset.yaml file. The trained model is saved in the
    runs/train/yolo11n_custom/ directory.
    
    Raises:
        FileNotFoundError: If the dataset.yaml file does not exist in the
            script directory.
    
    Note:
        - The model uses CUDA if available, otherwise CPU
        - Training parameters: 100 epochs, image size 1280, batch size 16,
          rotation up to 180 degrees
        - The base model used is yolo11n.pt
    """
    from pathlib import Path
    
    # Get the script directory and dataset.yaml path
    script_dir = Path(__file__).parent.absolute()
    dataset_yaml_path = script_dir / "dataset.yaml"
    
    if not dataset_yaml_path.exists():
        print(f"Error: The dataset.yaml file does not exist in {script_dir}")
        print("Please ensure the dataset.yaml file is present in the project directory.")
        return
    
    print(f"✓ Using configuration file: {dataset_yaml_path}")
    print(f"✓ Model used for training: {model}")
    
    model_train = YOLO(model)
    model_train.train(
        data=str(dataset_yaml_path),
        epochs=100,
        imgsz=1280,
        batch=16,
        degrees=180,
        device='cuda',
        save=True,
        project="runs/train",
        name="yolo11n_custom"
    )
    print("Training completed successfully.\n")

def speech_to_text():
    """
    Record voice and convert speech to text.
    
    This function records audio from the microphone while the Ctrl key is held down.
    Recording stops as soon as Ctrl is released. The audio signal is then converted
    to text using the Google Speech Recognition API.
    
    Returns:
        str: The recognized text from the audio, or None if recognition fails
            or if no audio was recorded.
    
    Raises:
        sr.UnknownValueError: If the API cannot understand the recorded audio.
        sr.RequestError: If an error occurs when requesting the API
            (internet connection required).
    
    Note:
        - The language used is French (fr-FR)
        - Recording uses a sampling rate of 44100 Hz
        - A temporary WAV file is created and automatically deleted
    """
    import pyaudio
    import wave
    import tempfile
    import os
    
    recognizer = sr.Recognizer()
    language_selected = "fr-FR"
    
    print(f"Hold 'Ctrl' to speak... (Language: French)")
    print("Release 'Ctrl' to stop recording.")

    while not keyboard.is_pressed('ctrl'):
        time.sleep(0.05)

    print("Recording...")
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    audio = pyaudio.PyAudio()
    sample_width = audio.get_sample_size(FORMAT)
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
    
    frames = []
    
    while keyboard.is_pressed('ctrl'):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        time.sleep(0.01) 

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    print("Recording stopped. Processing...")
    
    if not frames:
        print("No audio recorded.")
        return None
    
    duration = len(frames) * CHUNK / RATE
    print(f"Recording duration: {duration:.2f} seconds")
    
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file_path = tmp_file.name
        
        # Write the WAV file
        wf = wave.open(tmp_file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print("Analyzing audio...")

        with sr.AudioFile(tmp_file_path) as source:
            audio_data = recognizer.record(source)
        
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                time.sleep(0.1)
                os.unlink(tmp_file_path)
            except (PermissionError, OSError):
                pass

    print("Speech recognition in progress...")
    try:
        text = recognizer.recognize_google(audio_data, language=language_selected)
        print("=" * 50)
        print(f"RECOGNIZED TEXT: '{text}'")
        print("=" * 50)
        return text
    except sr.UnknownValueError:
        print("=" * 50)
        print("ERROR: Unable to understand the audio.")
        print("Please ensure you speak clearly and that the microphone is working.")
        print("=" * 50)
        return None
    except sr.RequestError as e:
        print("=" * 50)
        print(f"ERROR: Unable to get speech recognition results: {e}")
        print("Please check your internet connection.")
        print("=" * 50)
        return None


def speech_to_class():
    """
    Convert speech to a list of detected YOLO classes.
    
    This function uses speech recognition to capture a phrase, then analyzes
    the text to identify mentioned YOLO classes. It supports multiple synonyms
    for each class (French and English).
    
    Returns:
        list: List of YOLO class names detected in the speech.
            Returns an empty list if no classes are found or if
            speech recognition fails.
    
    Example:
        >>> speech_to_class()
        # User says: "je cherche une box et des smarties"
        # Returns: ['box', 'smarties']
    
    Note:
        - Supported classes are: 'box', 'pen', 'rag', 'smarties'
        - Supported synonyms:
            * box: "box", "boîte", "boite"
            * pen: "pen", "stylo", "crayon"
            * rag: "rag", "chiffon", "torchon"
            * smarties: "smarties", "smartie"
    """
    text = speech_to_text()
    if not text:
        print("No text recognized, no classes detected.")
        return []

    print(f"\nText to analyze: '{text}'")
    text = text.lower().strip()

    mapping = {

        "box": YOLO_CLASSES[0],
        "boîte": YOLO_CLASSES[0],
        "boite": YOLO_CLASSES[0],

        "pen": YOLO_CLASSES[1],
        "stylo": YOLO_CLASSES[1],
        "crayon": YOLO_CLASSES[1],

        "rag": YOLO_CLASSES[2],
        "chiffon": YOLO_CLASSES[2],
        "torchon": YOLO_CLASSES[2],

        "smarties": YOLO_CLASSES[3],
        "smartie": YOLO_CLASSES[3],
    }

    detected_classes = []

    for keyword, YOLO_class in mapping.items():
        if keyword in text:
            if YOLO_class not in detected_classes:
                detected_classes.append(YOLO_class)

    if detected_classes:
        print("-" * 50)
        print(f"✓ Detected classes: {detected_classes}")
        print("-" * 50)
    else:
        print("-" * 50)
        print("✗ No YOLO classes recognized in your speech.")
        print(f"   Keywords searched: {list(mapping.keys())}")
        print("-" * 50)

    return detected_classes

def track_model(webcam, model_track, class_indices):
    """
    Track specific objects in real-time via a webcam.
    
    This function captures images from the webcam and uses a trained YOLO model
    to detect and track objects of the specified classes. Results are displayed
    in real-time with visual annotations.
    
    Args:
        webcam (cv2.VideoCapture): VideoCapture object to access the webcam.
        model_track (YOLO): Trained YOLO model to use for detection.
        class_indices (list): List of class indices to detect and track.
            The indices correspond to positions in YOLO_CLASSES.
    
    Note:
        - Press 'q' to stop tracking
        - Closing the window also stops tracking
        - The tracker used is ByteTrack (bytetrack.yaml)
        - Confidence threshold: 0.6, IOU: 0.5
        - Image size: 960 pixels
        - Detection information (class and confidence) is displayed
          on the video in real-time
    """
    print(f"Tracking running for class indices {class_indices} (press 'q' to stop tracking)")
    window_name = "YOLO Tracking"
    cv2.namedWindow(window_name)

    while True:
        check, frame = webcam.read()

        if not check:
            print("Error reading from webcam.")
            break

        results = model_track.track(
            frame,
            persist=True,
            classes=class_indices,
            conf=0.6,
            iou=0.5,
            imgsz=960,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated = results[0].plot()
        
        # Display detection info on the frame
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            y_offset = 30
            for box in results[0].boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = YOLO_CLASSES[cls]
                info_text = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated, info_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') : 
            print("Tracking stopped. Returning to class selection.")
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Returning to class selection.")
            break

    cv2.destroyAllWindows()


def tracking_menu():
    """
    Interactive menu for object tracking via webcam with voice or text selection.
    
    This function opens the webcam and allows the user to select object classes
    to track, either by text input or voice command. Tracking can be repeated
    with different classes without restarting the webcam.
    
    Returns:
        None: The function returns None if the webcam cannot be opened.
    
    Note:
        - User first chooses input mode: 't' for text, 's' for speech
        - In text mode: enter class names separated by commas
        - In speech mode: hold Ctrl and say object names
        - Type 'back' in text mode to exit
        - Multiple classes can be tracked simultaneously
        - The webcam remains open until the user exits the menu
    """
    model_track = YOLO(model_path)
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Unable to access the webcam.")
        return
    
    while True:
        user_input_type = input("Choose input type for class selection - text (t) or speech (s): ").strip().lower()
        if user_input_type in ["t", "s"]:
            break
        print("Invalid selection. Please try again ('t' for test or 's' for speech).")
        

    while True:
        print(f"Available classes: {YOLO_CLASSES}")
        text_input =  []
        if user_input_type == "t": # text input
            text_input = input("Enter class name(s) to track (use a comma for multiple classes, 'back' to exit): ").strip().lower()
            if text_input == "back":
                print("Exiting tracking mode.")
                break
            # select classes (allow multiple classes separated by commas)
            class_names = [c.strip() for c in text_input.split(",") if c.strip()]
        
        else:  # speech input
            class_names = speech_to_class() # returns a LIST of classes
        
        class_indexes = []
        
        for name in class_names:
            if name in YOLO_CLASSES:
                class_indexes.append(YOLO_CLASSES.index(name))
            else:
                print(f"Unknown class ignored: {name}")

        if not class_indexes:
            print("No valid class found, try again.")
            continue
        
        track_model(webcam, model_track, class_indexes)

    webcam.release()
    cv2.destroyAllWindows()


def main():
    """
    Main menu for the object detection program.
    
    This function displays an interactive menu allowing the user to:
    - Train a new YOLO model on a custom dataset
    - Use the webcam to track objects in real-time
    - Exit the program
    
    Returns:
        None
    
    Note:
        - Option 1: Launches YOLO model training
        - Option 2: Opens the tracking menu with webcam
        - Option 3: Exits the program
    """
    while True:
        print("=== MENU ===")
        print("1. [TRAIN] the model")
        print("2. [TRACKING] a specific class with the webcam")
        print("3. Exit")

        choice = input("Enter your choice (1, 2 or 3): ").strip()

        if choice == "1":
            train_model()

        elif choice == "2":
            tracking_menu()

        elif choice == "3":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter 1, 2 or 3.\n")
# ====================================


if __name__ == "__main__":
    main()

