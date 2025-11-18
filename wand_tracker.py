import cv2
import numpy as np
import time
import math
from picamera2 import Picamera2
from libcamera import Transform
from gpiozero import RGBLED
import pickle
import os
from tensorflow import keras
import threading
import queue

# Initialize global variables
picam2 = None
camera_active = False
wand_points = []
kalman_initialized = False
canvas = None
rgbled = RGBLED(red=17, green=27, blue=22)

# --- Camera Reinitialization Function ---
def reinitialize_camera():
    global picam2, camera_active, kalman_initialized, wand_points, canvas
    print("Attempting to reinitialize camera...")
    try:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception as stop_e:
                print(f"Error stopping camera: {stop_e}")
        
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}, transform=Transform(hflip=1, vflip=1)))
        picam2.start()
        camera_active = True
        kalman_initialized = False
        wand_points.clear()
        canvas = None
        print("Camera reinitialized successfully.")
        return True
    except Exception as reinit_e:
        print(f"Camera reinitialization failed: {reinit_e}.")
        camera_active = False
        picam2 = None
        return False

# --- Initial Camera Setup ---
try:
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}, transform=Transform(hflip=1, vflip=1)))
    picam2.start()
    camera_active = True
    print("Initial camera started successfully.")
except Exception as e:
    print(f"Initial camera start failed: {e}. Attempting recovery...")
    if not reinitialize_camera():
        print("Failed to start camera. Exiting.")
        exit()


# Trail fade configuration
FADE_DURATION = 1.5
MAX_INTENSITY = 255
MIN_AREA = 100
MAX_AREA = 1200
MIN_CIRCULARITY = 0.3

# Object detector - background subtraction
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=32, detectShadows=False)

# --- Kalman Filter Setup ---
state_size = 4
measurement_size = 2
control_size = 0

kalman = cv2.KalmanFilter(state_size, measurement_size, control_size)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)

kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 10, 0],
                                    [0, 0, 0, 10]], np.float32) * 20

kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 1

kalman.errorCovPost = np.eye(state_size, dtype=np.float32) * 1

last_time = time.time()

# --- Spell Recognition Setup ---
MODEL_FILENAME = 'spell_model_nn.h5'
SPELLS_FILENAME = 'reference_spells_nn.pkl'
CLASS_LABELS_FILENAME = 'spell_class_labels.pkl'

reference_spells = {} 
spell_model = None
class_labels = []

recording_spell = False
current_recording_path = []

# This value requires re-recording spells and retraining the model if changed!
NN_INPUT_SEQUENCE_LENGTH = 30

MIN_POINTS_FOR_RECOGNITION = NN_INPUT_SEQUENCE_LENGTH / 2

# How often to run NN prediction (e.g., 5 means predict every 5th frame)
NN_PREDICTION_SKIP_INTERVAL = 5
frame_counter_for_nn = 0 # Counter for skipping frames

# --- Path Preprocessing Functions ---
def normalize_path(path):
    if not path:
        return np.array([])
    
    path_np = np.array(path, dtype=np.float32)
    centroid = np.mean(path_np, axis=0)
    translated_path = path_np - centroid
    max_val = np.max(np.abs(translated_path))
    if max_val == 0:
        return translated_path
    scaled_path = translated_path / max_val
    return scaled_path

def resample_path(path_np, num_points):
    if path_np.size == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(path_np) == num_points:
        return path_np
    if len(path_np) == 1:
        return np.tile(path_np[0], (num_points, 1))

    original_indices = np.linspace(0, len(path_np) - 1, len(path_np))
    new_indices = np.linspace(0, len(path_np) - 1, num_points)

    resampled_x = np.interp(new_indices, original_indices, path_np[:, 0])
    resampled_y = np.interp(new_indices, original_indices, path_np[:, 1])
    
    return np.array([(x, y) for x, y in zip(resampled_x, resampled_y)], dtype=np.float32)


# --- Load existing data and model ---
def load_data_and_model():
    global reference_spells, spell_model, class_labels

    if os.path.exists(SPELLS_FILENAME):
        with open(SPELLS_FILENAME, 'rb') as f:
            reference_spells = pickle.load(f)
        print(f"Loaded {len(reference_spells)} reference spell types from {SPELLS_FILENAME}.")
    else:
        print(f"No existing raw reference spell patterns found at {SPELLS_FILENAME}.")

    if os.path.exists(CLASS_LABELS_FILENAME):
        with open(CLASS_LABELS_FILENAME, 'rb') as f:
            class_labels = pickle.load(f)
        print(f"Loaded {len(class_labels)} class labels from {CLASS_LABELS_FILENAME}.")
    else:
        print(f"No class labels found at {CLASS_LABELS_FILENAME}. Please train the model first.")
        class_labels = []

    if os.path.exists(MODEL_FILENAME):
        try:
            spell_model = keras.models.load_model(MODEL_FILENAME)
            print(f"Loaded trained Neural Network model from {MODEL_FILENAME}.")
        except Exception as e:
            print(f"Error loading Neural Network model from {MODEL_FILENAME}: {e}")
            spell_model = None
    else:
        print(f"No trained Neural Network model found at {MODEL_FILENAME}. Please train the model first.")
        spell_model = None


# Load data and model at startup
load_data_and_model()

print("Wand tracking started. Press 'q' to quit, 'c' to clear trail.")
print("Press 'r' to START recording a spell. Press 's' to STOP recording and save the spell.")
print("Press 't' to TRAIN the system (see console for instructions).")

frame_count=0
start_time=time.time()
most_recent_spell = "Waiting for Spell" # Initial display text

# --- New function to capture a frame with a timeout ---
def capture_with_timeout(q_result):
    try:
        frame = picam2.capture_array()
        q_result.put(frame)
    except Exception as e:
        q_result.put(f"Error during capture: {e}")

while True:
    now = time.time()
    dt = now - last_time
    last_time = now

    frame = None
    # Use a queue to get the result from the thread
    q_frame = queue.Queue()
    capture_thread = threading.Thread(target=capture_with_timeout, args=(q_frame,))
    capture_thread.daemon = True
    capture_thread.start()
    
    # Wait for the thread to finish, with a timeout
    try:
        frame = q_frame.get(timeout=2) # 2-second timeout
        if isinstance(frame, str): # Check if the queue returned an error message
            print(frame)
            if not reinitialize_camera():
                print("Failed to reinitialize camera. Exiting loop.")
                break
            continue
    except queue.Empty:
        print("Camera capture timed out. The camera is likely frozen. Reinitializing...")
        if not reinitialize_camera():
            print("Failed to reinitialize camera. Exiting loop.")
            break
        continue
    except Exception as e:
        print(f"An unexpected error occurred during frame capture: {e}. Reinitializing...")
        if not reinitialize_camera():
            print("Failed to reinitialize camera. Exiting loop.")
            break
        continue

    if canvas is None:
        canvas = np.zeros_like(frame)

    #frame_count += 1
    #if now - start_time >= 1.0:
    #    fps = frame_count / (now - start_time)
    #    print(f"FPS: {fps:.2f}")
    #    print(most_recent_spell)
    #    frame_count = 0
    #    start_time = now

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = object_detector.apply(gray)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, bright_mask)

    # --- Morphological Operations to clean up the mask ---
    kernel = np.ones((3,3), np.uint8) 
    mask = cv2.erode(mask, kernel, iterations = 1) # Increased iterations
    mask = cv2.dilate(mask, kernel, iterations = 1) # Increased iterations

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_wand_point = None
    max_wand_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if MIN_AREA < area < MAX_AREA:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > MIN_CIRCULARITY and area > max_wand_area:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                best_wand_point = (int(x), int(y))
                max_wand_area = area

    # --- Kalman Filter Prediction and Correction ---
    kalman.transitionMatrix[0, 2] = dt 
    kalman.transitionMatrix[1, 3] = dt 

    prediction = kalman.predict()
    predicted_x, predicted_y = int(prediction[0][0]), int(prediction[1][0])

    if best_wand_point:
        measurement = np.array([[best_wand_point[0]], [best_wand_point[1]]], np.float32)
        
        if not kalman_initialized:
            kalman.statePost = np.array([[best_wand_point[0]], 
                                         [best_wand_point[1]], 
                                         [0], 
                                         [0]], np.float32) 
            kalman_initialized = True
        else:
            kalman.correct(measurement)
        
        filtered_x, filtered_y = int(kalman.statePost[0][0]), int(kalman.statePost[1][0])
        wand_points.append((filtered_x, filtered_y, now))

        if recording_spell:
            current_recording_path.append((filtered_x, filtered_y))

    elif kalman_initialized:
        wand_points.append((predicted_x, predicted_y, now))
        if recording_spell:
            current_recording_path.append((predicted_x, predicted_y))

    wand_points = [(x, y, t) for (x, y, t) in wand_points if now - t <= FADE_DURATION]

    trail_layer = np.zeros_like(canvas)

    for i in range(1, len(wand_points)):
        (x1, y1, t1) = wand_points[i - 1]
        (x2, y2, t2) = wand_points[i]
        
        age = now - t2
        if 0 <= age <= FADE_DURATION:
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            
            alpha = 1.0 - (age / FADE_DURATION)
            intensity = int(MAX_INTENSITY * alpha)
            color = (intensity, intensity, intensity)
            
            cv2.line(trail_layer, (x1, y1), (x2, y2), color, thickness=2)

    canvas = cv2.addWeighted(canvas, 0.9, trail_layer, 0.5, 0)
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    # --- Spell Recognition Logic (Neural Network) ---
    recognized_spell_current_frame = "Waiting for Spell"
    confidence = 0.0

    # Increment frame counter for NN prediction skipping
    frame_counter_for_nn += 1

    # Only run NN prediction if the interval has passed AND model/labels are loaded
    if spell_model and len(class_labels) > 0 and len(wand_points) >= MIN_POINTS_FOR_RECOGNITION and \
       (frame_counter_for_nn % NN_PREDICTION_SKIP_INTERVAL == 0):
        
        current_live_path_segment = [(p[0], p[1]) for p in wand_points[-NN_INPUT_SEQUENCE_LENGTH:]]
        
        normalized_resampled_path = resample_path(normalize_path(current_live_path_segment), NN_INPUT_SEQUENCE_LENGTH)
        
        nn_input = normalized_resampled_path.reshape(1, NN_INPUT_SEQUENCE_LENGTH, 2)

        try:
            predictions = spell_model.predict(nn_input, verbose=0)[0]
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[predicted_class_index]
            
            if confidence > 0.7 and predicted_class_index < len(class_labels):
                current_prediction_name = class_labels[predicted_class_index]
                if current_prediction_name != "Idle" and current_prediction_name != "":
                    # Update most_recent_spell directly if confident and not Idle/empty
                    most_recent_spell = current_prediction_name
                recognized_spell_current_frame = current_prediction_name # For internal tracking/logging
            else:
                recognized_spell_current_frame = "Low Confidence / Idle" 

        except Exception as e:
            print(f"Error during NN prediction: {e}")
            recognized_spell_current_frame = "Error"
            confidence = 0.0
    elif not spell_model:
        recognized_spell_current_frame = "NN Model Not Loaded"
    elif len(class_labels) == 0:
        recognized_spell_current_frame = "Class Labels Missing"
    elif len(wand_points) < MIN_POINTS_FOR_RECOGNITION:
        recognized_spell_current_frame = "Not enough points" # If not enough points for NN_INPUT_SEQUENCE_LENGTH
    # If NN prediction was skipped this frame, recognized_spell_current_frame retains its last value
    # However, 'most_recent_spell' is only updated inside the 'if' block when a confident, non-idle spell is found.

    #cv2.putText(combined_bgr, f"Most Recent Spell: {most_recent_spell}", (10, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #cv2.putText(combined_bgr, f"Current Confidence: {confidence:.2f}", (10, 70), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Wand Trail (Live)", combined_bgr)
    #print(len(wand_points))

    if most_recent_spell == "Lumos":
        rgbled.color = (1,1,1)
    elif most_recent_spell == "Nox":
        rgbled.off()
    elif most_recent_spell == "Alohomora":
        rgbled.color = (0,0,1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        wand_points.clear()
        canvas[:] = 0
        kalman_initialized = False
        most_recent_spell = "Waiting for Spell" # Reset display on clear
        frame_counter_for_nn = 0 # Reset frame counter for NN on clear
    elif key == ord('r'):
        recording_spell = True
        current_recording_path.clear()
        print("--- Recording spell. Press 's' to stop. ---")
    elif key == ord('s'):
        recording_spell = False
        if current_recording_path:
            spell_name = input("Enter spell name to save: ")
            if spell_name:
                if spell_name not in reference_spells:
                    reference_spells[spell_name] = []
                reference_spells[spell_name].append(list(current_recording_path))
                print(f"Spell '{spell_name}' recorded with {len(current_recording_path)} points.")
                print(f"Total recordings for '{spell_name}': {len(reference_spells[spell_name])}")
                
                try:
                    with open(SPELLS_FILENAME, 'wb') as f:
                        pickle.dump(reference_spells, f)
                    print(f"Reference spells saved to '{SPELLS_FILENAME}'.")
                except Exception as save_e:
                    print(f"Error saving reference spells to file: {save_e}")

            else:
                print("Spell name not entered. Recording discarded.")
        else:
            print("No wand movement recorded.")
        current_recording_path.clear()

rgbled.off()
cv2.destroyAllWindows()