import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

# --- File paths ---
SPELLS_FILENAME = 'reference_spells_nn.pkl'
MODEL_FILENAME = 'spell_model_nn.h5'
CLASS_LABELS_FILENAME = 'spell_class_labels.pkl'

# --- Preprocessing Parameters ---
# This MUST match NN_INPUT_SEQUENCE_LENGTH in your main wand_tracker.py
NN_INPUT_SEQUENCE_LENGTH = 30 

# --- Path Preprocessing Functions (copied from wand_tracker.py for consistency) ---
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

# --- Load raw reference spell data ---
if os.path.exists(SPELLS_FILENAME):
    with open(SPELLS_FILENAME, 'rb') as f:
        reference_spells = pickle.load(f)
    print(f"Loaded {len(reference_spells)} reference spell types from '{SPELLS_FILENAME}'.")
else:
    print(f"Error: '{SPELLS_FILENAME}' not found. Please record spells in the main application first.")
    exit()

# --- Prepare data for Neural Network training ---
X_data = [] # Features (preprocessed paths)
y_labels = [] # Labels (spell names)

all_paths_with_labels = []
for spell_name, paths in reference_spells.items():
    for path in paths:
        all_paths_with_labels.append((path, spell_name))

if not all_paths_with_labels:
    print("No spell patterns found in loaded data. Record spells and try again.")
    exit()

for path, spell_name in all_paths_with_labels:
    normalized_path = normalize_path(path)
    resampled_path = resample_path(normalized_path, NN_INPUT_SEQUENCE_LENGTH)
    
    # Reshape for NN: (sequence_length, num_features_per_point)
    # NN expects input shape (batch_size, timesteps, features)
    X_data.append(resampled_path)
    y_labels.append(spell_name)

X_data = np.array(X_data)
y_labels = np.array(y_labels)

if X_data.shape[0] < 1:
    print("Not enough data to train the model.")
    exit()

# Encode string labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)

# Save the class labels (order is crucial for inference)
with open(CLASS_LABELS_FILENAME, 'wb') as f:
    pickle.dump(label_encoder.classes_, f)
print(f"Class labels saved to '{CLASS_LABELS_FILENAME}'.")

# Convert numerical labels to one-hot encoding (required for categorical_crossentropy)
y_one_hot = keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# Split data into training and validation sets
# Stratify ensures that the proportion of classes is the same in train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_one_hot, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples.")
print(f"Input shape: {X_train.shape[1:]}") # Should be (NN_INPUT_SEQUENCE_LENGTH, 2)
print(f"Output classes: {num_classes}")

# --- Build the 1D CNN Model (Enhanced Architecture Example) ---
# This architecture is an example. You may need to tune filters, kernel_size, units, etc.
model = keras.Sequential([
    layers.Input(shape=(NN_INPUT_SEQUENCE_LENGTH, 2)), # Input layer expects (sequence_length, features_per_point)
    
    layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'), # More filters, larger kernel
    layers.MaxPooling1D(pool_size=2), # Reduces spatial dimension
    
    layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'), # Deeper convolutional layer
    layers.MaxPooling1D(pool_size=2),
    
    layers.Flatten(), # Flatten the output of convolutional layers into a 1D vector
    
    layers.Dense(256, activation='relu'), # More neurons in dense layer
    layers.Dropout(0.5), # Regularization to prevent overfitting
    
    layers.Dense(num_classes, activation='softmax') # Output layer, num_classes neurons
])

# --- Compile the Model ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Print model summary to see architecture and parameter count

# --- Train the Model ---
# EarlyStopping callback:
# monitor='val_loss': Stop training when validation loss stops improving
# patience=20: Number of epochs with no improvement after which training will be stopped.
# restore_best_weights=True: Restores model weights from the epoch with the best monitored quantity (val_loss).
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("\nStarting model training...")
history = model.fit(X_train, y_train,
                    epochs=200, # Max epochs, EarlyStopping will likely stop it sooner
                    batch_size=32, # Batch size for training
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping], # Add early stopping callback here
                    verbose=1)

print("\nModel training finished.")

# --- Evaluate the Model (Optional) ---
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Loss: {loss:.4f}")

# --- Save the trained model ---
model.save(MODEL_FILENAME)
print(f"Trained Neural Network model saved to '{MODEL_FILENAME}'.")