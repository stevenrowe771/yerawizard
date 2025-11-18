import pickle
import os

# --- Filename definition ---
# The name of the file containing the class label to spell name mapping.
LABELS_FILENAME = 'spell_class_labels.pkl'

# --- Main logic ---
# Check if the file exists before trying to open it.
if os.path.exists(LABELS_FILENAME):
    try:
        # Open the file in binary read mode ('rb')
        with open(LABELS_FILENAME, 'rb') as f:
            # Use pickle to load the data from the file.
            loaded_labels = pickle.load(f)
    
        print(f"--- Contents of {LABELS_FILENAME} ---")
        
        # Check if the loaded data is a dictionary, as expected.
        if isinstance(loaded_labels, dict):
            # Iterate through each key-value pair in the dictionary.
            # The key is the integer label and the value is the spell name.
            for label, spell_name in loaded_labels.items():
                print(f"Label {label}: '{spell_name}'")
            
            print("-" * 20)
            print(f"Total unique spell labels found: {len(loaded_labels)}")
        # If it's not a dictionary, check if it's a list
        elif isinstance(loaded_labels, list):
            # Iterate through each item in the list, using the index as the label.
            for label, spell_name in enumerate(loaded_labels):
                print(f"Label {label}: '{spell_name}'")
            
            print("-" * 20)
            print(f"Total unique spell labels found: {len(loaded_labels)}")
        else:
            # If the data is neither a dictionary nor a list, print an error message.
            print(f"Loaded object is not a dictionary or a list: {type(loaded_labels)}")
            print("The file may be corrupted or contain unexpected data.")

    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
else:
    print(f"File '{LABELS_FILENAME}' not found. Please ensure you have trained and saved the model.")