import pickle
import os

# --- Filename definition ---
# The name of the file containing your spell recordings.
SPELLS_FILENAME = 'spell_class_labels.pkl'

# --- Main logic ---
# Check if the file exists before trying to open it.
if os.path.exists(SPELLS_FILENAME):
    try:
        # Open the file in binary read mode ('rb')
        with open(SPELLS_FILENAME, 'rb') as f:
            # Use pickle to load the data from the file.
            loaded_spells = pickle.load(f)
    
        print(f"--- Contents of {SPELLS_FILENAME} ---")
        # Check if the loaded data is not empty.
        if loaded_spells:
            # Iterate through each key-value pair in the dictionary.
            # The key is the spell name, and the value is a list of recordings.
            for spell_name, paths_list in loaded_spells.items():
                print(f"Spell: '{spell_name}'")
                # Print the number of recordings for the current spell.
                print(f"  Number of recordings: {len(paths_list)}")
                
                # --- Optional: Uncomment the following lines to see more detail ---
                # This will print the first and last five points of each recording.
                # for i, path in enumerate(paths_list):
                #     print(f"    Recording {i+1} (first 5 points): {path[:5]} ...")
                #     print(f"    Recording {i+1} (last 5 points): {path[-5:]}")
                print("-" * 20)
        else:
            print("No spell patterns found in the file.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
else:
    print(f"File '{SPELLS_FILENAME}' not found. Please ensure you have recorded and saved spells.")