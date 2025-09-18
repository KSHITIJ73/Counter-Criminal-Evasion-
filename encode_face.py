#DataSet Training
import face_recognition
import pickle
import cv2
import os

# --- Configuration ---
DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pickle"
DETECTION_METHOD = "cnn" # or "cnn" for more accuracy (requires dlib with CUDA)
def resize_before_encoding(image, width=600):
    """
    Resize the image to a given width while maintaining aspect ratio.
    """
    (h, w) = image.shape[:2]
    if w > width:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized
    return image

def encode_faces():
    """
    Scans the dataset directory, encodes all faces, and saves them to a pickle file.
    """
    print("[INFO] Starting face encoding process...")
    known_encodings = []
    known_names = []

    # Check if the dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset directory not found at '{DATASET_PATH}'. Please create it and add images.")
        return

    # Loop over each person in the dataset directory
    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)
        
        # Skip any files that are not directories
        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Processing images for '{person_name}'...")
        # Loop over each image of the person
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            # Load the image and convert it from BGR (OpenCV default) to RGB
            try:
                # call function to resize image
                image = cv2.imread(image_path)
                image = resize_before_encoding(image)
                if image is None:
                    print(f"[WARNING] Could not read image: {image_path}. Skipping.")
                    continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}. Reason: {e}")
                continue
            
            # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face
            boxes = face_recognition.face_locations(rgb_image, model=DETECTION_METHOD)
            
            # Compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb_image, boxes)
            
            # Loop over the encodings (usually just one per image in our case)
            for encoding in encodings:
                # Add each encoding + name to our lists
                known_encodings.append(encoding)
                known_names.append(person_name)

    # Save the encodings and names to disk
    print("[INFO] Serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))
        
    print(f"[INFO] Encodings saved to '{ENCODINGS_FILE}'.")
    print(f"Total faces encoded: {len(known_encodings)}")

# --- Main Execution ---
if __name__ == "__main__":
    encode_faces()
