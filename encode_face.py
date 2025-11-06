#DataSet Training
import face_recognition
import pickle
import cv2
import os
#Confi
DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pickle"
DETECTION_METHOD = "hog" # or "cnn" for more accuracy (requires dlib with CUDA)
def resize_before_encoding(image, width=800, height=800):
    """
    Resize the image to a given width while maintaining aspect ratio.
    Safely returns None if image is None.
    """
    if image is None:
        return None
    (h, w) = image.shape[:2]
    if w > width:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)    #inter = downscaling
        return resized
    return image

def encode_faces():
    """
    Scans the dataset directory, encodes all faces, and saves them to a pickle file.
    """
    print("[INFO] Starting face encoding process...")
    known_encodings = []
    known_names = []

    # Check if DS path exists
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset directory not found at '{DATASET_PATH}'. Please create it and add images.")
        return

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        # Skip files not in Dir
        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Processing images for '{person_name}'...")
        # Loop for each person
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            image = resize_before_encoding(image)

            if image is None:
                print(f"[WARNING] Could not read image: {image_path}. Skipping.")
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Detect coordinates
            boxes = face_recognition.face_locations(rgb_image, model=DETECTION_METHOD)

            # Embeddings
            encodings = face_recognition.face_encodings(rgb_image, boxes)
            
            # Loop for endcoding
            for encoding in encodings:
                # Encoding + name
                known_encodings.append(encoding)
                known_names.append(person_name)

    # Save encodings and names
    print(f"[INFO] Encoded {len(known_encodings)} face(s). Saving...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Encodings saved to '{ENCODINGS_FILE}'. Process completed.")

#Main Execution
if __name__ == "__main__":
    encode_faces()
