import os
import json
import cv2

# Define the path to the testing dataset folder
testing_dataset_path = "F:/Project/Fingerprint and Hand Geometry biometric system/TestingDataset/"
training_data = {}  # Initialize as an empty dictionary

def load_training_data():
    """Load training data from JSON files."""
    global training_data
    for folder_num in range(1, 11):
        folder_path = os.path.join("F:/Project/Fingerprint and Hand Geometry biometric system/Dataset", str(folder_num))
        training_file_path = os.path.join(folder_path, f"{folder_num}_features.json")
        if os.path.exists(training_file_path):
            with open(training_file_path, 'r') as f:
                training_data[folder_num] = json.load(f)

def compare_features(train_features, test_features):
    """Compare palm and fingerprint features."""
    palm_match = (train_features["palm"]["defects_count"] == test_features["palm"]["defects_count"] and 
                  train_features["palm"]["area"] == test_features["palm"]["area"])

    fingerprint_matches = {}
    for finger_name in test_features["fingerprints"]:
        if finger_name in train_features["fingerprints"]:
            fingerprint_match = (train_features["fingerprints"][finger_name]["skeleton"] == 
                                 test_features["fingerprints"][finger_name]["skeleton"])
            fingerprint_matches[finger_name] = fingerprint_match

    return palm_match, fingerprint_matches

def process_palm_image(image):
    """Your palm image processing code here"""
    # Example processing logic
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    norm_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        feature_data = {
            "defects_count": len(cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))),
            "area": cv2.contourArea(max_contour),
            "skeleton": cv2.ximgproc.thinning(thresh).tolist()
            # Add other features if needed
        }
        return feature_data
    return None

def process_fingerprint_image(image):
    """Your fingerprint image processing code here"""
    # Example processing logic
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    norm_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"skeleton": cv2.ximgproc.thinning(thresh).tolist()}  # Add other features if needed

def run_testing(specific_folder_num):
    """Run the testing process based on the folder number."""
    output = []
    load_training_data()

    folder_path = os.path.join(testing_dataset_path, specific_folder_num)
    palm_image_name = f"{specific_folder_num}_Right_hand.jpg"
    palm_image_path = os.path.join(folder_path, palm_image_name)

    fingerprints = [
        f"{specific_folder_num}_Right_index_finger.bmp",
        f"{specific_folder_num}_Right_middle_finger.bmp",
        f"{specific_folder_num}_Right_ring_finger.bmp",
        f"{specific_folder_num}_Right_little_finger.bmp",
        f"{specific_folder_num}_Right_thumb_finger.bmp"
    ]

    output.append(f"Testing User: {specific_folder_num}\n")

    if os.path.exists(palm_image_path):
        palm_image = cv2.imread(palm_image_path)

        if palm_image is not None:
            test_palm_features = process_palm_image(palm_image)
            test_features = {"palm": test_palm_features, "fingerprints": {}}

            for finger_name in fingerprints:
                fingerprint_path = os.path.join(folder_path, finger_name)
                if os.path.exists(fingerprint_path):
                    fingerprint_image = cv2.imread(fingerprint_path, cv2.IMREAD_UNCHANGED)
                    test_fingerprint_features = process_fingerprint_image(fingerprint_image)
                    test_features["fingerprints"][finger_name] = test_fingerprint_features

            match_found = False
            matched_folders = []

            for train_folder_num, train_features in training_data.items():
                palm_match, fingerprint_matches = compare_features(train_features, test_features)
                
                if palm_match and all(fingerprint_matches.values()):
                    match_found = True
                    matched_folders.append(train_folder_num)

            if match_found:
                output.append(f"Match found against User: {', '.join(map(str, matched_folders))}.\n")
            else:
                output.append("No match found.\n")
        else:
            output.append(f"Failed to load testing palm image: {palm_image_name}\n")
    else:
        output.append(f"No match found.\n")

    return "\n".join(output)
