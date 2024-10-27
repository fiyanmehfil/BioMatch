import os
import cv2
import json
import numpy as np

# Paths to your datasets (for training)
dataset_path = "F:/Project/Fingerprint and Hand Geometry biometric system/Dataset/"
training_data = {}

def gabor_filter(image):
    """ Apply Gabor filters and return the filtered images. """
    gabor_features = []
    for theta in range(4):  # 0, 45, 90, 135 degrees
        theta = theta / 4.0 * np.pi
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        gabor_features.append(filtered)
    return gabor_features

def process_palm_image(image):
    """Process palm image for features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    norm_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        feature_data = {
            "defects_count": len(cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))),
            "area": cv2.contourArea(max_contour),
            "perimeter": cv2.arcLength(max_contour, True),
            "convex_hull_points": cv2.convexHull(max_contour).tolist(),
            "gabor_features": [gabor_feature.tolist() for gabor_feature in gabor_filter(gray)],
            "skeleton": cv2.ximgproc.thinning(thresh).tolist(),
            "minutiae_points": [],  # Placeholder
            "singularity_points": []  # Placeholder
        }
        return feature_data
    else:
        return None

def process_fingerprint_image(image):
    """Process fingerprint image for features."""
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    feature_data = {
        "directional_map": [],  # Placeholder
        "frequency_map": [],     # Placeholder
        "gabor_features": [],     # Placeholder
        "skeleton": cv2.ximgproc.thinning(thresh).tolist(),
        "minutiae_points": [],  # Placeholder
        "singularity_points": []  # Placeholder
    }

    return feature_data

def run_training():
    """Run the training process and return output as a string."""
    output = []

    output.append("Starting Training Phase...\n")
    for folder_num in range(1, 11):
        folder_path = os.path.join(dataset_path, str(folder_num))
        palm_image_name = f"{folder_num}_Right_hand.jpg"
        palm_image_path = os.path.join(folder_path, palm_image_name)

        fingerprints = [
            f"{folder_num}_Right_index_finger.bmp",
            f"{folder_num}_Right_middle_finger.bmp",
            f"{folder_num}_Right_ring_finger.bmp",
            f"{folder_num}_Right_little_finger.bmp",
            f"{folder_num}_Right_thumb_finger.bmp"
        ]

        if os.path.exists(palm_image_path):
            palm_image = cv2.imread(palm_image_path)
            
            if palm_image is not None:
                output.append(f"Processing User {folder_num}\n")
                palm_features = process_palm_image(palm_image)
                if palm_features:
                    combined_features = {"palm": palm_features, "fingerprints": {}}
                    
                    for finger_name in fingerprints:
                        fingerprint_path = os.path.join(folder_path, finger_name)
                        if os.path.exists(fingerprint_path):
                            fingerprint_image = cv2.imread(fingerprint_path, cv2.IMREAD_UNCHANGED)
                            fingerprint_features = process_fingerprint_image(fingerprint_image)
                            combined_features["fingerprints"][finger_name] = fingerprint_features
                    
                    training_file_path = os.path.join(folder_path, f"{folder_num}_features.json")
                    with open(training_file_path, 'w') as f:
                        json.dump(combined_features, f)
                    output.append(f"Features saved for User {folder_num}\n")
                else:
                    output.append(f"No contours found for palm image of User {folder_num}\n")
            else:
                output.append(f"Failed to load palm image for User {folder_num}\n")
        else:
            output.append(f"Palm image not found for User {folder_num}\n")

    return "\n".join(output)
