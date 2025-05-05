from ultralytics import YOLO
import cv2
import numpy as np
import math

# Load the trained model
model = YOLO(r'C:\Users\dell\Desktop\Projects\Palmistry model\Palm_Reader\Models\last.pt')

# Map class IDs to line names
class_map = {0: "head", 1: "heart", 2: "life"}

# Helper functions
def polygon_length(points):
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    return np.sum(distances)

def curvature_score(points):
    vecs = np.diff(points, axis=0)
    angles = []
    for i in range(1, len(vecs)):
        v1 = vecs[i - 1]
        v2 = vecs[i]
        angle = np.arccos(
            np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                -1.0, 1.0
            )
        )
        angles.append(angle)
    return np.mean(angles) if angles else 0

def analyze_line(length, angle_deg, line_type):
    if line_type == "heart":
        description = ("Emotionally open, intuitive, expressive." if angle_deg > 15 
                       else "Practical, reserved, and logical approach to emotions.")
        description += (" Strong capacity for love and empathy." if length > 250 
                        else " More reserved or less emotionally expressive nature." if length < 150 else "")
    elif line_type == "head":
        if length > 250 and angle_deg < 10:
            description = "Strong, logical mind; organized and may be a rule follower."
        elif angle_deg > 20:
            description = "Creative and intuitive mind; imaginative and inspired."
        elif length < 150:
            description = "Quick thinker; short attention span; impulsive."
        else:
            description = "Balanced and practical thinker."
    elif line_type == "life":
        description = ("Good health, vitality, and resilience." if length > 300 
                       else "Cautious personality or need for independence." if length < 150 else "Balanced energy levels.")
        description += (" Enthusiastic and adventurous nature." if angle_deg > 30 
                        else " Guarded and cautious in relationships." if angle_deg < 10 else "")
    else:
        description = "Unknown line type."
    return description

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Perform prediction
    results = model.predict(source=frame, conf=0.5, task='segment', save=False, verbose=False)
    result = results[0]
    annotated_frame = result.plot()

    # Extract mask and class data
    if result.masks is not None and result.boxes is not None:
        masks = result.masks.xy
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        # Keep best detection per class
        best_indices = {}
        for i, cls_id in enumerate(class_ids):
            if cls_id not in best_indices or confidences[i] > confidences[best_indices[cls_id]]:  # Check for best confidence
                best_indices[cls_id] = i

        # Analyze and overlay interpretation
        y_offset = 30
        for cls_id, idx in best_indices.items():
            mask = masks[idx]
            length = polygon_length(mask)
            curvature = curvature_score(mask)
            curvature_deg = math.degrees(curvature)
            line_type = class_map[int(cls_id)]
            interpretation = analyze_line(length, curvature_deg, line_type)

            # Display on frame
            text = f"{line_type.capitalize()} Line: {interpretation}"
            cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 35

    # Show result
    cv2.imshow('Live Palm Line Detection', annotated_frame)

    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting the prediction window...")
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

