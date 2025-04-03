import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Path to the photo folder
photo_dir = "photo"
reference_images = []
reference_points = []

# Extract points from images in the photo folder
for img_name in sorted(os.listdir(photo_dir)):
    if not img_name.endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(photo_dir, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        reference_images.append(img)
        reference_points.append(results.pose_landmarks)

# Function to normalize landmarks to relative positions
def normalize_landmarks(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    return [(lm.x - base_x, lm.y - base_y) for lm in landmarks]

# Function to calculate similarity between two sets of normalized points
def calculate_similarity(user_points, ref_points, threshold=0.05):
    if len(user_points) != len(ref_points):
        return 0.0

    matched_points = 0
    for user_point, ref_point in zip(user_points, ref_points):
        distance = np.linalg.norm(np.array(user_point) - np.array(ref_point))
        if distance < threshold:
            matched_points += 1

    return matched_points / len(ref_points)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the webcam frame
    frame = cv2.flip(frame, 1)

    # Process webcam frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # Initialize similarity percentage
    similarity_percentage = 0

    # Draw landmarks on the mirrored webcam frame
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Compare user's points with reference points
        if reference_points:
            user_normalized = normalize_landmarks(results.pose_landmarks.landmark)
            ref_normalized = normalize_landmarks(reference_points[0].landmark)
            similarity = calculate_similarity(user_normalized, ref_normalized)
            similarity_percentage = int(similarity * 100)
            if similarity >= 0.70:  
                cv2.putText(frame, "GOOD JOB POSITION IS CORRECT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display similarity percentage at the top of the frame with black text and white background
    text = f"Match: {similarity_percentage}%"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 30 + text_height), (255, 255, 255), -1) 
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  

    # Prepare reference image and points
    if reference_images:
        ref_img = reference_images[0].copy()
        ref_landmarks = reference_points[0]

        # Draw landmarks and connections on the reference image
        mp_draw.draw_landmarks(ref_img, ref_landmarks, mp_pose.POSE_CONNECTIONS)

        # Resize reference image to match the webcam frame height while maintaining aspect ratio
        height, width = frame.shape[:2]
        ref_height, ref_width = ref_img.shape[:2]
        scale = height / ref_height 
        new_width = int(ref_width * scale)
        new_height = height
        ref_img = cv2.resize(ref_img, (new_width, new_height))

    # Concatenate mirrored webcam frame and resized reference image side by side
    combined_frame = np.hstack((frame, ref_img))

    # Display the combined frame
    cv2.imshow("Webcam (Left) | Reference (Right)", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
