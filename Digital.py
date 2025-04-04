import cv2
import mediapipe as mp
import os
import numpy as np
from scipy.spatial import procrustes
import tkinter as tk
from tkinter import messagebox

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Path to the photo folder
exercises_dir = "exercises"
exercise_names = sorted(os.listdir(exercises_dir))  # List of exercise folders
current_exercise_index = 0
reference_images = []
reference_points = []

# Function to normalize landmarks using bounding box normalization
def normalize_landmarks(landmarks):
    landmarks_array = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)

    # Get bounding box
    min_x, min_y = np.min(landmarks_array, axis=0)
    max_x, max_y = np.max(landmarks_array, axis=0)

    # Normalize to range [0,1]
    norm_landmarks = (landmarks_array - [min_x, min_y]) / (max_x - min_x, max_y - min_y)
    
    return norm_landmarks

# Function to calculate similarity using Procrustes analysis
def calculate_similarity(user_points, ref_points):
    if len(user_points) != len(ref_points):
        return 0.0

    # Apply Procrustes analysis to align both sets of points
    mtx1, mtx2, disparity = procrustes(user_points, ref_points)

    # Convert disparity to percentage similarity (lower disparity = better match)
    similarity = max(0, 1 - disparity)  
    return similarity

# Extract points from images in the photo folder
def load_exercise_data(exercise_path):
    reference_images.clear()
    reference_points.clear()
    for img_name in sorted(os.listdir(exercise_path)):
        if not img_name.endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(exercise_path, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            reference_images.append(img)
            reference_points.append(normalize_landmarks(results.pose_landmarks.landmark))

# Load the first exercise
load_exercise_data(os.path.join(exercises_dir, exercise_names[current_exercise_index]))

# Initialize variables for tracking progress
current_image_index = 0
match_start_time = None

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
    if results.pose_landmarks and reference_points:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Normalize user points
        user_normalized = normalize_landmarks(results.pose_landmarks.landmark)
        ref_normalized = reference_points[current_image_index]

        # Calculate similarity
        similarity = calculate_similarity(user_normalized, ref_normalized)
        similarity_percentage = int(similarity * 100)

        if similarity >= 0.90:  
            if match_start_time is None:
                match_start_time = cv2.getTickCount()  
            else:
                elapsed_time = (cv2.getTickCount() - match_start_time) / cv2.getTickFrequency()
                if elapsed_time >= 5:  
                    current_image_index += 1
                    match_start_time = None  
                    if current_image_index >= len(reference_images):
                        current_exercise_index += 1
                        if current_exercise_index >= len(exercise_names):
                            cap.release()
                            cv2.destroyAllWindows()
                            
                            root = tk.Tk()
                            root.withdraw()  
                            messagebox.showinfo("Session Complete", "YOU COMPLETED THE SESSION")
                            root.destroy()
                            exit()
                        else:
                            # Load the next exercise
                            load_exercise_data(os.path.join(exercises_dir, exercise_names[current_exercise_index]))
                            current_image_index = 0 
        else:
            match_start_time = None  

        # if similarity >= 0.9:
        #     cv2.putText(frame, "✅ 100% Matched! Perfect Position!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # elif similarity >= 0.80:
        #     cv2.putText(frame, "✔️ Good Job! Adjust Slightly", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # else:
        #     cv2.putText(frame, "⚠️ Adjust Your Pose!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display similarity percentage
    text = f"Match: {similarity_percentage}%"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 30 + text_height), (255, 255, 255), -1)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the exercise name
    exercise_name = exercise_names[current_exercise_index].replace("_", " ").capitalize()
    cv2.putText(frame, f"Exercise: {exercise_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # Prepare reference image

    if reference_images:
        ref_img = reference_images[current_image_index].copy()

        # Resize reference image to match the webcam frame height
        height, width = frame.shape[:2]
        ref_img = cv2.resize(ref_img, (width, height))

        # Concatenate mirrored webcam frame and resized reference image side by side
        combined_frame = np.hstack((frame, ref_img))

        # Display the combined frame
        cv2.imshow("Webcam (Left) | Reference (Right)", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
exit()
