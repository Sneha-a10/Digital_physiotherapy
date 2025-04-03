import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

# Load YOLOv8 Pose Model
model = YOLO("yolov8n-pose.pt")
photo_dir = "photo"
exercises = []
for img_name in sorted(os.listdir(photo_dir)):  # Use processed_dir instead of photo
    img_path = os.path.join(photo_dir, img_name)  # Use processed_dir here as well
    exercises.append((img_name.split('.')[0], img_path))

current_exercise = 0
current_step = 0  # 0 for first image, 1 for second image

# Sample reference poses (Replace with actual keypoints)
reference_poses = [
    [(100, 200), (120, 220), (140, 250)],  # Step 1 keypoints for Exercise 1
    [(110, 210), (130, 230), (150, 260)],  # Step 2 keypoints for Exercise 1
]

# Tkinter GUI setup
root = tk.Tk()
root.title("Digital Physio Trainer")

# Labels
exercise_label = Label(root, text=f"Current Exercise: {exercises[current_exercise][0]} - Step {current_step + 1}", font=("Arial", 16))
exercise_label.pack()

feedback_label = Label(root, text="Do the pose!", font=("Arial", 14), fg="blue")
feedback_label.pack()

# Exercise reference images
canvas = Canvas(root, width=600, height=200)
canvas.pack()

# Webcam frame
video_label = Label(root)
video_label.pack()

# Open webcam
cap = cv2.VideoCapture(0)

def show_reference_images():
    """Displays the correct reference image for the current step."""
    img_path = exercises[current_exercise][1] if current_step == 0 else exercises[current_exercise][2]
    img = Image.open(img_path)
    img = ImageTk.PhotoImage(img.resize((300, 200)))
    canvas.create_image(300, 100, anchor=tk.CENTER, image=img)
    canvas.image = img  # Prevent garbage collection

def is_pose_correct(user_pose, reference_pose, threshold=10):
    """Checks if the user’s pose matches the reference pose and returns incorrect keypoints."""
    if len(user_pose) != len(reference_pose):
        return False, []

    incorrect_points = []
    for i in range(len(user_pose)):
        if np.linalg.norm(np.array(user_pose[i]) - np.array(reference_pose[i])) > threshold:
            incorrect_points.append(i)

    return len(incorrect_points) == 0, incorrect_points

def update_frame():
    """Updates the webcam feed and checks for correct posture."""
    global current_exercise, current_step

    ret, frame = cap.read()
    if not ret:
        return

    results = model(frame)

    for result in results:
        keypoints = result.keypoints.xy.tolist()  # Extract detected keypoints

        correct, incorrect_points = is_pose_correct(keypoints, reference_poses[current_step])

        if correct:
            feedback_label.config(text="✅ Correct Pose! Moving to Next Step...", fg="green")
            root.after(1000)
            current_step += 1

            if current_step > 1:  # If both steps are completed
                current_step = 0
                current_exercise += 1
                if current_exercise >= len(exercises):
                    feedback_label.config(text="🎉 Workout Complete!", fg="purple")
                    cap.release()
                    return
            
            exercise_label.config(text=f"Current Exercise: {exercises[current_exercise][0]} - Step {current_step + 1}")
            show_reference_images()
        else:
            feedback_label.config(text=f"⚠️ Incorrect! Adjust points: {incorrect_points}", fg="red")

    # Draw pose estimation overlay
    frame = results[0].plot()

    # Convert frame to Tkinter-compatible format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    root.after(10, update_frame)  # Refresh frame

# Start the live feed
show_reference_images()
update_frame()
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
