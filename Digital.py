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
        landmarks = results.pose_landmarks.landmark
        points = [(lm.x, lm.y) for lm in landmarks]
        reference_images.append(img)
        reference_points.append(points)

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

    # Draw landmarks on the mirrored webcam frame
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        webcam_points = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

    # Prepare reference image and points
    if reference_images:
        ref_img = reference_images[0].copy()
        ref_points = reference_points[0]
        for point in ref_points:
            cv2.circle(ref_img, (int(point[0] * ref_img.shape[1]), int(point[1] * ref_img.shape[0])), 5, (0, 255, 0), -1)

        # Resize reference image to match webcam frame height
        ref_img = cv2.resize(ref_img, (frame.shape[1], frame.shape[0]))

    # Concatenate mirrored webcam frame and reference image side by side
    combined_frame = np.hstack((frame, ref_img))

    # Display the combined frame
    cv2.imshow("Webcam (Left) | Reference (Right)", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
