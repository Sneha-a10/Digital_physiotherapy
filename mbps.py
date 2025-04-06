import streamlit as st
import os
import cv2
import time
import numpy as np
from scipy.spatial import procrustes
import mediapipe as mp

class SportsTherapyApp:
    def __init__(self, exercises_dir, exercise_names):
        self.exercises_dir = exercises_dir
        self.exercise_names = exercise_names
        self.current_exercise_index = 0
        self.current_image_index = 0
        self.similarity_percentage = 0
        self.match_start_time = None
        self.reference_images = []
        self.reference_points = []

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks using bounding box normalization."""
        landmarks_array = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)

        # Get bounding box
        min_x, min_y = np.min(landmarks_array, axis=0)
        max_x, max_y = np.max(landmarks_array, axis=0)

        # Normalize to range [0,1]
        norm_landmarks = (landmarks_array - [min_x, min_y]) / (max_x - min_x, max_y - min_y)
        return norm_landmarks

    def calculate_similarity(self, user_points, ref_points):
        """Calculate similarity using Procrustes analysis."""
        if len(user_points) != len(ref_points):
            return 0.0

        # Apply Procrustes analysis to align both sets of points
        _, _, disparity = procrustes(user_points, ref_points)

        # Convert disparity to percentage similarity (lower disparity = better match)
        similarity = max(0, 1 - disparity)
        return similarity

    def load_exercise_data(self, exercise_path):
        """Load reference images and landmarks for the current exercise."""
        self.reference_images.clear()
        self.reference_points.clear()
        for img_name in sorted(os.listdir(exercise_path)):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(exercise_path, img_name)
                self.reference_images.append(img_path)

                # Load image and extract pose landmarks
                image = cv2.imread(img_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

                if results.pose_landmarks:
                    normalized_landmarks = self.normalize_landmarks(results.pose_landmarks.landmark)
                    self.reference_points.append(normalized_landmarks)

    def load_reference_image(self):
        exercise_path = os.path.join(self.exercises_dir, self.exercise_names[self.current_exercise_index])
        if not os.path.exists(exercise_path):
            st.error(f"Exercise folder not found: {exercise_path}")
            st.stop()

        self.load_exercise_data(exercise_path)
        if not self.reference_images:
            st.error(f"No images found in: {exercise_path}")
            st.stop()

        return self.reference_images[self.current_image_index]

    def layout_header(self):
        st.title("Sports Therapy Pose Evaluator")
        st.subheader("Evaluate and improve your form with precision-driven pose detection and feedback.")

    def layout_main_content(self, ref_image_path):
        col1, col2 = st.columns(2)

        # Left column: Reference Image
        with col1:
            ref_image_placeholder = st.empty()  # Placeholder for the reference image
            ref_image = cv2.imread(ref_image_path)
            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            ref_image_placeholder.image(ref_image_rgb, caption="Ideal Pose", use_container_width=True)

        # Right column: Live Webcam Feed with Pose Detection
        with col2:
            st.text("Your Pose")
            stframe = st.empty()  # Placeholder for the video feed
            feedback_placeholder = st.empty()  # Placeholder for feedback
            progress_placeholder = st.empty()  # Placeholder for the progress bar

            # Open the webcam
            cap = cv2.VideoCapture(0)  # 0 is the default webcam
            if not cap.isOpened():
                st.error("Unable to access the webcam.")
                return

            # Continuously capture frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Unable to read frame from webcam.")
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame and calculate similarity
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    user_normalized = self.normalize_landmarks(results.pose_landmarks.landmark)
                    ref_normalized = self.reference_points[self.current_image_index]
                    similarity = self.calculate_similarity(user_normalized, ref_normalized)
                    self.similarity_percentage = int(similarity * 100)

                    # Display feedback only when conditions change
                    if self.similarity_percentage >= 90:
                        feedback_placeholder.success("Perfect Match!")
                        time.sleep(1)  # Pause briefly to show feedback
                        self.current_image_index += 1  # Move to the next image
                        if self.current_image_index >= len(self.reference_images):
                            self.current_image_index = 0
                            self.current_exercise_index += 1
                            if self.current_exercise_index >= len(self.exercise_names):
                                st.success("Congratulations, you've completed all exercises!")
                                cap.release()
                                return
                            else:
                                # Load the next exercise
                                exercise_path = os.path.join(self.exercises_dir, self.exercise_names[self.current_exercise_index])
                                self.load_exercise_data(exercise_path)
                                ref_image_path = self.reference_images[self.current_image_index]
                                ref_image = cv2.imread(ref_image_path)
                                ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                                ref_image_placeholder.image(ref_image_rgb, caption="Ideal Pose", use_container_width=True)
                        else:
                            # Update the reference image for the current exercise
                            ref_image_path = self.reference_images[self.current_image_index]
                            ref_image = cv2.imread(ref_image_path)
                            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                            ref_image_placeholder.image(ref_image_rgb, caption="Ideal Pose", use_container_width=True)
                    elif self.similarity_percentage >= 80:
                        feedback_placeholder.warning("Good Effort! Adjust Slightly.")
                    else:
                        feedback_placeholder.error("Needs Improvement.")

                    # Update the progress bar dynamically
                    self.layout_progress_tracker(progress_placeholder)

                # Display the frame in the Streamlit app
                stframe.image(frame_rgb, caption="Live Webcam Feed", use_container_width=False)

                # Add a small delay to prevent high CPU usage
                time.sleep(0.03)

            cap.release()

    def layout_progress_tracker(self, progress_placeholder):
        total_exercises = len(self.exercise_names)
        if total_exercises == 0:
            st.error("No exercises available.")
            st.stop()

        # Calculate total images across all exercises
        total_images = sum(len(os.listdir(os.path.join(self.exercises_dir, ex))) for ex in self.exercise_names)
        completed_images = self.current_image_index + sum(
            len(os.listdir(os.path.join(self.exercises_dir, self.exercise_names[i])))
            for i in range(self.current_exercise_index)
        )

        # Calculate progress as a percentage
        progress = completed_images / total_images
        progress_placeholder.progress(progress)  # Update the progress bar
        progress_placeholder.text(f"Progress: {int(progress * 100)}%")

    def layout_feedback_section(self):
        if self.similarity_percentage >= 90:
            feedback_text = "Perfect Match!"
            feedback_color = "green"
        elif self.similarity_percentage >= 80:
            feedback_text = "Good Effort! Adjust Slightly"
            feedback_color = "yellow"
        else:
            feedback_text = "Needs Improvement"
            feedback_color = "red"
        st.markdown(f"<span style='color:{feedback_color};font-size:20px;'>{feedback_text}</span>", unsafe_allow_html=True)

    def layout_completion_popup(self):
        total_exercises = len(self.exercise_names)
        if self.current_exercise_index >= total_exercises and self.current_image_index >= len(self.reference_images):  # Adjust if needed
            st.success("Congratulations, you've completed your therapy session!")
            st.download_button("Download Progress Report", data="Your progress report data", file_name="progress_report.txt")

    def layout_footer(self):
        st.markdown("""
        ---
        **About** | **Contact** | **Privacy Policy**
        """)

    def run(self):
        if not self.exercise_names:
            st.error("No exercises found in the directory.")
            st.stop()

        ref_image_path = self.load_reference_image()
        self.layout_header()
        self.layout_main_content(ref_image_path)
        self.layout_feedback_section()
        self.layout_completion_popup()
        self.layout_footer()

# Example Usage:
exercises_dir = "exercises"
if not os.path.exists(exercises_dir):
    st.error(f"Exercises directory not found: {exercises_dir}")
else:
    exercise_names = sorted(os.listdir(exercises_dir))
    app = SportsTherapyApp(exercises_dir, exercise_names)
    app.run()