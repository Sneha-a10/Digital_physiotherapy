import streamlit as st
import os
import cv2
import time
from Digital import calculate_similarity, normalize_landmarks, load_exercise_data

class SportsTherapyApp:
    def __init__(self, exercises_dir, exercise_names):
        self.exercises_dir = exercises_dir
        self.exercise_names = exercise_names
        self.current_exercise_index = 0
        self.current_image_index = 0
        self.similarity_percentage = 0
        self.match_start_time = None

    def load_reference_image(self):
        exercise_path = os.path.join(self.exercises_dir, self.exercise_names[self.current_exercise_index])
        if not os.path.exists(exercise_path):
            st.error(f"Exercise folder not found: {exercise_path}")
            st.stop()

        image_files = sorted([img for img in os.listdir(exercise_path) if img.endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            st.error(f"No images found in: {exercise_path}")
            st.stop()

        ref_image_path = os.path.join(exercise_path, image_files[self.current_image_index])
        return ref_image_path

    def layout_header(self):
        st.title("Sports Therapy Pose Evaluator")
        st.subheader("Evaluate and improve your form with precision-driven pose detection and feedback.")

    def layout_main_content(self, ref_image_path):
        col1, col2 = st.columns(2)

        # Left column: Reference Image
        with col1:
            st.image(ref_image_path, caption="Ideal Pose", use_container_width=True)

        # Right column: Live Webcam Feed with Pose Detection
        with col2:
            st.text("Your Pose")
            stframe = st.empty()  # Placeholder for the video feed

            # Open the webcam
            cap = cv2.VideoCapture(0)  # 0 is the default webcam
            if not cap.isOpened():
                st.error("Unable to access the webcam.")
                return

            # Load exercise data
            exercise_path = os.path.join(self.exercises_dir, self.exercise_names[self.current_exercise_index])
            load_exercise_data(exercise_path)

            # Continuously capture frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Unable to read frame from webcam.")
                    break

                # Process the frame and calculate similarity
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    user_normalized = normalize_landmarks(results.pose_landmarks.landmark)
                    ref_normalized = reference_points[self.current_image_index]
                    similarity = calculate_similarity(user_normalized, ref_normalized)
                    similarity_percentage = int(similarity * 100)

                    # Display feedback
                    if similarity_percentage >= 90:
                        st.success("Perfect Match!")
                    elif similarity_percentage >= 80:
                        st.warning("Good Effort! Adjust Slightly.")
                    else:
                        st.error("Needs Improvement.")

                # Display the frame in the Streamlit app
                stframe.image(frame, caption="Live Webcam Feed", use_container_width=True)

                # Add a small delay to prevent high CPU usage
                time.sleep(0.03)

            cap.release()

    def layout_progress_tracker(self):
        total_exercises = len(self.exercise_names)
        if total_exercises == 0:
            st.error("No exercises available.")
            st.stop()

        progress = (self.current_image_index + 1) / 5  # Assuming 5 images per exercise
        st.progress(progress)
        st.text(f"Exercise {self.current_exercise_index + 1} of {total_exercises}")

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
        if self.current_exercise_index >= total_exercises and self.current_image_index >= 5:  # Adjust if needed
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
        self.layout_progress_tracker()
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