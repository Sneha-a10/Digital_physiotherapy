import streamlit as st
import os  # Import the os module

class SportsTherapyApp:
    def __init__(self, exercises_dir, exercise_names):
        self.exercises_dir = exercises_dir
        self.exercise_names = exercise_names
        self.current_exercise_index = 0
        self.current_image_index = 0
        self.similarity_percentage = 0
        self.match_start_time = None

    def load_reference_image(self):
        exercise_path = f"{self.exercises_dir}/{self.exercise_names[self.current_exercise_index]}"
        image_files = sorted([img for img in os.listdir(exercise_path) if img.endswith(('.png', '.jpg', '.jpeg'))])
        ref_image_path = f"{exercise_path}/{image_files[self.current_image_index]}"
        return ref_image_path

    def layout_header(self):
        st.title("Sports Therapy Pose Evaluator")
        st.subheader("Evaluate and improve your form with precision-driven pose detection and feedback.")

    def layout_main_content(self, ref_image_path):
        col1, col2 = st.columns(2)

        # Left column: Reference Image
        with col1:
            st.image(ref_image_path, caption="Ideal Pose", use_column_width=True)

        # Right column: Webcam Feed (placeholder for video integration)
        with col2:
            st.text("Your Pose")
            st.video("your_webcam_feed.mp4")  # Replace with actual webcam feed integration

    def layout_progress_tracker(self):
        progress = (self.current_image_index + 1) / 5  # Assuming 5 exercises
        st.progress(progress)
        st.text(f"Exercise {self.current_exercise_index + 1} of {len(self.exercise_names)}")

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
        if self.current_exercise_index >= len(self.exercise_names) and self.current_image_index >= 5:  # Adjust if needed
            st.success("Congratulations, you've completed your therapy session!")
            st.download_button("Download Progress Report", data="Your progress report data", file_name="progress_report.txt")

    def layout_footer(self):
        st.markdown("""
        ---
        **About** | **Contact** | **Privacy Policy**
        """)

    def run(self):
        ref_image_path = self.load_reference_image()
        self.layout_header()
        self.layout_main_content(ref_image_path)
        self.layout_progress_tracker()
        self.layout_feedback_section()
        self.layout_completion_popup()
        self.layout_footer()

# Example Usage:
exercises_dir = "exercises"
exercise_names = sorted(os.listdir(exercises_dir))  
app = SportsTherapyApp(exercises_dir, exercise_names)
app.run()