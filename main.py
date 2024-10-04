import time
import tkinter as tk
import cv2
from utils import *
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import winsound

# Initialise camera & mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()  # class has default values, check documentation


class App:
    def __init__(self):
        self.main_window = tk.Tk(className="Pose Corrector")
        self.main_window.geometry("1200x520+350+100")

        # Calibrate button
        self.calibrate_button = get_button(
            window=self.main_window,
            text='Start calibration',
            color="green",
            fg="white",
            command=self.calibrate
        )
        self.calibrate_button.place(x=750, y=200)

        # Webcam label for displaying frames
        self.webcam_label = get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        # Capture webcam video feed
        self.cap = cv2.VideoCapture(0)

        # Flags for calibration and posture tracking
        self.calibrating = False
        self.is_calibrated = False
        self.num_frames = 0

        # Calibration data
        self.calib_angle_nose_left = []
        self.calib_angle_nose_right = []
        self.calib_angle_at_nose = []
        self.calib_angle_eye = []

        # Timer for incorrect posture
        self.incorrect_posture_start_time = None
        self.incorrect_posture_threshold = 10  # seconds threshold for beep

        # Start webcam updates
        self.update_webcam()

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip the frame for mirrored effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose estimation using Mediapipe
            result = pose.process(rgb_frame)
            if result.pose_landmarks:
                self.process_landmarks(frame, result.pose_landmarks)

            # Convert image to Tkinter compatible format
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)

        # Call this function again after 20 ms to update the webcam feed
        self.main_window.after(20, self.update_webcam)

    def process_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        landmarks = landmarks.landmark

        # Extracting key body parts (nose, shoulders, ears)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]  # mirrored image
        right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]  # mirrored image
        left_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]  # mirrored image
        right_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]  # mirrored image

        # Get pixel coordinates
        nose_coords = (int(nose.x * w), int(nose.y * h))
        left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_ear_coords = (int(left_ear.x * w), int(left_ear.y * h))
        right_ear_coords = (int(right_ear.x * w), int(right_ear.y * h))

        # Calculate angles
        angle_nose_left = calculate_angle(nose_coords, left_shoulder_coords, right_shoulder_coords)
        angle_nose_right = calculate_angle(nose_coords, right_shoulder_coords, left_shoulder_coords)
        angle_eye = calculate_angle(left_ear_coords, nose_coords, right_ear_coords)
        angle_shoulder = calculate_angle(right_shoulder_coords, left_shoulder_coords,
                                         (right_shoulder_coords[0], left_shoulder_coords[1]))

        if not self.is_calibrated and self.num_frames < 30 and self.calibrating:
            self.calib_angle_nose_left.append(angle_nose_left)
            self.calib_angle_nose_right.append(angle_nose_right)
            self.calib_angle_eye.append(angle_eye)
            self.num_frames += 1
            cv2.putText(frame, f'Calibrating {self.num_frames}/30', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
        elif self.num_frames >= 30:
            self.is_calibrated = True

        # Check if the posture is incorrect
        if self.is_calibrated:
            self.check_posture(frame, angle_nose_left, angle_nose_right, angle_eye, angle_shoulder,
                               nose_coords, left_shoulder_coords, right_shoulder_coords,
                               left_ear_coords, right_ear_coords)

    def check_posture(self, frame, angle_nose_left, angle_nose_right, angle_eye, angle_shoulder,
                      nose_coords, left_shoulder_coords, right_shoulder_coords, left_ear_coords, right_ear_coords):

        calibrated_angle_nose_left = int(np.mean(self.calib_angle_nose_left))
        calibrated_angle_nose_right = int(np.mean(self.calib_angle_nose_right))
        calibrated_angle_eye = int(np.mean(self.calib_angle_eye))

        # Display angles
        cv2.putText(frame, f"{int(angle_nose_left)}", (left_shoulder_coords[0] + 20,
                left_shoulder_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"{int(angle_nose_right)}", (right_shoulder_coords[0] - 50,
                right_shoulder_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"{int(angle_eye)}", (nose_coords[0] - 5, nose_coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw line between shoulders
        cv2.line(frame, left_shoulder_coords, right_shoulder_coords, (0, 255, 0), 2)
        # Check if angles are within the calibrated range
        if (angle_nose_left not in range(calibrated_angle_nose_left - 10, calibrated_angle_nose_left + 10) or
                angle_nose_right not in range(calibrated_angle_nose_right - 10, calibrated_angle_nose_right + 10) or
                angle_eye not in range(calibrated_angle_eye - 10, calibrated_angle_eye + 10) or angle_shoulder > 5):

            draw_triangle(frame, nose_coords, left_shoulder_coords, right_shoulder_coords,
                          left_ear_coords, right_ear_coords, (0, 0, 255))

            # Start the timer for incorrect posture
            if self.incorrect_posture_start_time is None:
                self.incorrect_posture_start_time = time.time()

            elapsed_time = time.time() - self.incorrect_posture_start_time
            if elapsed_time > self.incorrect_posture_threshold:
                # Play beep sound if posture is incorrect for more than 10 seconds
                winsound.Beep(1000, 1000)

        else:
            draw_triangle(frame, nose_coords, left_shoulder_coords, right_shoulder_coords,
                          left_ear_coords, right_ear_coords, (0, 255, 0))
            # Reset the timer
            self.incorrect_posture_start_time = None

    def calibrate(self):
        # Start calibration
        self.calibrating = True

    def start(self):
        self.main_window.mainloop()


if __name__ == '__main__':
    app = App()
    app.start()
