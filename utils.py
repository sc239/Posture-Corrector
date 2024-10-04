import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    # a, b, c are tuples representing the coordinates of three points
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Second point (e.g., elbow)
    c = np.array(c)  # Third point (e.g., wrist)

    # Calculate the vectors BA and BC
    ba = a - b
    bc = c - b

    # Calculate the dot product and the magnitudes of the vectors
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Calculate the angle and convert it to degrees
    angle = np.degrees(np.arccos(cosine_angle))

    return np.floor(angle)


def draw_triangle(frame, nose_coords, left_shoulder_coords, right_shoulder_coords, left_ear_coords, right_ear_coords, color):
    cv2.line(
        img=frame,
        pt1=nose_coords,
        pt2=left_shoulder_coords,
        thickness=2,
        color=color
    )
    cv2.line(
        img=frame,
        pt1=nose_coords,
        pt2=right_shoulder_coords,
        thickness=2,
        color=color
    )
    cv2.line(
        img=frame,
        pt1=right_shoulder_coords,
        pt2=left_shoulder_coords,
        thickness=2,
        color=color
    )
    cv2.line(
        img=frame,
        pt1=left_ear_coords,
        pt2=nose_coords,
        thickness=2,
        color=color
    )
    cv2.line(
        img=frame,
        pt1=right_ear_coords,
        pt2=nose_coords,
        thickness=2,
        color=color
    )
    cv2.line(
        img=frame,
        pt1=left_shoulder_coords,
        pt2=(right_shoulder_coords[0], left_shoulder_coords[1]),
        thickness=2,
        color=(255,255,255)
    )
    # Draw line between shoulders
    cv2.line(
        img=frame,
        pt1=left_shoulder_coords,
        pt2=right_shoulder_coords,
        color=color,
        thickness=2
    )


# Function to calculate the start angle of the arc based on the position of the lines
def calculate_start_angle(a, b):
    delta_y = a[1] - b[1]
    delta_x = a[0] - b[0]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def addWebcam(self, label):
    if 'cap' not in self.__dict__:
        self.cap = cv2.VideoCapture(0)

    self._label = label
    self.processWebcam()

def processWebcam(self):
    ret, frame = self.cap.read()

    self.most_recent_frame = frame
    img_ = cv2.cvtColor(self.most_recent_frame, cv2.COLOR_BGR2RGB)
    self.most_recent_capture_pil = Image.fromarray(img_)
    imgtk = ImageTk.PhotoImage(
        image=self.most_recent_capture_pil
    )
    self._label.imgtk = imgtk
    self._label.configure(image=imgtk)

    self._label.after(20, self.processWebcam)