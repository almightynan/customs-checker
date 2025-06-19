import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import time
from PIL import ImageFont

WHITE_TONES = ['#f5e7da', '#f1c8a6', '#e8b183']
BLACK_TONES = ['#8d5524', '#7d3f1d', '#4b2e1e']

def get_average_skin_color_hex(frame, face_rect):
    x, y, w, h = face_rect
    face_roi = frame[y:y+h, x:x+w]
    face_ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(face_ycrcb, lower, upper)
    skin_pixels = cv2.bitwise_and(face_roi, face_roi, mask=skin_mask)
    skin_rgb = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2RGB)
    non_black = skin_rgb[np.all(skin_rgb != [0, 0, 0], axis=2)]
    if len(non_black) == 0:
        return None, None
    avg_color = np.mean(non_black, axis=0).astype(int)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*avg_color)
    luminance = (0.2126*avg_color[0] + 0.7152*avg_color[1] + 0.0722*avg_color[2]) / 255
    return hex_color, luminance

def luminance_to_class(lum):
    if lum > 0.6:
        return "White"
    else:
        return "Black"

class SkinColorCustomsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Border Control")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)

        self.face_detected_time = None
        self.classification_done = False

        self.home_frame = tk.Frame(root, bg="#1e1e1e")
        self.home_frame.pack(fill="both", expand=True)

        self.title_label = tk.Label(self.home_frame, text="Border Control", font=("Arial", 24, "bold"), fg="white", bg="#1e1e1e")
        self.title_label.pack(pady=(50, 10))

        self.desc_label = tk.Label(self.home_frame, text="""This app verifies identities at borders using facial data. It matches live images to official records through 
biometric analysis, ensuring "fair" and accurate results across all demographics. Built with global privacy 
standards in mind, it enables secure, fast, and unbiased border control.""", font=("Arial", 14), fg="white", bg="#1e1e1e")
        self.desc_label.pack(pady=(0, 30))

        self.start_button = ttk.Button(self.home_frame, text="Start Now", command=self.start_detection)
        self.start_button.pack()

        self.detection_frame = tk.Frame(root, bg="#121212")

        self.video_label = tk.Label(self.detection_frame, bg="black")
        self.video_label.place(x=0, y=0, width=720, height=600)

        self.info_frame = tk.Frame(self.detection_frame, bg="#222222")
        self.info_frame.place(x=720, y=0, width=280, height=600)

        self.status_label = tk.Label(self.info_frame, text="Thinking...", fg="white", bg="#222222", font=("Arial", 14))
        self.status_label.pack(pady=20)

        self.hint_label = tk.Label(self.info_frame, text="", fg="gray", bg="#222222", font=("Arial", 10))
        self.hint_label.pack(pady=(5, 10))

        self.class_label = tk.Label(self.info_frame, text="", fg="white", bg="#222222", font=("Arial", 16, "bold"))
        self.class_label.pack(pady=10)

        self.palette_container = tk.Frame(self.info_frame, bg="#222222")
        self.palette_container.pack(pady=20)

        self.white_label = tk.Label(self.palette_container, text="Allowed", bg="#222222", fg="white", font=("Arial", 12, "bold"))
        self.white_label.grid(row=0, column=0, pady=5)
        self.black_label = tk.Label(self.palette_container, text="Not allowed", bg="#222222", fg="white", font=("Arial", 12, "bold"))
        self.black_label.grid(row=0, column=1, pady=5)

        self.white_palette = [tk.Label(self.palette_container, bg=col, width=8, height=2) for col in WHITE_TONES]
        self.black_palette = [tk.Label(self.palette_container, bg=col, width=8, height=2) for col in BLACK_TONES]

        for i in range(3):
            self.white_palette[i].grid(row=i+1, column=0, padx=10, pady=2)
            self.black_palette[i].grid(row=i+1, column=1, padx=10, pady=2)

        self.retake_button = ttk.Button(self.info_frame, text="Retake Test", command=self.retake_test)
        self.retake_button.pack(pady=30)
        self.retake_button.configure(state='disabled')

        self.cap = None
        self.detecting = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def start_detection(self):
        self.home_frame.pack_forget()
        self.detection_frame.pack(fill="both", expand=True)
        self.cap = cv2.VideoCapture(0)
        self.detecting = True
        self.face_detected_time = None
        self.classification_done = False
        self.status_label.config(text="Thinking...")
        self.hint_label.config(text="")
        self.class_label.config(text="")
        self.retake_button.configure(state='disabled')
        self.update_frame()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def retake_test(self):
        self.face_detected_time = None
        self.classification_done = False
        self.status_label.config(text="Thinking...")
        self.hint_label.config(text="")
        self.class_label.config(text="")
        self.retake_button.configure(state='disabled')
        self.cap = cv2.VideoCapture(0)
        self.detecting = True
        # Restore black background to video label on retake
        self.video_label.config(bg="black")
        self.update_frame()

    def show_popup(self, classification):
        popup = tk.Toplevel(self.root)
        popup.geometry("700x180")
        popup.resizable(False, False)
        popup.title("Check Complete")

        # Center the popup
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (700 // 2)
        y = (popup.winfo_screenheight() // 2) - (180 // 2)
        popup.geometry(f"+{x}+{y}")

        label = tk.Label(popup, font=("Arial", 14, "bold"), fg="black")
        label.pack(expand=True, pady=20)

        def close():
            popup.destroy()

        close_btn = ttk.Button(popup, text="Close", command=close)
        close_btn.pack(pady=10)

        if classification == "White":
            popup.configure(bg="green")
            label.config(text="Great news! You're in the allowed zone. Please progress to the next step.", bg="green")
        else:
            label.config(text="Suspicious profile detected, the security team has been notified.\nSTAY WHERE YOU ARE AND DON'T MOVE.", bg="#440000")

            colors = ["#DD0000", "#0000FF"]
            def flash(i=0):
                if not popup.winfo_exists():
                    return
                popup.configure(bg=colors[i % 2])
                label.configure(bg=colors[i % 2])
                popup.after(500, flash, (i + 1) % 2)
            flash()

    def update_frame(self):
        if not self.detecting:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0 and not self.classification_done:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self.face_detected_time is None:
                self.face_detected_time = time.time()

            elapsed = time.time() - self.face_detected_time
            self.status_label.config(text=f"Thinking... {int(elapsed)}s")
            self.hint_label.config(text="Hold still and look directly at the camera.")

            if elapsed >= 3:
                hex_color, luminance = get_average_skin_color_hex(frame, (x, y, w, h))
                if hex_color:
                    skin_class = luminance_to_class(luminance)
                    self.status_label.config(text="Test Complete")
                    self.hint_label.config(text="")
                    self.class_label.config(text=f"Class: {skin_class}")
                    self.retake_button.configure(state='normal')
                    self.classification_done = True
                    self.detecting = False
                    self.stop_camera()

                    # Show solid gray in video display area to pause camera view
                    gray_img = Image.new('RGB', (720, 600), color='gray')
                    draw = ImageDraw.Draw(gray_img)

                    title = f"Check Complete"

                    # Use a larger font for the text
                    try:
                        font = ImageFont.truetype("arial.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    draw.text((250, 200), title, fill="black", font=font)

                    imgtk = ImageTk.PhotoImage(gray_img)
                    self.video_label.config(image=imgtk)
                    self.video_label.imgtk = imgtk
                    self.root.update_idletasks()
                    self.show_popup(skin_class)
        else:
            self.face_detected_time = None
            if not self.classification_done:
                self.status_label.config(text="Face Not Detected")
                self.hint_label.config(text="Ensure good lighting and look at the camera.")

        if not self.classification_done:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((720, 600))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        if self.detecting:
            self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = SkinColorCustomsApp(root)
    root.mainloop()
