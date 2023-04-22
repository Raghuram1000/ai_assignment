import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import TKinterModernThemes as TKMT
from PIL import Image, ImageTk

from imagedeblur import train_nnet


def open_image():
    filename = filedialog.askopenfilename()
    if filename:
        # open the image file and convert it to a Tkinter-compatible format
        image = Image.open(filename)
        image = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
        photo = ImageTk.PhotoImage(image)

        window = tk.Toplevel()
        # add label
        header_label = tk.Label(window, text="Deblurring image")
        header_label.pack()
        label = tk.Label(window, image=photo)
        progress = ttk.Progressbar(window, orient="horizontal", length=int(image.size[0] / 2), mode="determinate")
        progress["value"] = 50
        progress["maximum"] = 100

        label.pack()
        progress.pack()
        label.image = photo


def buttonCMD():
    print("Button clicked!")


def buttonCMD2(func):
    def wrapper():
        func()

    return wrapper


class App(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True):
        super().__init__(str("Blur removal"), theme, mode, usecommandlineargs, usethemeconfigfile)
        self.button_frame = self.addLabelFrame(str("Main Functions"))  # placed at row 1, col 0
        self.button_frame.Button(str("Train Neural Network"), buttonCMD2(lambda: train_nnet()))
        self.button_frame.Button(str("Deblur an image"), open_image)
        self.button_frame_2 = self.addLabelFrame(str("Analysis"))  # placed at row 1, col 0
        self.button_frame_2.Button(str("Run Search algorithms"), buttonCMD)
        self.button_frame_2.Button(str("Compare search algorithms"), buttonCMD)
        self.debugPrint()
        self.run()


if __name__ == "__main__":
    App(str("park"), str("dark"))
