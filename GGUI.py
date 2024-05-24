import tkinter as tk
from tkinter import filedialog
from PIL import Image , ImageTk
import cv2
import main




def detect_license_plate() :
    file_path = filedialog.askopenfilename()

    if file_path :
        image = cv2.imread(file_path)

        resized_image = cv2.resize(image , (640 , 640))

        processed_image , license_plate_text , license_plate_crop  = main.detect_from_image(image)

        resized_image = cv2.cvtColor(resized_image , cv2.COLOR_BGR2RGB)
        resized_image_pil = Image.fromarray(resized_image)
        resized_photo = ImageTk.PhotoImage(resized_image_pil)


        image_label.config(image=resized_photo)
        image_label.image = resized_photo


        license_plate_crop_resized = cv2.resize(license_plate_crop , (640 , 320))
        license_plate_crop_resized = cv2.cvtColor(license_plate_crop_resized , cv2.COLOR_BGR2RGB)
        license_plate_crop_pil = Image.fromarray(license_plate_crop_resized)
        license_plate_crop_photo = ImageTk.PhotoImage(license_plate_crop_pil)

        license_plate_label.config(image=license_plate_crop_photo)
        license_plate_label.image = license_plate_crop_photo

        result_label.config(text=license_plate_text)


window = tk.Tk()
window.title("License Plate Detection")


upload_title = tk.Label(window , text="Upload Image" , font=("Arial" , 14 , "bold"))
upload_title.pack()

upload_frame = tk.Frame(window)
upload_frame.pack(side=tk.LEFT , padx=10 , pady=10)

result_title = tk.Label(window , text="Detection Result" , font=("Arial" , 14 , "bold"))
result_title.pack()

result_frame = tk.Frame(window)
result_frame.pack(side=tk.RIGHT , padx=10 , pady=10)

upload_label = tk.Label(upload_frame , text="Upload Image:")
upload_label.pack()

upload_button = tk.Button(upload_frame , text="Upload" , command=detect_license_plate)
upload_button.pack()

image_label = tk.Label(upload_frame)
image_label.pack()

license_plate_label = tk.Label(result_frame)
license_plate_label.pack()

result_label = tk.Label(result_frame , text="" , font=("Arial" , 40) , pady=20)
result_label.pack()

window.mainloop()
