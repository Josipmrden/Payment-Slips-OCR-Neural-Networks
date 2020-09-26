from uplatnica import finalCode
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile
from uplatnica.finalCode import Uplatnica
from keras.models import load_model

num = './models/model2.hdf5'
num_let = './models/konacna_slova.hdf5'

models = [load_model(num), load_model(num_let)]


root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

panel = tk.Label(frame, image = None)
panel.pack()
text = tk.Text(frame, borderwidth = 2, relief = "groove")
text.pack()
text.insert(tk.END, "Podaci nisu uneseni")
text.config(state = tk.DISABLED)
filename = ""

def getImage():
    str1 = ".png"
    str2 = ".jpg"

    global filename
    filename = askopenfile().name

    index = filename.rfind('.')
    extension = filename[index:].lower()
    if extension != str1 and extension != str2:
        return
    else:
        pil_image = Image.open(filename)
        resized = pil_image.resize((400, 200))
        img = ImageTk.PhotoImage(resized)
        panel.configure(image = img)
        panel.image = img
        prepoznaj.config(state = tk.NORMAL)


def recognizeImage():
    global models
    global text
    img_name = filename
    slip = finalCode.getSlipDetails(img_name, models)
    text.config(state=tk.NORMAL)
    text.delete('1.0', tk.END)
    text.insert('1.0', slip.toString())
    text.config(state=tk.DISABLED)


dodaj = tk.Button(frame, text = "Dodaj uplatnicu", command = getImage)
dodaj.pack()

prepoznaj = tk.Button(frame, text = "Prepoznaj", command = recognizeImage)
prepoznaj.pack()
prepoznaj.config(state = tk.DISABLED)



root.mainloop()