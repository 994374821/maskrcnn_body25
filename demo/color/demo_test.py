#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:27:36 2019

@author: gaomingda
"""

from PIL import Image as pilImage
from PIL import ImageTk
import numpy as np
from tkinter import *

#setting up a tkinter canvas with scrollbars
root = Tk()
frame = Frame(root, bd=2, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

canvas = Canvas(frame,width=320,height=400)
canvas.grid(row=0,column=0)

frame.pack(fill=BOTH,expand=1)

png_name = '/home/gaomingda/Documents/maskrcnn_fcn/demo/color/color_map.jpg'
#color_png = pilImage.open('/home/gaomingda/Documents/maskrcnn_fcn/demo/color/color_map.jpg').resize((320,100))
#filename = ImageTk.PhotoImage(color_png)
##canvas.image = filename  # <--- keep reference of your image
#canvas.create_image(0,0,anchor='nw',image=filename)
#
#color_png1 = pilImage.open('/home/gaomingda/Documents/maskrcnn_fcn/demo/lip_val/4527_428354.jpg').resize((320,100))
#filename1 = ImageTk.PhotoImage(color_png1)
#canvas.create_image(0,200,anchor='nw',image=filename1)

def printcoords():
#    png_name = '/home/gaomingda/Documents/maskrcnn_fcn/demo/color/color_map.jpg'
    global filename
    global color_png
    color_png = pilImage.open('/home/gaomingda/Documents/maskrcnn_fcn/demo/lip_val/4527_428354.jpg').resize((320,100))
    filename = ImageTk.PhotoImage(color_png)
#    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(0,0,anchor='nw',image=filename)
    
def choose():
    global color_png1
    global filename1
    color_png1 = pilImage.open(png_name).resize((320,100))
    filename1 = ImageTk.PhotoImage(color_png1)
#    canvas.image = filename1 
    canvas.create_image(0,200,anchor='nw',image=filename1)

b = Button(root, text='printcoords',command=printcoords).pack(side=LEFT, padx=50)
#b.grid(row=0,column=0)
b1 = Button(root, text='choose',command=choose).pack(side=LEFT)
#b1.grid(row=0,column=0)

root.mainloop()