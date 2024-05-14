
'''
Scripttitel: Picture label system
Hersteller: Benjamin Dierk
Datum: 15.04.2023
'''

import time
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk
import io
import csv
import json
import pandas as pd
import os
import queue
import copy
import random
import threading
import shutil


class GUI():
    
    # Graphical user Interface class to rate pictures

    def __init__(self):
        
        #initialize username, the opcua client class, the Window amd widgets of the GUI
        #Also paramters to define window size image size and so 
        self.pictureservice = self.PictureServer()
        self.pictureservice.load_images_to_queue()
        self.lock = threading.Lock()
        self.destination_folder_1 = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\1'
        self.destination_folder_2 = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\2'
        self.destination_folder_3 = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\3'
        self.destination_folder_4 = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\4'
        self.destination_folder_5 = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\5'


        self.root = tk.Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.8)
        image_ratio = 1280 / 720
        self.image_width = int(screen_width * 0.6)
        self.image_height = int(self.image_width / image_ratio)
        
        self.root.title("Image rating interface")
        self.root.geometry(f"{window_width}x{window_height}")
        
        widget_width = int(window_width * 0.7)
        widget_height = int(window_height * 1)
        button_width = int(screen_width * 0.007)
        button_height = int(screen_height * 0.007)
        
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)
    
        start_image = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\andere\Instructions.jpg'
        self.old_image_path = start_image
        self.image = Image.open(start_image)
        self.image = self.image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)
        self.label = tk.Label(self.root, image=self.photo,width=widget_width, height=widget_height)
        self.label.grid(row=0, column=0,rowspan=7, columnspan=7)
        
        self.button_exit = tk.Button(self.root, text="Exit",command=self.exit_program,width= button_width, height=button_height)
        self.button_start = tk.Button(self.root, text="Start",command=self.on_button_start_click,width=button_width, height=button_height)
        self.button_score_5 = tk.Button(self.root, text="Mangelhaft",command=self.on_button_5_click,width=button_width, height=button_height)
        self.button_score_4 = tk.Button(self.root, text="Ausreichend",command=self.on_button_4_click,width=button_width, height=button_height)
        self.button_score_3 = tk.Button(self.root, text="Befriedigend",command=self.on_button_3_click,width=button_width, height=button_height)
        self.button_score_2 = tk.Button(self.root, text="Gut",command=self.on_button_2_click,width=button_width, height=button_height)
        self.button_score_1 = tk.Button(self.root, text="Sehr gut",command=self.on_button_1_click, width=button_width, height=button_height)
        self.explanation_1 = tk.Button(self.root, text="Hilfe",command=self.on_questionmark_button_click,width=button_width, height=button_height)

        # Spacer-Spalte erstellen
        spacer = tk.Frame(self.root, width=100)
        spacer.grid(row=0, column=8, rowspan=7)
        spacer.grid(row=0, column=9, rowspan=7)
        spacer.grid(row=0, column=12, rowspan=7)
        
        self.button_exit.grid(row=3, column=13)
        self.explanation_1.grid(row=2, column=13)
        self.button_start.grid(row=1, column=13)
        
        self.button_score_5.grid(row=5, column=10)
        self.button_score_4.grid(row=4, column=10)
        self.button_score_3.grid(row=3, column=10)
        self.button_score_2.grid(row=2, column=10)
        self.button_score_1.grid(row=1, column=10)
        

        
        # Bind the event handler to the button
        self.root.bind('1', self.on_button_1_click)
        self.root.bind('2', self.on_button_2_click)
        self.root.bind('3', self.on_button_3_click)
        self.root.bind('4', self.on_button_4_click)
        self.root.bind('5', self.on_button_5_click)


    
        # Define event handlers (Buttons that you click)
    def on_button_start_click(self):
        self.new_img = self.pictureservice.image_queue.get()
        self.old_image_path = self.new_img
        self.image = Image.open(self.new_img)
        self.image = self.image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)
        self.label = tk.Label(self.root, image=self.photo)
        self.label.grid(row=0, column=0,rowspan=7, columnspan=7)



    #Giving the picture that appears in the middle of the gui the score 1
    #After a new image gets loaded  
    def on_button_1_click(self,event=None):
        
        self.label.config(text="sehr gut")
        dir_names = os.path.split(self.old_image_path)
        img_name = dir_names[1]
        shutil.move(self.old_image_path, os.path.join(self.destination_folder_1, img_name))

        try:
            img_path = self.pictureservice.image_queue.get(block=False)
        except queue.Empty:
            self.root.destroy()

        self.old_image_path = img_path    
        
        image = Image.open(img_path)
        image = image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        new_photo = ImageTk.PhotoImage(image)
        self.label.image = new_photo
        self.label.config(image=new_photo)

        
    #Giving the picture that appears in the middle of the gui the score 2
    #After a new image gets loaded 
    def on_button_2_click(self,event=None):
    
        self.label.config(text="gut")
        dir_names = os.path.split(self.old_image_path)
        img_name = dir_names[1]
        shutil.move(self.old_image_path, os.path.join(self.destination_folder_2, img_name))

        try:
            img_path = self.pictureservice.image_queue.get(block=False)
        except queue.Empty:
            self.root.destroy()

        self.old_image_path = img_path    
        
        image = Image.open(img_path)
        image = image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        new_photo = ImageTk.PhotoImage(image)
        self.label.image = new_photo
        self.label.config(image=new_photo)
   
    #Giving the picture that appears in the middle of the gui the score 3
    #After a new image gets loaded 
    def on_button_3_click(self,event=None):
       
        self.label.config(text="befriedigend")
        dir_names = os.path.split(self.old_image_path)
        img_name = dir_names[1]
        shutil.move(self.old_image_path, os.path.join(self.destination_folder_3, img_name))

        try:
            img_path = self.pictureservice.image_queue.get(block=False)
        except queue.Empty:
            self.root.destroy()

        self.old_image_path = img_path    
        
        image = Image.open(img_path)
        image = image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        new_photo = ImageTk.PhotoImage(image)
        self.label.image = new_photo
        self.label.config(image=new_photo)
        
    #Giving the picture that appears in the middle of the gui the score 4
    #After a new image gets loaded 
    def on_button_4_click(self,event=None):
        
        
        self.label.config(text="ausreichend")
        dir_names = os.path.split(self.old_image_path)
        img_name = dir_names[1]
        shutil.move(self.old_image_path, os.path.join(self.destination_folder_4, img_name))

        try:
            img_path = self.pictureservice.image_queue.get(block=False)
        except queue.Empty:
            self.root.destroy()

        self.old_image_path = img_path    
        
        image = Image.open(img_path)
        image = image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        new_photo = ImageTk.PhotoImage(image)
        self.label.image = new_photo
        self.label.config(image=new_photo)
        
    #Giving the picture that appears in the middle of the gui the score 5
    #After a new image gets loaded 
    def on_button_5_click(self,event=None):
        
        self.label.config(text="mangelhaft")
        dir_names = os.path.split(self.old_image_path)
        img_name = dir_names[1]
        shutil.move(self.old_image_path, os.path.join(self.destination_folder_5, img_name))

        try:
            img_path = self.pictureservice.image_queue.get(block=False)
        except queue.Empty:
            self.root.destroy()

        self.old_image_path = img_path    
        
        image = Image.open(img_path)
        image = image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        new_photo = ImageTk.PhotoImage(image)
        self.label.image = new_photo
        self.label.config(image=new_photo)



    #Questionmark Buttons Explaining how to rate the pictures.      
    
    def on_questionmark_button_click(self):
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Hilfe")
        
        help_text = (
            "Score 1 : Objekt aus allen Perspektiven sehr gut erkennbar und ausgelichtet, Wenig Clutter, keine Occlusion\n\n"
            "Score 2 : Gute Ausleuchtung, wenig Occlusion, Scharfe Abbildung, z.B. anderer Hintergrund, Occlusion < 10 %\n\n"
            "Score 3 : Überstrahlung, wenig Kontrast (dunkles Objekt), Occlusion < 25 % ok, solange wichtige Merkmale passen, Oberfläche eingeschränkt erkennbar\n\n"
            "Score 4 : Objekt sichtbar aber Artefakte beeinflussen das Bild, > 50 % sichtbar, Oberflächenstruktur ist nicht erkennbar\n\n"
            "Score 5 : Man könnte was lernen aber die KI muss es nicht erkennen können. Nutzerfehler < 50 % zu sehen"
        )
        
        label = tk.Label(help_window, text=help_text, justify="left")
        label.pack(padx=10, pady=10)

    
    #Button to leave the application. Either leave with exit Button or Cross in up top left corner. 
    def exit_program(self):
        self.root.destroy()

    class PictureServer: 

        def __init__(self):

            self.image_queue = queue.Queue()

        def load_images_to_queue(self):
            
            folder_path = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Beispiel Bilder zum bewerten\3'
            # Überprüfen, ob der Ordner existiert
            if not os.path.isdir(folder_path):
                print("Der angegebene Pfad ist kein Ordner.")
                return None

            # Durchlaufen aller Dateien im Ordner
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # Sicherstellen, dass es sich um eine Bilddatei handelt
                if any(file_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    # Pfad zur Warteschlange hinzufügen
                    self.image_queue.put(file_path)  

if __name__ == '__main__':

    graphic_user_interface = GUI()
    graphic_user_interface.root.mainloop()
