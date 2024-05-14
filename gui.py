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


class GUI():
    
    # Graphical user Interface class to rate pictures

    def __init__(self):
        
        #initialize username, the opcua client class, the Window amd widgets of the GUI
        #Also paramters to define window size image size and so 
        self.pictureservice = self.PictureServer()
        self.lock = threading.Lock()
        self.score_data = pd.DataFrame(columns=['Index','Date','image_name', 'oen', 'image_set', 'pg', 'score'])
        if os.path.exists(self.pictureservice.rated_pictures) and os.path.getsize(self.pictureservice.rated_pictures) > 0:
            self.index = len(pd.read_csv(self.pictureservice.rated_pictures))
        else:
            with open(self.pictureservice.rated_pictures, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.score_data)
            self.index = 1
        background_thread_image_queue = threading.Thread(target=self.pictureservice.fill_image_queue, daemon=True)
        background_thread_image_queue.start()

        self.score_1 = 1
        self.score_2 = 2
        self.score_3 = 3
        self.score_4 = 4
        self.score_5 = 5

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
        button_width = int(screen_width * 0.005)
        button_height = int(screen_height * 0.005)
        
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)
    
        start_image = r'Instructions.jpg'
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
        dir_names = os.path.split(dir_names[0])
        img_set = dir_names[1]
        dir_names = os.path.split(dir_names[0])
        img_oen = dir_names[1]
        
        row = self.pictureservice.df['OEN'] == img_oen
        img_pg = self.pictureservice.df.loc[row, 'productGroup'].iloc[0]

        row = [self.index,datetime.now().strftime('%d/%m/%Y-%H:%M%S'), img_name, img_oen, img_set,
                   img_pg, self.score_1]

        if all(row):
            self.score_data = pd.concat([self.score_data, pd.DataFrame([row], columns=self.score_data.columns)])
            with open(self.pictureservice.rated_pictures, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            self.index += 1
        
        img_path = self.pictureservice.image_queue.get()
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
        dir_names = os.path.split(dir_names[0])
        img_set = dir_names[1]
        dir_names = os.path.split(dir_names[0])
        img_oen = dir_names[1]
        
        row = self.pictureservice.df['OEN'] == img_oen
        img_pg = self.pictureservice.df.loc[row, 'productGroup'].iloc[0]

        row = [self.index,datetime.now().strftime('%d/%m/%Y-%H:%M%S'), img_name, img_oen, img_set,
                   img_pg, self.score_2]

        if all(row):
            self.score_data = pd.concat([self.score_data, pd.DataFrame([row], columns=self.score_data.columns)])
            with open(self.pictureservice.rated_pictures, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            self.index += 1
        
        img_path = self.pictureservice.image_queue.get()
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
        dir_names = os.path.split(dir_names[0])
        img_set = dir_names[1]
        dir_names = os.path.split(dir_names[0])
        img_oen = dir_names[1]
        
        row = self.pictureservice.df['OEN'] == img_oen
        img_pg = self.pictureservice.df.loc[row, 'productGroup'].iloc[0]

        row = [self.index,datetime.now().strftime('%d/%m/%Y-%H:%M%S'), img_name, img_oen, img_set,
                   img_pg, self.score_3]

        if all(row):
            self.score_data = pd.concat([self.score_data, pd.DataFrame([row], columns=self.score_data.columns)])
            with open(self.pictureservice.rated_pictures, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            self.index += 1


        img_path = self.pictureservice.image_queue.get()
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
        dir_names = os.path.split(dir_names[0])
        img_set = dir_names[1]
        dir_names = os.path.split(dir_names[0])
        img_oen = dir_names[1]
        
        row = self.pictureservice.df['OEN'] == img_oen
        img_pg = self.pictureservice.df.loc[row, 'productGroup'].iloc[0]

        row = [self.index,datetime.now().strftime('%d/%m/%Y-%H:%M%S'), img_name, img_oen, img_set,
                   img_pg, self.score_4]

        if all(row):
            self.score_data = pd.concat([self.score_data, pd.DataFrame([row], columns=self.score_data.columns)])
            with open(self.pictureservice.rated_pictures, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            self.index += 1

        img_path = self.pictureservice.image_queue.get()
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
        dir_names = os.path.split(dir_names[0])
        img_set = dir_names[1]
        dir_names = os.path.split(dir_names[0])
        img_oen = dir_names[1]
        
        row = self.pictureservice.df['OEN'] == img_oen
        img_pg = self.pictureservice.df.loc[row, 'productGroup'].iloc[0]

        row = [self.index,datetime.now().strftime('%d/%m/%Y-%H:%M%S'), img_name, img_oen, img_set,
                   img_pg, self.score_5]

        if all(row):
            self.score_data = pd.concat([self.score_data, pd.DataFrame([row], columns=self.score_data.columns)])
            with open(self.pictureservice.rated_pictures, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            self.index += 1


        img_path = self.pictureservice.image_queue.get()
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
        # Speichere die Daten am Ende deines Programms
        with open('data_dict.json', 'w') as datei:
            json.dump(self.pictureservice.data_dict, datei)

        self.root.destroy()

    class PictureServer:

        #Class Provides images in a balanced way. Image view and productgroups are provided in a balanced way.   

        def __init__(self):

            self.path_to_oens = r'Z:\DataSets\EIBA\01_Verlesedaten\03_DataSet\Part Number'
            self.df = pd.read_csv(r'\\syno2.ipk.fraunhofer.de\oe420_lab-visionlab\DataSets\EIBA\01_Verlesedaten\02_MetaDaten\data_objects\prodGroupData.csv')
            self.rated_pictures = r'C:\Users\ben61519\Desktop\Optimierung der Trainingsdatenselektion\GUI\rated_pictures.csv'
            self.pgs = self.df['productGroup'].unique()
            self.image_queue = queue.Queue()
            self.image_queue_length = 6
            self.lock = threading.Lock()

                # Prüfe, ob das Dictionary verfügbar ist
            if os.path.exists('data_dict.json'):
            	# Lade das vorhandene Dictionary
                with open('data_dict.json', 'r') as datei:
                    self.data_dict = json.load(datei)

            else:
                # Erstelle das Dictionary, wenn es nicht vorhanden ist
                print('dict_data nicht gefunden. Erstelle neues Dictionary')
                self.data_dict = {}
                self.data_dict = self.erstelle_Dict(self.path_to_oens, self.pgs, self.df)


        def erstelle_Dict(self, path_to_oens, List_of_all_Pg, df):
    
            data_dict = {}  # Das Dictionary, in dem alle Informationen gespeichert werden
            existing_oen_folders = [entry.name for entry in os.scandir(path_to_oens) if entry.is_dir()]

            # Schreibe alle PGs ins Dictionary und initialisiere bool und Zählvariablen
            for pg in List_of_all_Pg:
                
                data_dict[pg] = {'used': False, 'count': 0, 'OENs': {}}
                search_oen_list = [oen for oen in df.loc[df['productGroup'] == pg, 'OEN'].unique()]
                existing_oen_numbers = [oen for oen in search_oen_list if oen in existing_oen_folders]
            
                for oen_folder in existing_oen_numbers:    
                    oen_path = os.path.join(path_to_oens, oen_folder)

                    # Überprüfe, ob es sich um ein Verzeichnis handelt
                    if os.path.isdir(oen_path):
                        data_dict[pg]['OENs'][oen_folder] = {'used': False, 'Sets': {}}

                        # Finde alle Sets und füge sie zum Dictionary hinzu
                        for set_folder in os.scandir(oen_path):
                            # Gehe durch alle SET-Ordner
                            set_path = os.path.join(oen_path, set_folder)
                            
                            
                            if os.path.isdir(set_path):
                                data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name] = {'used': False, 'Bilder': {}}
                                
                                for bild in os.listdir(set_path):
                                    if 'Part' in bild and not 'mask' in bild and bild.endswith('.jpg'):
                                        data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name]['Bilder'][bild] = {'used': False}

                                # Wenn das Set keine Bilder enthält, setze es auf False
                                if not data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name]['Bilder']:
                                    data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name]['used'] = True

                            # Wenn der OEN-Ordner keine Sets enthält, setze ihn auf True
                            if not data_dict[pg]['OENs'][oen_folder]:
                                data_dict[pg]['OENs'][oen_folder]['used'] = True

            return data_dict

        
        
        def add_images_to_queue(self, data_dict):

            with self.lock:
                # Sortiere die Produktgruppen nach der count-Variable aufsteigend
                sorted_pgs = sorted([pg for pg in self.pgs if not data_dict.get(pg, {}).get('used', True)], key=lambda pg: data_dict[pg]['count'])
                
                if not sorted_pgs:
                    return  # Alle Produktgruppen sind bereits als 'used' markiert oder es gibt keine Produktgruppen
                
                pg = sorted_pgs[0]  
                pg_data = data_dict[pg]
                
                if not pg_data['OENs'] or all(oen_data['used'] for oen_data in pg_data['OENs'].values()):
                    data_dict[pg]['used'] = True
                    return
                
                oens_mit_unused_pg = [oen_name for oen_name, oen_data in data_dict[pg]['OENs'].items() if not oen_data['used']]
                
                if not oens_mit_unused_pg:
                    data_dict[pg]['used'] = True
                    return  # Alle OENs in der ausgewählten Produktgruppe sind bereits als 'used' markiert
                
                oen_name = random.choice(oens_mit_unused_pg)
                sets_in_oen = [set_name for set_name, set_data in data_dict[pg]['OENs'][oen_name]['Sets'].items() if not set_data['used']]
                
                if not sets_in_oen:
                    data_dict[pg]['OENs'][oen_name]['used'] = True
                    return  # Alle Sets in der ausgewählten OEN sind bereits als 'used' markiert
                
                set_name = random.choice(sets_in_oen)

                for image_name, image_data in data_dict[pg]['OENs'][oen_name]['Sets'][set_name]['Bilder'].items():
                    if ('Top' in image_name or 'Left' in image_name or 'Right' in image_name) and image_name.endswith('jpg'):
                        image_path = os.path.join(self.path_to_oens, oen_name, set_name, image_name)
                        self.image_queue.put(image_path)
                        data_dict[pg]['OENs'][oen_name]['Sets'][set_name]['Bilder'][image_name]['used'] = True


                data_dict[pg]['OENs'][oen_name]['Sets'][set_name]['used'] = True
                data_dict[pg]['count'] += 1
                
                return
            
        def fill_image_queue(self):

            while True:
            
                if self.image_queue.qsize() < self.image_queue_length:
                    self.add_images_to_queue(self.data_dict)
                else:
                    # Wenn die Queue voll ist, warte eine Weile und prüfe erneut
                    time.sleep(5)  # Wartezeit in Sekunden
                if all(self.data_dict[pg]['used'] for pg in self.data_dict):
                    break

            

if __name__ == '__main__':

    graphic_user_interface = GUI()
    graphic_user_interface.root.mainloop()

    
