import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
import matplotlib.transforms as mtransforms


def load_images(folder):
    images = []
    file_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
        if img is not None:
            # Konvertiere das Bild in Graustufen
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Rezisiere das Bild auf die Größe 455x256
            img_resized = cv2.resize(img_gray, (455, 256))
            images.append(img_resized)
            file_names.append(filename)
    return np.array(images), file_names


def load_images_in_subfolders(main_folder):
    images = []
    file_names = []
    for sub_folder in os.listdir(main_folder):
        sub_folder_path = os.path.join(main_folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            for filename in os.listdir(sub_folder_path):
                img_path = os.path.join(sub_folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Lese das Bild in Graustufen ein
                if img is not None:
                    # Rezisiere das Bild auf die Größe 455x256
                    img_resized = cv2.resize(img, (455, 256))
                    images.append(img_resized)
                    file_names.append(filename)
    return np.array(images), file_names



def plot_images_3d(images, file_names, components=3):
    images_flat = images.reshape(images.shape[0], -1)
    U, s, Vt = np.linalg.svd(images_flat.T, full_matrices=False)
    pca = np.dot(images_flat, U[:, :components])

    # Erstellen der Liste der Datenpunkte mit Bildpfaden
    data_points = [(tuple(pca[i]), os.path.join(input_folder, file_names[i])) for i in range(len(file_names))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    
    def gety(x,y):
        # store the current mousebutton
        b = ax.button_pressed
        # set current mousebutton to something unreasonable
        ax.button_pressed = -1
        # get the coordinate string out
        s = ax.format_coord(x,y)
        # set the mousebutton back to its previous state
        ax.button_pressed = b
        return s

    def extract_coordinates(zdata):
        # Aufteilen der Zeichenkette an den Kommas
        parts = zdata.split(',')
        # Initialisieren der Variablen für die Koordinaten
        x, y, z = None, None, None
        # Durchlaufen der Teile und Extrahieren der Koordinaten
        
        for part in parts:
            part = part.strip()  # Entfernen von Leerzeichen am Anfang und Ende
            
            if 'x' in part:
                
                if '−'in part:
                    part = part.replace('−', '-')   
                x = float(part.split('=')[1])
           
            elif 'y' in part:
                
                if '−'in part:
                    part = part.replace('−', '-')    
                y = float(part.split('=')[1])
               
            elif 'z' in part:
                if '−'in part:
                    part = part.replace('−', '-')   
                
                z = float(part.split('=')[1])

        return x, y, z


    def on_click(event):
       
        if event.inaxes == ax and event.button == 3:  # Check if the click occurred in the axes and if it's a left-click
            
            coordinates = gety(event.xdata,event.ydata)
            x,y,z = extract_coordinates(coordinates)
            distances = [] 
          
            for coord, path in data_points:
                    dist = np.sqrt((x - coord[0])**2 + (y - coord[1])**2 + (z - coord[2])**2)
                    distances.append((dist, path))
                      
             # Sort the distances and get the 5 nearest neighbors
            distances.sort()
            nearest_neighbors = distances[:5]

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, (dist, path) in enumerate(nearest_neighbors):
                img = cv2.imread(path)
                img_resized = cv2.resize(img, (int(img.shape[1]*0.2), int(img.shape[0]*0.2)))  # Resize image
                axes[i].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                axes[i].set_title(f'Dist: {dist:.2f}')
                axes[i].axis('off')
            plt.show()


    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# Angeben des Ordners mit den Bildern
input_folder = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Datensatz unausgewogen viele Bilder pro Klasse'

# Laden der Bilder
images, file_names = load_images_in_subfolders(input_folder)

# Darstellung der Bilder in 3D mit PCA
plot_images_3d(images, file_names)


