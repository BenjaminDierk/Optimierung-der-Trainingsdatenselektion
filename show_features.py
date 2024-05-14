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
            images.append(img)
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
       
        if event.inaxes == ax and event.button == 1:  # Check if the click occurred in the axes and if it's a left-click
            
            coordinates = gety(event.xdata,event.ydata)
            x,y,z = extract_coordinates(coordinates)
           
            if x is not None and y is not None:    
                min_dist = np.inf
                min_path = None

            for coord, path in data_points:
                    dist = np.sqrt((x - coord[0])**2 + (y - coord[1])**2 + (z - coord[2])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        min_path = path

            if min_dist:
                    img = cv2.imread(min_path)
                    img_resized = cv2.resize(img, (227, 128))  # Ändern Sie die Größe des Bildes
                    cv2.imshow('Bild', img_resized)
                    cv2.moveWindow('Bild', 100, 100)  # Verschieben des Fensters
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()


    fig.canvas.mpl_connect('button_press_event', on_click)
    

    plt.show()

# Angeben des Ordners mit den Bildern
input_folder = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\resize bilder MSE\original images'

# Laden der Bilder
images, file_names = load_images(input_folder)

# Darstellung der Bilder in 3D mit PCA
plot_images_3d(images, file_names)

