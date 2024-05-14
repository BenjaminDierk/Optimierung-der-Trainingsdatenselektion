import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def resize_image_proportionally(image, new_height):
    # Bestimme das Seitenverhältnis des Originalbildes
    height, width = image.shape[:2]
    aspect_ratio = width / height
    
    # Berechne die Breite entsprechend dem neuen Seitenverhältnis
    new_width = int(new_height * aspect_ratio)
    
    # Verwende cv2.resize, um das Bild proportional zu verkleinern
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image

def resize_to_original_size(image, original_size):
    # Verwende die Höhe und Breite des Originalbildes
    original_height, original_width = original_size
    
    # Verwende cv2.resize, um das Bild auf die Größe des Originalbildes zu skalieren
    resized_image = cv2.resize(image, (original_width, original_height))
    
    return resized_image


def main():

   # Verzeichnis mit den Originalbildern
    original_dir = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\resize bilder MSE\original images'
    # Verzeichnis zum Speichern der verkleinerten Bilder
    resized_dir = r'C:\Users\benni\Desktop\Humanoide_Robotik\Bachlorarbeit\Bachlorarbeit\resize bilder MSE\resized images'

    # Liste der Bildernamen im Originalverzeichnis
    image_names = os.listdir(original_dir)

    # Durchschnittlicher SSIM über alle Bilder
    total_ssim = 0

    # Listen zum Speichern der SSIM-Werte und der Bildnummern
    ssim_values = []
    image_numbers = []

    # Iteriere über die Bilder
    for idx, image_name in enumerate(image_names, start=1):
        # Lade das Originalbild
        original_image = cv2.imread(os.path.join(original_dir, image_name), cv2.IMREAD_GRAYSCALE)
        print(original_image.shape)
        
        original_height, original_width = original_image.shape[:2]

        # Verkleinere das Bild proportional auf eine Höhe von 256 Pixel
        resized_image = resize_image_proportionally(original_image, 256)
        print(resized_image.shape)

        resized_image_resized = resize_to_original_size(resized_image, (original_height, original_width))

        # Berechne den SSIM
        image_ssim = ssim(original_image, resized_image_resized)
        total_ssim += image_ssim

        ssim_values.append(image_ssim)
        image_numbers.append(idx)

        print(f"SSIM für {image_name}: {image_ssim}")

        # Speichere das verkleinerte Bild
        cv2.imwrite(os.path.join(resized_dir, image_name), resized_image_resized)

    # Durchschnittlicher SSIM über alle Bilder
    average_ssim = total_ssim / len(image_names)
    print(f"Durchschnittlicher SSIM über alle Bilder: {average_ssim}")

    # Plotte den Graphen
    plt.plot(image_numbers, ssim_values, marker='o')
    plt.xlabel('Bildnummer')
    plt.ylabel('SSIM-Wert')
    plt.title('SSIM-Werte für verkleinerte Bilder(256 x 455)')
    plt.ylim(0, 1)  # Skaliere die Y-Achse auf den Bereich von 0 bis 1
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()





 

