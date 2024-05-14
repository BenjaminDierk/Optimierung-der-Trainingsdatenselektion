import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

# Pfad zum Trainingsdatensatz
image_dir = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\train_dir\2'

# Definieren der Transformationen für die Datenaugmentation
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Zufällige Rotation um bis zu 10 Grad
    transforms.RandomHorizontalFlip(),      # Zufällige horizontale Spiegelung
    transforms.RandomVerticalFlip(),        # Zufällige vertikale Spiegelung
    transforms.ToTensor()                   # Konvertierung in Tensor
])

# Zähler für die transformierten Bilder
counter = 0

# Durchlaufen aller Bilder im Ordner
for image_file in os.listdir(image_dir):
    # Überprüfen, ob es sich um eine Bilddatei handelt
    if image_file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        image_path = os.path.join(image_dir, image_file)
        
        # Lesen des Originalbildes
        original_image = Image.open(image_path)
        
        # Erstellen von drei Kopien des Originalbildes
        for i in range(3):
            # Kopie des Originalbildes erstellen
            transformed_image = original_image.copy()
            
            # Anwenden der Transformationen
            transformed_image = transform(transformed_image)
            
            # Speichern des transformierten Bildes mit eindeutigem Dateinamen
            save_image(transformed_image, os.path.join(image_dir, f'{counter}_{image_file}'))
            
            # Inkrementieren des Zählers
            counter += 1

print("Transformationen wurden für alle Bilder angewendet und gespeichert.")





