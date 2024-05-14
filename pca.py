import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


def load_and_convert_images(folder_path, target_shape):
    image_vectors = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            # Laden des RGB-Bildes
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # Ändern der Größe des Bildes
            resized_image = cv2.resize(image, target_shape[::-1])  # target_shape ist (width, height)
            # Bild als 3D-Array darstellen (Höhe x Breite x Kanäle)
            image_array = np.array(resized_image)
            image_vectors.append(image_array)
            
    return np.array(image_vectors)

# Beispielaufruf
folder_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\3'
target_shape = (227, 455)
image_arrays = load_and_convert_images(folder_path, target_shape)

# Reshape der Bilder zu einem 2D-Array
num_images = image_arrays.shape[0]
image_shape = image_arrays.shape[1:3]  # (Höhe, Breite)
num_channels = image_arrays.shape[3]  # Anzahl der Farbkanäle

# Reshape der Bilder zu einem 2D-Array, wobei jeder Pixel oder Farbkanal eine Spalte ist
image_vectors = image_arrays.reshape(num_images, -1)
print(image_vectors.shape)
'''
# PCA durchführen und die Daten reduzieren
pca = PCA(n_components=200, svd_solver='randomized', random_state=42)  # Anpassen Sie die Anzahl der Hauptkomponenten nach Bedarf an
pca_image_vectors = pca.fit_transform(image_vectors)
print(pca_image_vectors.shape)


# Konvertieren der reduzierten Bildvektoren in ein DataFrame
df = pd.DataFrame(pca_image_vectors)

# Speichern des DataFrames in einer CSV-Datei
'''
#csv_file_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_train\reduced_image_vectors_1.csv'

#df.to_csv(csv_file_path, index=False)



pca = PCA().fit(image_vectors)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plotten der kumulativen erklärten Varianz
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()




