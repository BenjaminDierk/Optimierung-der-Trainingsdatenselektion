from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Pfade zu den CSV-Dateien
file_paths_train = [
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_train\reduced_image_vectors_1.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_train\reduced_image_vectors_2.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_train\reduced_image_vectors_3.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_train\reduced_image_vectors_4.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_train\reduced_image_vectors_5.csv'
]

# Labels zuweisen und DataFrames zusammenführen
train_dfs_with_labels = []
for idx, file in enumerate(file_paths_train):
    df = pd.read_csv(file, header=None)
    df['label'] = idx
    train_dfs_with_labels.append(df)

print(train_dfs_with_labels)

# Alle DataFrames zusammenführen
all_train_features = pd.concat(train_dfs_with_labels, ignore_index=True)

# Merkmalsmatrix extrahieren
X_train = all_train_features.drop('label', axis=1)
y_train = all_train_features['label']


# Pfade zu den CSV-Dateien
file_paths_test = [
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_test\reduced_image_vectors_1.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_test\reduced_image_vectors_2.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_test\reduced_image_vectors_3.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_test\reduced_image_vectors_4.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\pca_vec_test\reduced_image_vectors_5.csv'
]


# Labels zuweisen und DataFrames zusammenführen
test_dfs_with_labels = []
for idx, file in enumerate(file_paths_test):
    df = pd.read_csv(file, header=None)
    df['label'] = idx
    test_dfs_with_labels.append(df)

print(test_dfs_with_labels)
# Alle DataFrames zusammenführen
all_test_features = pd.concat(test_dfs_with_labels, ignore_index=True)

# Merkmalsmatrix extrahier
X_test = all_test_features.drop('label', axis=1)
y_test = all_test_features['label']


num_samples = X_train.shape[0]
print("Anzahl der Datenpunkte:", num_samples)

k = 200

# KNN-Modell initialisieren und trainieren
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Vorhersagen für das Testset machen
y_pred = knn.predict(X_test)

# Genauigkeit bewerten
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Genauigkeit: {accuracy}')

# Berechnen Sie die Anzahl der korrekt und falsch klassifizierten Beispiele
correct = np.sum(y_test == y_pred)
incorrect = len(y_test) - correct

# Confusion Matrix berechnen
conf_matrix = confusion_matrix(y_test, y_pred)

# Klassenbeschriftungen
class_labels = ['Klasse 1', 'Klasse 2', 'Klasse 3', 'Klasse 4', 'Klasse 5']

# Confusion Matrix plotten
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Vorhersagte Klasse')
plt.ylabel('Tatsächliche Klasse')
plt.title(f'Confusionsmatrix PCA upsampling Dataset KNN: K = 200: Test accuracy: {accuracy}')
plt.show()

# Ergebnis der Confusionsmatrix in der Konsole ausgeben
print("Confusion Matrix:")
print(conf_matrix)