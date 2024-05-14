from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Daten laden und vorbereiten
data_class1 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\unausgewogen dino\feature_vectors_1.csv')
data_class2 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\unausgewogen dino\feature_vectors_2.csv')
data_class3 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\unausgewogen dino\feature_vectors_3.csv')
data_class4 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\unausgewogen dino\feature_vectors_4.csv')
data_class5 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\unausgewogen dino\feature_vectors_5.csv')

# Merkmalsvektoren extrahieren und in DataFrames umwandeln
features_1 = pd.DataFrame(data_class1['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
features_2 = pd.DataFrame(data_class2['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
features_3 = pd.DataFrame(data_class3['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
features_4 = pd.DataFrame(data_class4['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
features_5 = pd.DataFrame(data_class5['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())

# Labels zuweisen und DataFrames zusammenführen
for idx, features in enumerate([features_1, features_2, features_3, features_4, features_5], 1):
    features['label'] = idx

# Alle DataFrames zusammenführen
all_features = pd.concat([features_1, features_2, features_3, features_4, features_5], ignore_index=True)

# Die ersten Zeilen des zusammengeführten DataFrames ausgeben
print(all_features.head())

# Merkmalsmatrix und Labels trennen
X = all_features.drop('label', axis=1)
y = all_features['label']


# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_samples = X.shape[0]
print("Anzahl der Datenpunkte:", num_samples)

k = 10

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
plt.title(f'Confusionsmatrix Dino unbalanced Dataset KNN: K = 10:  Test accuracy: {accuracy}')
plt.show()

# Ergebnis der Confusionsmatrix in der Konsole ausgeben
print("Confusion Matrix:")
print(conf_matrix)