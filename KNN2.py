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
train_data_class1 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec train\train_feature_vectors_1.csv')
train_data_class2 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec train\train_feature_vectors_2.csv')
train_data_class3 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec train\train_feature_vectors_3.csv')
train_data_class4 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec train\train_feature_vectors_4.csv')
train_data_class5 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec train\train_feature_vectors_5.csv')

# Daten laden und vorbereiten
test_data_class1 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec test\test_feature_vectors_1.csv')
test_data_class2 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec test\test_feature_vectors_2.csv')
test_data_class3 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec test\test_feature_vectors_3.csv')
test_data_class4 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec test\test_feature_vectors_4.csv')
test_data_class5 = pd.read_csv(r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec test\test_feature_vectors_5.csv')


# Merkmalsvektoren extrahieren und in DataFrames umwandeln
train_features_1 = pd.DataFrame(train_data_class1['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
train_features_2 = pd.DataFrame(train_data_class2['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
train_features_3 = pd.DataFrame(train_data_class3['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
train_features_4 = pd.DataFrame(train_data_class4['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
train_features_5 = pd.DataFrame(train_data_class5['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())

# Merkmalsvektoren extrahieren und in DataFrames umwandeln
test_features_1 = pd.DataFrame(test_data_class1['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
test_features_2 = pd.DataFrame(test_data_class2['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
test_features_3 = pd.DataFrame(test_data_class3['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
test_features_4 = pd.DataFrame(test_data_class4['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())
test_features_5 = pd.DataFrame(test_data_class5['Feature_1'].apply(lambda x: ast.literal_eval(x)).tolist())


# Labels zuweisen und DataFrames zusammenführen
for idx, features in enumerate([train_features_1, train_features_2, train_features_3, train_features_4, train_features_5]):
    features['label'] = idx

# Labels zuweisen und DataFrames zusammenführen
for idx, features in enumerate([test_features_1, test_features_2, test_features_3, test_features_4, test_features_5]):
    features['label'] = idx

# Alle DataFrames zusammenführen
all_train_features = pd.concat([train_features_1, train_features_2, train_features_3, train_features_4, train_features_5], ignore_index=True)
# Alle DataFrames zusammenführen
all_test_features = pd.concat([test_features_1, test_features_2, test_features_3, test_features_4, test_features_5], ignore_index=True)

# Merkmalsmatrix und Labels trennen
X_train = all_train_features.drop('label', axis=1)
y_train = all_train_features['label']


# Merkmalsmatrix und Labels trennen
X_test = all_test_features.drop('label', axis=1)
y_test = all_test_features['label']

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
plt.title(f'Confusionsmatrix Dino upsampling Dataset KNN: K = 10: Test accuracy: {accuracy}')
plt.show()

# Ergebnis der Confusionsmatrix in der Konsole ausgeben
print("Confusion Matrix:")
print(conf_matrix)