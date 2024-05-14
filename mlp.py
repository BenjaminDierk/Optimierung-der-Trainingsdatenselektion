
import torch
import torch.nn as nn
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset

# Definition des Modells
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
'''


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x


'''

def evaluate_model(model, test_data):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        # Daten auf das Gerät übertragen
        for inputs, labels in test_data:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy, all_labels, all_preds



# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('GPU gefunden:', torch.cuda.get_device_name(0))
else:
    print("Keine GPU gefunden. Bitte überprüfen Sie Ihre PyTorch-Installation.")



# Hyperparameter
input_size = 384 # Anzahl der Merkmale in Ihrem Vektor
hidden_size = 256 # Größe des versteckten Layers
num_classes = 5 # Anzahl der Klassen in Ihren Daten
learning_rate = 0.01
num_epochs = 200
# Festlegen der Batchgröße
batch_size = 32

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

class_counts = y_train.value_counts()
total_samples = len(y_train)
class_weights = torch.tensor([total_samples / count for count in class_counts], dtype=torch.float32)
class_weights = class_weights.to(device)


# Konvertiere pandas DataFrames in PyTorch-Tensoren
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)

# Aufteilen des Trainingsdatensatzes in Trainings- und Validierungsdatensätze
X_train, X_val, y_train, y_val = train_test_split(X_train_tensor, y_train_tensor, test_size=0.2, random_state=42)

# Erstellen von PyTorch-Datasets und -DataLoaders für Training und Validierung
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Konvertiere pandas DataFrames in PyTorch-Tensoren
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.int64)

# Erstelle ein Dataset und DataLoader für das Training
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Modellinitialisierung
model = MLP(input_size, hidden_size, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Trainingsschleife
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:  # Annahme: train_loader ist Ihr DataLoader für die Trainingsdaten
        # Daten auf das Gerät übertragen
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Nullen der Gradienten
        optimizer.zero_grad()
        
        # Vorwärtsdurchlauf
        outputs = model(inputs)
        
        # Berechnen des Verlusts
        loss = criterion(outputs, labels)
        
        # Rückwärtsdurchlauf und Optimierung
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Durchschnittlichen Verlust pro Epoch ausgeben
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
    # Annahme: Modell und Verlustfunktion bereits definiert

    # Setzen Sie das Modell in den Evaluationsmodus
    model.eval()

    # Initialisieren Sie Verlust und Genauigkeit
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Iterieren Sie über den Validierungsdaten-Lader
    for inputs, labels in val_loader:
        # Daten auf das Gerät übertragen
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Vorwärtsdurchlauf
        outputs = model(inputs)
        
        # Berechnen des Verlusts
        loss = criterion(outputs, labels)
        
        # Aktualisieren Sie den Gesamtverlust
        val_loss += loss.item() * inputs.size(0)
        
        # Berechnen Sie die Anzahl der korrekten Vorhersagen
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        
        # Aktualisieren Sie die Anzahl der gesamten Beispiele
        total_samples += labels.size(0)

    # Berechnen Sie den durchschnittlichen Verlust und die Genauigkeit für den Validierungsdatensatz
    average_val_loss = val_loss / total_samples
    accuracy = correct_predictions / total_samples

    # Geben Sie den durchschnittlichen Verlust und die Genauigkeit aus
    print(f'Validation Loss: {average_val_loss}, Accuracy: {accuracy}')



accuracy, all_labels, all_preds = evaluate_model(model, test_loader)

# Confusion Matrix berechnen
conf_matrix = confusion_matrix(all_labels, all_preds)

# Klassenbeschriftungen
class_labels = ['Klasse 1', 'Klasse 2', 'Klasse 3', 'Klasse 4', 'Klasse 5']

# Confusion Matrix plotten
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Vorhersagte Klasse')
plt.ylabel('Tatsächliche Klasse')
plt.title(f'Confusionsmatrix Dino upsampling Dataset MLP: Test accuracy: {accuracy}')
plt.show()

# Ergebnis der Confusionsmatrix in der Konsole ausgeben
print(f'Confusion Matrix: {conf_matrix}')
print(f'accuracy: {accuracy}')
