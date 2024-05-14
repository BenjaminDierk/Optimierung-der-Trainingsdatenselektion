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
input_size = 300 # Anzahl der Merkmale in Ihrem Vektor
hidden_size = 128 # Größe des versteckten Layers
num_classes = 5 # Anzahl der Klassen in Ihren Daten
learning_rate = 0.001
num_epochs = 200
# Festlegen der Batchgröße
batch_size = 16

# Pfade zu den CSV-Dateien
file_paths = [
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\downsampling pca\reduced_image_vectors_1.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\downsampling pca\reduced_image_vectors_2.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\downsampling pca\reduced_image_vectors_3.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\downsampling pca\reduced_image_vectors_4.csv',
    r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\csv files\downsampling pca\reduced_image_vectors_5.csv'
]

# Labels zuweisen und DataFrames zusammenführen
dfs_with_labels = []
for idx, file in enumerate(file_paths):
    df = pd.read_csv(file, header=None)
    df['label'] = idx
    dfs_with_labels.append(df)

# Alle DataFrames zusammenführen
all_features = pd.concat(dfs_with_labels, ignore_index=True)

# Die ersten Zeilen des zusammengeführten DataFrames ausgeben


# Merkmalsmatrix extrahieren
X = all_features.drop('label', axis=1)
print(X.shape)
y = all_features['label']
print(y.shape)

# Aufteilen der Daten in Trainings-, Validierungs- und Testdaten (80% Trainingsdaten, 10% Validierungsdaten, 10% Testdaten)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

# Konvertiere pandas DataFrames in PyTorch-Tensoren
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)

X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.int64)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.int64)

# Erstelle PyTorch-Datasets und -DataLoader für Training, Validierung und Test
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Modellinitialisierung
model = MLP(input_size, hidden_size, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
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
plt.title(f'Confusionsmatrix PCA unausgewogen Datensatz MLP: accuracy: {accuracy}')
plt.show()

# Ergebnis der Confusionsmatrix in der Konsole ausgeben
print(f'Confusion Matrix: {conf_matrix}')
print(f'accuracy: {accuracy}')
