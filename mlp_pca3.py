import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# Definiere die Hyperparameter
input_size = 200  # Eingabegröße entsprechend der Anzahl der Hauptkomponenten
hidden_size1 = 256
hidden_size2 = 128
output_size = 5  # Anzahl der Klassen

# Erstelle eine Instanz des Modells
model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Definiere die Verlustfunktion und den Optimierer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Ausgabe des Modells
print(model)
