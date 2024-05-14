import pandas as pd
import torch
import numpy as np

# Pfad zur CSV-Datei
csv_file_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec test\test_feature_vectors_2.csv'  

# Daten aus der CSV-Datei lesen
data = pd.read_csv(csv_file_path)

# Annahme: Ihre Daten enthalten Features (X) und Labels (y), die in Spalten enthalten sind
X = data.iloc[:, 1:].values  # Merkmale, außer der ersten Spalte (Annahme: die erste Spalte enthält den Dateinamen)
y = data.iloc[:, 0].values   # Labels sind in der ersten Spalte enthalten

teile = y[0].split(',')
print(len(teile))

for x in range(len(teile)):
    print(teile[x])
