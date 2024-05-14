import os
import shutil
from sklearn.model_selection import train_test_split
import random

# Pfad zu den Ordnern mit den Bildern für jede Klasse
class_folders = {
    'Klasse_1': r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\1',
    'Klasse_2': r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\2',
    'Klasse_3': r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\3',
    'Klasse_4': r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\4',
    'Klasse_5': r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdantensatz unausgewogen\5'
}



# Zielverzeichnis für Trainings- und Testdatensätze
train_dir = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\train_dir'
test_dir = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\test_dir'


# Verhältnis für die Aufteilung der Daten (z.B. 80% für das Training, 20% für Tests)
test_size_ratio = 0.2

# Finden der kleinsten Klasse
min_class_size = min(len(os.listdir(folder_path)) for folder_path in class_folders.values())

# Durchlauf durch jede Klasse
for class_name, folder_path in class_folders.items():
    # Alle Dateinamen für die aktuelle Klasse sammeln
    filenames = os.listdir(folder_path)
    # Berechnung der Anzahl der Bilder für den Testdatensatz
    test_size = int(min_class_size * test_size_ratio)
    # Begrenzung der Anzahl der Bilder auf 20% der kleinsten Klasse für den Testdatensatz
    test_files = filenames[:test_size]
    train_files = filenames[test_size:]
    
    # Erstellen der Zielverzeichnisse für Trainings- und Testdaten, falls sie nicht existieren
    if not os.path.exists(os.path.join(train_dir, class_name)):
        os.makedirs(os.path.join(train_dir, class_name))
    if not os.path.exists(os.path.join(test_dir, class_name)):
        os.makedirs(os.path.join(test_dir, class_name))
    
    # Kopieren der Bilder in die entsprechenden Verzeichnisse und Beibehalten der Labels
    for filename in train_files:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(train_dir, class_name, filename)
        shutil.copyfile(src, dst)
    for filename in test_files:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(test_dir, class_name, filename)
        shutil.copyfile(src, dst)

print("Aufteilung in Trainings- und Testdatensätze abgeschlossen.")


