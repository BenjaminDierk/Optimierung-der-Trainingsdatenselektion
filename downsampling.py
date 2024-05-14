import os
import random
from collections import defaultdict

# Pfade zu den Ordnern jeder Klasse
class1 = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz downsampling\1'
class2 = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz downsampling\2'
class3 = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz downsampling\3'
class4 = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz downsampling\4'
class5 = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz downsampling\5'

class_folders = [class1, class2, class3, class4, class5]

# Dictionary, um die Anzahl der Bilder pro Klasse und Perspektive zu zählen
class_perspective_count = defaultdict(lambda: defaultdict(int))

# Zähle die Anzahl der Bilder pro Klasse und Perspektive
for class_folder in class_folders:
    for filename in os.listdir(class_folder):
        perspective = filename.split('_')[2]
        class_perspective_count[class_folder][perspective] += 1
        

# Finde die minimale Anzahl von Bildern pro Klasse und Perspektive
min_count = min(min(counts.values()) for counts in class_perspective_count.values())


# Drucke das gefüllte Dictionary class_perspective_count
print("Anzahl der Bilder pro Klasse und Perspektive:")
for class_folder, perspective_counts in class_perspective_count.items():
    print(f"Klasse: {class_folder}")
    for perspective, count in perspective_counts.items():
        print(f"- Perspektive: {perspective}, Anzahl der Bilder: {count}")



# Entferne zufällig überschüssige Bilder aus jeder Klasse und Perspektive
for class_folder in class_folders:
    for perspective, count in class_perspective_count[class_folder].items():
        while count > min_count:
            # Wähle zufällig ein Bild aus dieser Klasse und Perspektive aus
            filename = random.choice([f for f in os.listdir(class_folder) if f.split('_')[2] == perspective])
            # Lösche das ausgewählte Bild
    
            os.remove(os.path.join(class_folder, filename))
            count -= 1


print("Datensatz wurde erfolgreich ausgeglichen.")
