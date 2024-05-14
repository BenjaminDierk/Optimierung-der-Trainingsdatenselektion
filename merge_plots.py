import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Pfade zu den gespeicherten Plot-Bildern
plot1_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\Arbeit\Dino downsampling\Confusionmatrix Dino downsamling knn k1.png'
plot2_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\Arbeit\Dino downsampling\Confusionmatrix Dino downsamling knn k5.png'
plot3_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\Arbeit\Dino downsampling\Confusionmatrix Dino downsamling knn k10.png'

# Lade die Plot-Bilder
plot1_img = mpimg.imread(plot1_path)
plot2_img = mpimg.imread(plot2_path)
plot3_img = mpimg.imread(plot3_path)

# Erstelle eine neue Figur und Achsen
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Füge die Plot-Bilder in die Achsen ein
axes[0].imshow(plot1_img)
axes[1].imshow(plot2_img)
axes[2].imshow(plot3_img)

# Füge Titel hinzu (optional)
axes[0].set_title('Plot 1')
axes[1].set_title('Plot 2')
axes[2].set_title('Plot 3')

# Zeige den kombinierten Plot an
plt.tight_layout()
plt.show()
