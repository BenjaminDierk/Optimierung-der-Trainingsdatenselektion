import cv2
import os

def process_images(input_folder, output_folder, target_size=(455, 256)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of files in input folder
    files = os.listdir(input_folder)
    
    for file_name in files:
        # Read image
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        resized_img = cv2.resize(gray_img, target_size)
        
        # Save processed image
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, resized_img)

# Specify input and output folders
# Verzeichnis mit den Originalbildern
original_dir = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\resize bilder MSE\original images'
# Verzeichnis zum Speichern der verkleinerten Bilder
resized_dir = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\resize bilder MSE\resized images'


# Process images
process_images(original_dir, resized_dir)