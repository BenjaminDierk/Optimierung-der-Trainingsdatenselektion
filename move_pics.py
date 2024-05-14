import os
import random
import shutil

def count_images_in_folder(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return len(images)

def move_images_to_new_folder(original_folder, new_folder, percentage):
    images = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    random.shuffle(images)
    
    num_images = len(images)
    num_images_to_move = int(num_images * percentage)
    
    images_to_move = images[:num_images_to_move]
    
    for image in images_to_move:
        src = os.path.join(original_folder, image)
        dst = os.path.join(new_folder, image)
        shutil.move(src, dst)

if __name__ == "__main__":
    original_folder_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\train_dir\2'
    new_folder_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\test_dir\test 2'
    
    total_images = count_images_in_folder(original_folder_path)
    percentage_to_move = 0.3  # 20%
    
    move_images_to_new_folder(original_folder_path, new_folder_path, percentage_to_move)
    
    print("Moved {} images out of {} total images.".format(int(total_images * percentage_to_move), total_images))
