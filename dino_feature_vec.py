
import os
import torch.nn as nn
import numpy as np
from PIL import Image
from DINO import utils
from DINO import vision_transformer as vits
from torchvision import transforms as pth_transforms
import csv


# Function to get feature vectors from images
def get_vectors(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    
    if device == 'cuda':
        vector = model(img.cuda(non_blocking=True))
    if device == 'cpu':
        vector = model(img)
    
    vector = nn.functional.normalize(vector, dim=1, p=2)
    
    return vector.detach().cpu().numpy()

# Function to process all images in a folder
def process_images_in_folder(folder_path, model, device):
    feature_vectors = []
    image_paths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            vector = get_vectors(image_path, model, device)
            feature_vectors.append([filename] + vector.tolist())

    return image_paths, feature_vectors

if __name__ == '__main__':
    
    device = 'cuda'  # 'cuda' or 'cpu'

    folder_path = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\train_dir\5'
    output_csv = r'C:\Users\benni\Desktop\Humanoide Robotik\Bachlorarbeit Dokumente -Praxisphase dokumente\Bachlorarbeit\GUI-20240217T110612Z-001\GUI\Trainingsdatensatz upsampling neu\feature_vec train\train_feature_vectors_5.csv'

    # Model setup
    model = vits.vit_small()
    #model = nn.hub.load('facebookresearch/dino:main', 'dino_vits16')

    if device == 'cuda':
        model.cuda()
    elif device == 'cpu':
        model = model.cpu()

    
    utils.load_pretrained_weights(model=model, pretrained_weights=True, checkpoint_key='teacher', model_name='vit_small',patch_size=16)
    
    model.eval()

    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_paths, feature_vectors = process_images_in_folder(folder_path, model, device)

    # Save feature vectors to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename'] + [f'Feature_{i+1}' for i in range(len(feature_vectors[0])-1)])  # Header
        writer.writerows(feature_vectors)