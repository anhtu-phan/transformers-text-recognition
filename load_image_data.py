from torch.utils.data import Dataset
import torch
import os
from skimage import io


class ImageDataset(Dataset):
    def __init__(self, label_file, root_dir, image_folder_name, transform=None):
        self.image_folder = os.path.join(root_dir, image_folder_name)
        self.samples = os.listdir(os.path.join(root_dir, image_folder_name))
        self.map_img_to_label = {}
        with open(os.path.join(root_dir, label_file), 'r') as f:
            for line in f:
                content = line.split(',')
                content[1] = content[1].replace('"', '')
                self.map_img_to_label[content[0].strip()] = content[1].strip()

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.samples[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = io.imread(img_path)
        target = self.map_img_to_label[img_name]

        if self.transform:
            image = self.transform(image)
        return image, target
