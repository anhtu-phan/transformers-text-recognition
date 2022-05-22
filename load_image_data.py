import string

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import cv2
# from skimage import io
# from skimage.transform import resize
# # from skimage.color import rgba2rgb


class ImageDataset(Dataset):
    def __init__(self, label_file, root_dir, image_folder_name, max_out_len=10, transform=None):
        self.image_folder = os.path.join(root_dir, image_folder_name)
        self.samples = os.listdir(os.path.join(root_dir, image_folder_name))
        self.map_img_to_label = {}
        self.vocabulary = string.printable
        self.max_out_len = max_out_len
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
        image = cv2.imread(img_path)
        # image = image[:,:,:3]
        image = cv2.resize(image, (300, 100))
        label = self.map_img_to_label[img_name]
        target = []
        for c in label:
            target.append(self.vocabulary.index(c)+1)
        if len(target) < self.max_out_len:
            target += [0]*(self.max_out_len-len(target))
        else:
            target = target[:self.max_out_len]
        target = torch.tensor(target, dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        return image, target


def get_data(batch_size, max_out_len, split_rate=0.9):
    transform = list()
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    transform = transforms.Compose(transform)

    full_train_data = ImageDataset('gt.txt', './dataset/train', 'images', max_out_len, transform)

    num_samples = len(full_train_data)
    train_samples = int(num_samples*split_rate)
    valid_samples = num_samples - train_samples

    train_data, valid_data = torch.utils.data.random_split(full_train_data, [train_samples, valid_samples])
    print(f"{'-'*10}Len of train data: {len(train_data)}{'-'*10}\n")
    print(f"{'-'*10}Len of valid data: {len(valid_data)}{'-'*10}\n")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False)

    return train_loader, valid_loader


def get_test_data(batch_size, max_out_len):
    transform = list()
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    transform = transforms.Compose(transform)

    test_data = ImageDataset('gt.txt', './dataset/test', 'images', max_out_len, transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
    return test_loader
