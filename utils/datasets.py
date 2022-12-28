import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class UTKFace(Dataset):
    def __init__(self, 
                 root_dir: str = "age_data/UTKFace",
                 sample_size: int = 200):
        super(UTKFace, self).__init__()
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
             transforms.Resize(size=sample_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225]),
             ])
        self.file_names = os.listdir(self.root_dir)            

    def make_label(self, actual_age: int = 0):
        if actual_age >= 0 and actual_age < 13:
            return 0
        elif actual_age >= 13 and actual_age < 21:
            return 1
        elif actual_age >= 21 and actual_age < 45:
            return 2
        elif actual_age >= 45 and actual_age < 66:
            return 3
        else:
            return 4
    
    def __getitem__(self, index: int = 0):
        file_name = self.file_names[index]
        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path)
        image = self.transform(image)
        actual_age = int(file_name.split('_')[0])
        return image, self.make_label(actual_age)

    def __len__(self):
        return len(self.file_names)

class FaceDataloader:
    def __init__(self, 
                 root_dir = None, 
                 val_size = 0.2, 
                 dataset_ratio = None,
                 random_seed = 0,
                 batch_size_train = 64,
                 batch_size_val = 32,
                 save_label_dict = False):
        torch.manual_seed(random_seed)
        if dataset_ratio is None:
            dataset_ratio = [0.8, 0.2]
        self.dataset = FaceDataset(root_dir=root_dir)
        self.num_classes = self.dataset.num_classes
        self.val_size = int(val_size * self.dataset.__len__())
        self.train_size = self.dataset.__len__() - self.val_size
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, dataset_ratio)
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        if save_label_dict:
            self.dataset.save_label_dict()

    def get_dataloaders(self, num_worker = 8):
        train_loader = DataLoader(self.train_set,
                                batch_size = self.batch_size_train,
                                shuffle = True,
                                num_workers = num_worker,
                                drop_last=True)
        val_loader = DataLoader(self.val_set,
                                batch_size = self.batch_size_val,
                                shuffle = False,
                                num_workers = num_worker)
        return train_loader, val_loader


if __name__ == "__main__":
    train_loader = DataLoader(dataset = dataset,
                              batch_size = 8,
                              shuffle = True,
                              num_workers = 8,
                              drop_last=True)
