import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from face_detector import detect

class UTKFace(Dataset):
    def __init__(self,
                 root_dir: str = "./data/UTKFace",
                 sample_size: int = 200,
                 mode: str = "train",
                 use_context: bool = False):
        super(UTKFace, self).__init__()
        """
        mode (str): train or val
        """
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
             transforms.Resize((sample_size, sample_size)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225]),
            ])
        self.file_names = os.listdir(os.path.join(self.root_dir, mode))            
        self.num_classes = 6
        self.use_context = use_context
        self.mode = mode

    def make_label(self, actual_age: int = 0):
        if actual_age >= 0 and actual_age < 13:
            return 0
        elif actual_age >= 13 and actual_age < 21:
            return 1
        elif actual_age >= 21 and actual_age < 45:
            return 2
        elif actual_age >= 45 and actual_age < 66:
            return 3
        elif actual_age >= 66 and actual_age < 85:
            return 4
        else:
            return 5
    
    def __getitem__(self, index: int = 0):
        file_name = self.file_names[index]
        aligned_face_name = file_name + ".chip.jpg"
        image_path = os.path.join(self.root_dir, self.mode, file_name)
        cropped_image_path = os.path.join(self.root_dir, "cropped", aligned_face_name)
        actual_age = int(file_name.split('_')[0])
        label = self.make_label(int(actual_age))
        if self.use_context:
            image, detections = detect(image_path)
            context = image.copy()
            detection = detections[0] # make sure that there is one face per image
            # crop face and mask the image to get context
            xmin = int(detection[0])
            ymin = int(detection[1])
            xmax = int(detection[2])
            ymax = int(detection[3])
            face = image[ymin:ymax, xmin:xmax, :]
            context[ymin:ymax, xmin:xmax, :] = 0
            # BGR to RGB
            # face = Image.fromarray(face[:, :, ::-1]).convert('RGB')
            face = Image.open(cropped_image_path).convert('RGB')
            context = Image.fromarray(context[:, :, ::-1]).convert('RGB')
            return face, context, label
        else:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, label

    def __len__(self):
        return len(self.file_names)

class MegaAgeAsian(Dataset):
    def __init__(self):
        pass

class FaceDataloader:
    def __init__(self,
                 train_set = None,
                 val_set = None,
                 dataset_ratio = None,
                 random_seed = 0,
                 batch_size_train = 16,
                 batch_size_val = 16):
        torch.manual_seed(random_seed)
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

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
