from trainer import Trainer
from utils.dataset import FaceDataset, FaceDataloader
from arcface import ArcFaceModel
from utils.losses import get_loss
from utils.optimizers import SAM, Lamb
from adan_pytorch import Adan

import json
import torch
from PIL import ImageFile
from tqdm import tqdm
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/arcface.json', help='path to the config file')
    parser.add_argument('--device', type=str, default='0', help='train, test')
    return parser.parse_args()


