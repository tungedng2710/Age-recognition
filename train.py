from trainer import Trainer
from utils.datasets import UTKFace, FaceDataloader
# from models.caer import CAERSNet
from models.rexnet import ReXNetV1
from models.irse import *
from utils.losses import get_loss
from utils.optimizers import SAM, Lamb

import json
import torch
from PIL import ImageFile
from tqdm import tqdm
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/arcface.json', help='path to the config file')
    parser.add_argument('--device', type=str, default='0', help='gpu id (singular')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size train and val')
    parser.add_argument('--verbose', type=int, default=1, help='show training progress')
    return parser.parse_args()

def train(args):
    device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    dataset = UTKFace(root_dir="./data/UTKFace",
                      sample_size=224)
    dataloader = FaceDataloader(dataset=dataset, 
                                batch_size_train=args.batch_size, 
                                batch_size_val=args.batch_size)
    train_loader, val_loader = dataloader.get_dataloaders()

    # model = ReXNetV1(classes=5)
    model = IR_SE_50(input_size = [224, 224])
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-3, 
                                 weight_decay=1e-5)
    loss_function = get_loss()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', pytorch_total_params)
    trainer = Trainer(model=model,
                      n_epochs=args.n_epochs,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      device=device,
                      train_loader=train_loader,
                      val_loader=val_loader)
    trained_model = trainer.train(verbose=args.verbose)

if __name__ == "__main__":
    args = get_args()
    train(args)