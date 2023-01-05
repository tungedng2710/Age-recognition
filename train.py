"""
Copyright tungedng2710
"""
import argparse
import yaml
import torch

from trainer import Trainer
from utils.datasets import UTKFace, FaceDataloader
from utils.losses import get_loss
from models.age_model import TonNet

def get_args():
    """
    Get parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/UTKFace_train.yaml",
                        help="path to the config file")
    parser.add_argument("--device", type=str, default='0', help="gpu id (singular)")
    parser.add_argument("--save", action='store_true', help="save model")
    return parser.parse_args()

def train(parsing_args):
    """
    Train model with arguments
    """
    with open(parsing_args.config) as stream:
        configs = yaml.safe_load(stream)
    device = torch.device("cuda:" + parsing_args.device if torch.cuda.is_available() else "cpu")
    dataset = UTKFace(root_dir=configs["root_dir"],
                      sample_size=configs["input_size"])
    dataloader = FaceDataloader(dataset=dataset,
                                dataset_ratio=configs["dataset_ratio"],
                                batch_size_train=configs["batch_size_train"],
                                batch_size_val=configs["batch_size_val"])
    train_loader, val_loader = dataloader.get_dataloaders(num_worker=8)

    model = TonNet(model_name=configs["model"], num_classes=dataset.num_classes)
    if configs["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=configs["lr"], momentum=0.9)
    elif configs["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], weight_decay=1e-5)
    else:
        print("unsupported optimizer!")
    loss_function = get_loss()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", pytorch_total_params)
    trainer = Trainer(model=model,
                      n_epochs=configs["num_epochs"],
                      optimizer=optimizer,
                      loss_function=loss_function,
                      device=device,
                      train_loader=train_loader,
                      val_loader=val_loader)
    trained_model = trainer.train(scheduler_config=configs["scheduler"])
    if parsing_args.save:
        trainer.save_trained_model(trained_model=trained_model,
                                   backbone_name=configs["model"])

if __name__ == "__main__":
    args = get_args()
    train(parsing_args=args)
