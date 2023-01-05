import argparse
import yaml
import torch

from trainer import Trainer
from utils.datasets import UTKFace, FaceDataloader
from utils.losses import get_loss
from models.age_model import TonNet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/UTKFace_train.yaml",
                        help="path to the config file")
    parser.add_argument("--device", type=str, default='0', help="gpu id (singular)")
    parser.add_argument("--save", type=str, default='0', help="gpu id (singular)")
    return parser.parse_args()

def train(args):
    """
    Train model with arguments
    """
    with open(args.config) as stream:
        configs = yaml.safe_load(stream)
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    dataset = UTKFace(root_dir=configs["root_dir"],
                      sample_size=configs["input_size"])
    dataloader = FaceDataloader(dataset=dataset,
                                dataset_ratio=configs["dataset_ratio"],
                                batch_size_train=configs["batch_size_train"],
                                batch_size_val=configs["batch_size_val"])
    train_loader, val_loader = dataloader.get_dataloaders(num_worker=8)

    model = TonNet(model_name=configs["model"], num_classes=dataset.num_classes)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=configs["lr"],
                                 weight_decay=1e-5)
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
    trainer.save_trained_model(trained_model = trained_model,
                               backbone_name=configs["model"])

if __name__ == "__main__":
    args = get_args()
    with open(args.config) as stream:
        configs = yaml.safe_load(stream)

    train(args)