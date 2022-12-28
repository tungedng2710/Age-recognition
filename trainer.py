import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import datetime
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def create_writer():
    now = '{0:%Y%m%d}'.format(datetime.datetime.now())
    if not os.path.exists('./logs/'+now):
        os.mkdir('./logs/'+now)
    path = './logs/'+now+'/'
    writer = SummaryWriter(path)
    return writer

class Trainer:
    def __init__(self,
                 model = None,
                 n_epochs = 10,
                 optimizer = None,
                 loss_function = None,
                 train_loader = None,
                 val_loader = None,
                 device = torch.device('cuda:0')):
        assert model is not None
        assert optimizer is not None
        assert loss_function is not None
        assert train_loader is not None
        assert val_loader is not None

        self.device = device
        self.model = model
        self.model.to(self.device)
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = create_writer()

    def get_scheduler(self, scheduler_config):
        if scheduler_config['name'] == 'StepLR':
            lr_scheduler = StepLR(self.optimizer, 
                                step_size=scheduler_config['StepLR']['step_size'], 
                                gamma=scheduler_config['StepLR']['gamma'],
                                verbose=scheduler_config['StepLR']['verbose'])
        elif scheduler_config['name'] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(self.optimizer, 
                                             T_max=scheduler_config['CosineAnnealingLR']['T_max'])
        else:
            raise Exception("Unavailable scheduler")
        return lr_scheduler
    
    def train(self, 
              verbose = 0, 
              use_sam_optim = False,
              scheduler_config = None):
        '''
        verbose: 
            0: show nothing
            1: show results per epoch only
            2: show train losses per iteration
        use_sam_optim: True if using SAM Optimizer
        '''
        best_model = self.model
        best_acc = -1
        train_loss = 0.0
        if scheduler_config is not None:
            lr_scheduler = self.get_scheduler(scheduler_config)
        for epoch in range(self.n_epochs):
            self.model.train()
            print("--------------------------------------------------------------------------")
            print("Epoch: ", epoch+1)
            print("Training...")
            for idx, (images, y_train) in enumerate(self.train_loader):
                pass