import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from ..utils.logger import setup_logger
from ..utils.checkpoint import save_checkpoint

class BaseTrainer:
    """
    Base class for trainers.
    """
    def __init__(self, model, config, output_dir):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(config['device'] if config['device'] != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        self.logger = setup_logger(output_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        
        self.start_epoch = 0
        self.best_metric = float('inf')

    def train_one_epoch(self, train_loader, optimizer, epoch):
        raise NotImplementedError

    def validate(self, val_loader, epoch):
        raise NotImplementedError

    def fit(self, train_loader, val_loader, optimizer, scheduler=None):
        for epoch in range(self.start_epoch, self.config['epochs']):
            self.logger.info(f"Epoch {epoch}/{self.config['epochs']}")
            
            train_stats = self.train_one_epoch(train_loader, optimizer, epoch)
            val_stats = self.validate(val_loader, epoch)
            
            if scheduler:
                scheduler.step()
                
            # Logging and Checkpointing logic would go here
            # (To be specialized in child classes)
