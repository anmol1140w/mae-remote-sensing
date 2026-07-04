import torch
import torch.nn as nn
from tqdm import tqdm
import os
from .trainer import BaseTrainer
from utils.metrics import AverageMeter
from utils.checkpoint import save_checkpoint
from utils.visualization import visualize_reconstruction
from optim.scheduler import adjust_learning_rate

class Pretrainer(BaseTrainer):
    """
    Trainer for MAE Pre-training.
    """
    def __init__(self, model, criterion, config, output_dir):
        super().__init__(model, config, output_dir)
        self.criterion = criterion
        self.patch_size = model.encoder.patch_embed.patch_size

    def train_one_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for i, (imgs, _) in enumerate(pbar):
            # Adjust LR
            lr = adjust_learning_rate(optimizer, epoch + i / len(train_loader), self.config)
            
            imgs = imgs.to(self.device, non_blocking=True)
            
            # Forward
            with torch.cuda.amp.autocast():
                pred, mask = self.model(imgs, mask_ratio=self.config['mask_ratio'])
                loss = self.criterion(imgs, pred, mask, self.patch_size)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            if self.config.get('clip_grad'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])
            optimizer.step()
            
            losses.update(loss.item(), imgs.size(0))
            pbar.set_postfix(loss=losses.avg, lr=lr)
            
            self.writer.add_scalar('train/loss_step', loss.item(), epoch * len(train_loader) + i)
            
        self.writer.add_scalar('train/loss_epoch', losses.avg, epoch)
        self.writer.add_scalar('train/lr', lr, epoch)
        return {'loss': losses.avg}

    @torch.no_grad()
    def validate(self, val_loader, optimizer, epoch):
        self.model.eval()
        losses = AverageMeter()
        
        for i, (imgs, _) in enumerate(val_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            
            pred, mask = self.model(imgs, mask_ratio=self.config['mask_ratio'])
            loss = self.criterion(imgs, pred, mask, self.patch_size)
            
            losses.update(loss.item(), imgs.size(0))
            
            # Visualize periodically
            if i == 0 and epoch % self.config.get('vis_freq', 10) == 0:
                vis_path = os.path.join(self.output_dir, f'vis_epoch_{epoch}.png')
                visualize_reconstruction(imgs, pred, mask, self.patch_size, vis_path)
        
        self.writer.add_scalar('val/loss', losses.avg, epoch)
        self.logger.info(f"Val Loss: {losses.avg:.4f}")
        
        # Save best model
        is_best = losses.avg < self.best_metric
        if is_best:
            self.best_metric = losses.avg
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'best_loss': self.best_metric,
            "config": self.config,
        }, is_best, self.output_dir)
        
        return {'loss': losses.avg}
