import torch
import torch.nn as nn
from tqdm import tqdm
import os
from .trainer import BaseTrainer
from utils.metrics import AverageMeter, calculate_metrics
from utils.checkpoint import save_checkpoint
from optim.scheduler import adjust_learning_rate

class Finetuner(BaseTrainer):
    """
    Trainer for MAE Fine-tuning (Classification).
    """
    def __init__(self, model, criterion, config, output_dir):
        super().__init__(model, config, output_dir)
        self.criterion = criterion
        self.best_acc = 0

    def train_one_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch}")
        for i, (imgs, targets) in enumerate(pbar):
            lr = adjust_learning_rate(optimizer, epoch + i / len(train_loader), self.config)
            
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward
            with torch.cuda.amp.autocast():
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc = (outputs.argmax(dim=1) == targets).float().mean()
            losses.update(loss.item(), imgs.size(0))
            accs.update(acc.item(), imgs.size(0))
            
            pbar.set_postfix(loss=losses.avg, acc=accs.avg, lr=lr)
            
        self.writer.add_scalar('train/loss', losses.avg, epoch)
        self.writer.add_scalar('train/acc', accs.avg, epoch)
        return {'loss': losses.avg, 'acc': accs.avg}

    @torch.no_grad()
    def validate(self, val_loader, optimizer, epoch):
        self.model.eval()
        losses = AverageMeter()
        all_preds = []
        all_targets = []
        
        for imgs, targets in val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(imgs)
            loss = self.criterion(outputs, targets)
            
            losses.update(loss.item(), imgs.size(0))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
        metrics = calculate_metrics(all_targets, all_preds)
        val_acc = metrics['accuracy']
        
        self.writer.add_scalar('val/loss', losses.avg, epoch)
        self.writer.add_scalar('val/acc', val_acc, epoch)
        self.logger.info(f"Val Loss: {losses.avg:.4f}, Val Acc: {val_acc:.4f}")
        
        is_best = val_acc > self.best_acc
        if is_best:
            self.best_acc = val_acc
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'best_loss': self.best_metric,
            "config": self.config,
        }, is_best, self.output_dir)
        
        return metrics
