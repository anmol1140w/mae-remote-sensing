import argparse
import yaml
import os
import torch
import torch.nn as nn
from models.mae import MaskedAutoencoder
from models.classifier import ViTClassifier
from engine.finetuner import Finetuner
from utils.dataset import get_datasets, get_dataloaders
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    
    output_dir = os.path.join(config['output_dir'], config['experiment_name'], 'finetune')
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset
    train_ds, val_ds, _ = get_datasets(
        config['dataset']['path'], 
        img_size=config['dataset']['image_size']
    )
    train_loader, val_loader, _ = get_dataloaders(
        train_ds, val_ds, val_ds, 
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers']
    )
    
    # Model
    mae = MaskedAutoencoder(
        img_size=config['dataset']['image_size'],
        patch_size=config['model']['patch_size'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        mlp_ratio=config['model']['mlp_ratio']
    )
    
    # Load pretrained weights
    if args.pretrained:
        load_checkpoint(mae, filename=args.pretrained)
        print(f"Loaded pretrained weights from {args.pretrained}")
    
    # Create Classifier
    model = ViTClassifier(mae.encoder, num_classes=config['dataset']['num_classes'])
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['finetune']['base_lr'], weight_decay=config['finetune']['weight_decay'])
    
    # Trainer
    trainer = Finetuner(model, criterion, config['finetune'], output_dir)
    
    # Start training
    trainer.fit(train_loader, val_loader, optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Fine-tuning')
    parser.add_argument('--config', default='configs/config.yaml', help='path to config file')
    parser.add_argument('--pretrained', help='path to pretrained MAE checkpoint')
    args = parser.parse_args()
    main(args)
