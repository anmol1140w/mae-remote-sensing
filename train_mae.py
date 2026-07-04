import argparse
import yaml
import os
import torch
from models.mae import MaskedAutoencoder
from losses.reconstruction_loss import MAELoss
from engine.pretrainer import Pretrainer
from utils.dataset import get_datasets, get_dataloaders
from utils.seed import set_seed

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    
    # Update config with CLI args
    output_dir = os.path.join(config['output_dir'], config['experiment_name'], 'pretrain')
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
    model = MaskedAutoencoder(
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
    
    # Criterion
    criterion = MAELoss(norm_pix_loss=config['model']['norm_pix_loss'])
    
    # Optimizer
    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=config['pretrain']['base_lr'], weight_decay=config['pretrain']['weight_decay'])
    
    # Trainer
    trainer = Pretrainer(model, criterion, config['pretrain'], output_dir)
    trainer.config['mask_ratio'] = config['model']['mask_ratio']
    trainer.config['vis_freq'] = config['vis']['save_freq']
    
    # Start training
    trainer.fit(train_loader, val_loader, optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Pre-training')
    parser.add_argument('--config', default='configs/config.yaml', help='path to config file')
    args = parser.parse_args()
    main(args)
