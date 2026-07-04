import argparse
import yaml
import os
import torch
from models.mae import MaskedAutoencoder
from models.classifier import ViTClassifier
from engine.evaluator import Evaluator
from utils.dataset import get_datasets, get_dataloaders
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    
    # Dataset
    _, _, test_ds = get_datasets(
        config['dataset']['path'], 
        img_size=config['dataset']['image_size']
    )
    _, _, test_loader = get_dataloaders(
        test_ds, test_ds, test_ds, 
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
    
    model = ViTClassifier(mae.encoder, num_classes=config['dataset']['num_classes'])
    
    # Load weights
    load_checkpoint(model, filename=args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(model, config, device)
    
    # Evaluate
    class_names = test_ds.dataset.classes if hasattr(test_ds, 'dataset') else [f'class_{i}' for i in range(config['dataset']['num_classes'])]
    metrics = evaluator.evaluate(test_loader, class_names)
    
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Evaluation')
    parser.add_argument('--config', default='configs/config.yaml', help='path to config file')
    parser.add_argument('--checkpoint', required=True, help='path to finetuned checkpoint')
    args = parser.parse_args()
    main(args)
