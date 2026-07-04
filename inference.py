import argparse
import yaml
import torch
from PIL import Image
from torchvision import transforms
from models.mae import MaskedAutoencoder
from models.classifier import ViTClassifier
from utils.checkpoint import load_checkpoint
import os

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    img = Image.open(args.image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
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
    load_checkpoint(model, filename=args.checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor.to(device))
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = prob[0, pred].item()
    
    print(f"Prediction: Class {pred}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Inference')
    parser.add_argument('--config', default='configs/config.yaml', help='path to config file')
    parser.add_argument('--checkpoint', required=True, help='path to finetuned checkpoint')
    parser.add_argument('--image', required=True, help='path to input image')
    args = parser.parse_args()
    main(args)
