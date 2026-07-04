import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.visualization import plot_tsne
import os 

class Evaluator:
    """
    Evaluator for classification models.
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, test_loader, class_names):
    
        self.model.eval()
    
        all_preds = []
        all_targets = []
        all_features = []
    
        for imgs, targets in tqdm(test_loader, desc="Evaluating"):
    
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
    
            outputs = self.model(imgs)
    
            preds = outputs.argmax(dim=1)
    
            features = self.model.forward_features(imgs)
    
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_features.append(features.cpu().numpy())
    
        metrics = calculate_metrics(all_targets, all_preds)
    
        features = np.concatenate(all_features, axis=0)
        targets = np.array(all_targets)
    
        if len(features) > self.config.get("vis", {}).get("tsne_samples", 1000):
            idx = np.random.choice(len(features), 1000, replace=False)
            features = features[idx]
            targets = targets[idx]
    
        save_path = os.path.join(self.config["output_dir"], "tsne.png")
    
        plot_tsne(features, targets, class_names, save_path=save_path)
    
        return metrics
