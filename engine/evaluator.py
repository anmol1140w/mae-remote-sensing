import torch
import numpy as np
from tqdm import tqdm
from ..utils.metrics import calculate_metrics
from ..utils.visualization import plot_tsne

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
        all_preds = []
        all_targets = []
        all_features = []
        
        pbar = tqdm(test_loader, desc="Evaluating")
        for imgs, targets in pbar:
            imgs = imgs.to(self.device)
            
            # Forward
            # Assuming model is ViTClassifier which returns logits
            outputs = self.model(imgs)
            
            # For t-SNE, we might want features before the head
            # This depends on the model architecture
            if hasattr(self.model, 'encoder'):
                # Extract features from encoder
                latent, _, _ = self.model.encoder(imgs, mask_ratio=0.0)
                features = latent[:, 0].cpu().numpy() # cls token
                all_features.append(features)
            
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
        metrics = calculate_metrics(all_targets, all_preds)
        
        if all_features:
            features = np.concatenate(all_features, axis=0)
            targets = np.array(all_targets)
            # Sample for t-SNE if too many
            if len(features) > self.config.get('tsne_samples', 1000):
                idx = np.random.choice(len(features), self.config['tsne_samples'], replace=False)
                features = features[idx]
                targets = targets[idx]
            
            plot_tsne(features, targets, class_names, save_path='outputs/tsne_plot.png')
            
        return metrics
