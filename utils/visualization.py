import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE

def visualize_reconstruction(imgs, pred, mask, patch_size, save_path=None):
    """
    Visualize original, masked, and reconstructed images.
    """
    # imgs: [N, 3, H, W]
    # pred: [N, L, p*p*3]
    # mask: [N, L]
    
    # Select first image in batch
    img = imgs[0].permute(1, 2, 0).cpu().numpy()
    # Unnormalize for visualization
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # Prepare pred
    p = patch_size
    h = w = int(pred.shape[1]**.5)
    pred_img = pred[0].detach().cpu().reshape(h, w, p, p, 3)
    pred_img = torch.einsum('hwpqc->hpwqc', pred_img).reshape(h*p, w*p, 3).numpy()
    # Unnormalize pred
    pred_img = pred_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    pred_img = np.clip(pred_img, 0, 1)
    
    # Prepare masked image
    mask_img = mask[0].detach().cpu().reshape(h, w).numpy()
    mask_img = np.repeat(np.repeat(mask_img, p, axis=0), p, axis=1)
    mask_img = np.expand_dims(mask_img, -1)
    
    masked_img = img * (1 - mask_img)
    
    # Combine original and reconstructed
    reconstructed_img = img * (1 - mask_img) + pred_img * mask_img
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[1].imshow(masked_img)
    axes[1].set_title("Masked")
    axes[2].imshow(pred_img)
    axes[2].set_title("Reconstruction (Patches)")
    axes[3].imshow(reconstructed_img)
    axes[3].set_title("Final Result")
    
    for ax in axes:
        ax.axis('off')
        
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_tsne(features, labels, class_names, save_path=None):
    """
    Plot t-SNE visualization of features.
    """
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=class_name, alpha=0.6)
    
    plt.legend()
    plt.title("t-SNE Visualization of Features")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
