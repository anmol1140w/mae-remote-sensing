import torch
import os

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, 'model_best.pth')
        torch.save(state, best_filepath)

def load_checkpoint(model, optimizer=None, scheduler=None, filename='checkpoint.pth'):
    """
    Load model checkpoint.
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch']
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0
