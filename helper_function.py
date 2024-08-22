import torch
import os
import random
import gc
import re
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

def seed_set(SEED = 42):    
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    print(f'Seed set at {SEED}')

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def visualize_melspectograms(df, mel_dir_path, n_samples, ncols = 4):
    
    nrows = ceil(n_samples / ncols) 
    
    samples = df.sample(n=n_samples)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(40, 30))
    axes = axes.flatten()
    
    for  i, (_, row) in enumerate(samples.iterrows()):                
        np_path = mel_dir_path / f"{row['file_name']}.npy"
        label = 'Sleep' if row['label'] == 1 else 'Non-Sleep'
        image_array = np.load(np_path)
        
        axes[i].imshow(image_array, aspect='auto', origin='lower')
        axes[i].set_title(label, fontsize=20)
        axes[i].axis('off')
        
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
        

def extract_val_loss(checkpoint_path):
    match = re.search(r"val_loss([\d.]+)", str(checkpoint_path))
    return float(match.group(1)) if match else float('inf')

def get_best_checkpoint(checkpoint_dir):
    checkpoint_paths = list(checkpoint_dir.glob("*.ckpt"))
    sorted_checkpoints = sorted(checkpoint_paths, key=extract_val_loss)
    return sorted_checkpoints[0] if sorted_checkpoints else None

    