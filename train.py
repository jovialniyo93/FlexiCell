import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from net import FlexiCell 
from utils import *
import numpy as np
import cv2
from tqdm import tqdm
import logging
from torchvision.transforms import transforms
import os
import gc

# Memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

model_path = 'checkpoints/'
imgs_path = 'data/imgs/'
mask_path = 'data/mask/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
y_transforms = transforms.Compose([
    transforms.ToTensor()
])

def __normalize(mask):
    min_val, max_val = np.unique(mask)[0], np.unique(mask)[-1]
    mask = mask / 1.0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i][j] = (mask[i][j] - min_val) / (max_val - min_val)
    mask = mask.astype(np.float16)
    return mask

def record_result(string):
    file_name = "train_record.txt"
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            print("successfully create record file")
    with open(file_name, 'a') as f:
        f.write(string + "\n")
    print(string + " has been recorded")

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def train_model(model, criterion, optimizer, dataload, keep_training, num_epochs=5):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    # Mixed precision training setup for memory efficiency
    scaler = torch.amp.GradScaler('cuda')
    
    if keep_training:
        checkpoints = os.listdir(model_path)
        checkpoints.sort()
        final_ckpt = checkpoints[-1]
        print("Continue training from ", final_ckpt)
        restart_epoch = final_ckpt.replace("CP_epoch", "").replace(".pth", "")
        restart_epoch = int(restart_epoch)
        model.load_state_dict(torch.load(model_path + final_ckpt, map_location=device))
    else:
        restart_epoch = 1
        if os.path.isfile("train_record.txt"):
            os.remove("train_record.txt")
            print("Old result has been cleaned!")
    
    # Clear memory before training
    clear_memory()
    print("Starting training...")
    print_gpu_memory()
    
    for epoch in range(restart_epoch - 1, num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        data_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        
        for x, y in tqdm(dataload):
            step += 1
            inputs = x.to(device, non_blocking=True)
            labels = y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass to save memory
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                segmentation_output, edges_output = outputs
                
                # Handle shape mismatch - adjust labels to match model output
                if segmentation_output.shape[1] == 2 and labels.shape[1] == 1:
                    # Convert single channel labels to 2-class format
                    # Assuming labels are 0 for background, 1 for foreground
                    labels_2class = torch.zeros_like(segmentation_output)
                    labels_2class[:, 0] = (labels[:, 0] == 0).float()  # Background
                    labels_2class[:, 1] = (labels[:, 0] == 1).float()  # Foreground
                    loss = criterion(segmentation_output, labels_2class)
                elif segmentation_output.shape[1] == 1:
                    # Single class output - direct loss
                    loss = criterion(segmentation_output, labels)
                else:
                    # Shapes already match
                    loss = criterion(segmentation_output, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Clear memory every few steps to prevent accumulation
            if step % 3 == 0:
                clear_memory()
            
            # Memory monitoring (optional - remove if too verbose)
            if step % 10 == 0:
                print_gpu_memory()
        
        print("epoch %d loss:%.3f" % (epoch + 1, epoch_loss / step))
        record_result("epoch %d loss:%.3f" % (epoch + 1, epoch_loss / step))
        
        try:
            os.mkdir(model_path)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        
        torch.save(model.state_dict(), model_path + f'CP_epoch{str(epoch + 1).zfill(2)}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')
        
        # Clear memory after each epoch
        clear_memory()

if __name__ == "__main__":
    keep_training = False
    
    # OPTION 1: Binary segmentation (1 class) - more memory efficient
    model = FlexiCell(n_channels=1, n_classes=1)
    
    # OPTION 2: Two-class segmentation (uncomment if you need 2 classes)
    # model = FlexiCell(n_channels=1, n_classes=2)
    
    # Reduced batch size for memory efficiency
    batch_size = 1  # Start with 1, try 2 or 4 if no memory errors
    
    criterion = nn.BCEWithLogitsLoss()
    # Slightly lower learning rate for stability with mixed precision
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    data = TrainDataset(imgs_path, mask_path, x_transforms, y_transforms)
    # Reduced num_workers to save memory
    dataloader = DataLoader(data, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Clear any existing memory before starting
    clear_memory()
    
    train_model(model, criterion, optimizer, dataloader, keep_training, num_epochs=5)