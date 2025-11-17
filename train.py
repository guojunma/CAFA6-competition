import sys
tag_ = 'code/'
sys.path.append(tag_)

import numpy as np
import pandas as pd
import os
import datetime
import pickle
import argparse
from tqdm import tqdm
import json
import time
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F

from gvp_transformer import GVPTransformerModel
from data import Alphabet, Alphabet_goclean
from model_util import get_n_params, CreateDataset, BatchGvpesmConverter


def train_epoch(model, device, dataloader, optimizer, prediction_go_mask, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training", unit="batch")
    for batch_idx, data in enumerate(pbar):
        optimizer.zero_grad()
        
        # Forward pass
        model_out = model(
            data['esm1vs'], 
            data['coords'], 
            data['seqs'], 
            data['padding_mask'], 
            data['plddts'],
            return_all_hiddens=False
        )
        
        logits = torch.squeeze(model_out[0], 1)  # [batch_size, num_tokens]
        
        # Filter to only GO terms
        logits_go = logits[:, prediction_go_mask]
        labels_go = data['labels'][:, prediction_go_mask]
        
        # Binary cross-entropy loss for multi-label classification
        loss = F.binary_cross_entropy_with_logits(logits_go, labels_go)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, device, dataloader, prediction_go_mask, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Validation", unit="batch")
        for batch_idx, data in enumerate(pbar):
            # Forward pass
            model_out = model(
                data['esm1vs'], 
                data['coords'], 
                data['seqs'], 
                data['padding_mask'], 
                data['plddts'],
                return_all_hiddens=False
            )
            
            logits = torch.squeeze(model_out[0], 1)
            
            # Filter to only GO terms
            logits_go = logits[:, prediction_go_mask]
            labels_go = data['labels'][:, prediction_go_mask]
            
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(logits_go, labels_go)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, checkpoint_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch


def main():
    parser = argparse.ArgumentParser(description='Train PANDA-3D model for protein function prediction')
    parser.add_argument('--config', type=str, default='code/args.json', help='Path to config file')
    parser.add_argument('--database_dir', type=str, required=True, help='Path to database directory with preprocessed features')
    parser.add_argument('--train_df', type=str, default='train_0130_df.pkl', help='Training dataframe pickle file')
    parser.add_argument('--valid_df', type=str, default='valid_0130_df.pkl', help='Validation dataframe pickle file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0 or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load model configuration
    print("Loading model configuration...")
    model_args = json.load(open(args.config, 'r'))
    model_args = Namespace(**model_args)
    
    # Override config with command-line arguments
    model_args.lr = args.lr
    model_args.batch_size = args.batch_size
    model_args.device = [args.device]
    
    print(f"Using device: {model_args.device[0]}")
    print(f"Batch size: {model_args.batch_size}")
    print(f"Learning rate: {model_args.lr}")
    
    # Load alphabets
    print("Loading alphabets and GO terms...")
    alphabet = Alphabet.from_architecture(model_args.arch)
    terms = list(pd.read_pickle(tag_ + model_args.terms_pkl))
    alphabet_go = Alphabet_goclean(terms)
    
    # Create GO term mask (only predict GO terms)
    prediction_go_mask = np.array(['GO' in tok for tok in alphabet_go.all_toks])
    print(f"Number of GO terms to predict: {prediction_go_mask.sum()}")
    
    # Initialize model
    print("Initializing model...")
    model = GVPTransformerModel(
        model_args,
        alphabet,
        alphabet_go,
    )
    model.to(model_args.device[0])
    print(get_n_params(model))
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=model_args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Load datasets
    print("Loading training dataset...")
    train_data = CreateDataset(args.database_dir, args.train_df, batch_size_=model_args.batch_size)
    batch_converter = BatchGvpesmConverter(alphabet, alphabet_go, model_args.coords_mask_plddt_th, model_args.device[0])
    train_dataloader = DataLoader(train_data, batch_size=model_args.batch_size, shuffle=True, collate_fn=batch_converter)
    
    print("Loading validation dataset...")
    valid_data = CreateDataset(args.database_dir, args.valid_df, batch_size_=model_args.batch_size)
    valid_dataloader = DataLoader(valid_data, batch_size=model_args.batch_size, shuffle=False, collate_fn=batch_converter)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, model_args.device[0], train_dataloader, optimizer, prediction_go_mask, epoch)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, model_args.device[0], valid_dataloader, prediction_go_mask, epoch)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, checkpoint_path)
        
        # Save latest checkpoint
        latest_checkpoint = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, latest_checkpoint)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print("="*60)


if __name__ == "__main__":
    main()
