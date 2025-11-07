"""
Deep learning models for GO term prediction.
Includes CNN and Transformer-based models for protein sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm


class ProteinSequenceDataset(Dataset):
    """Dataset for protein sequences and GO term labels."""
    
    def __init__(self, sequences: List[str], labels: np.ndarray, max_length: int = 1000):
        """
        Initialize dataset.
        
        Args:
            sequences: List of protein sequences
            labels: Binary label matrix (n_samples x n_go_terms)
            max_length: Maximum sequence length (will pad/truncate)
        """
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
        # Amino acid vocabulary
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: idx + 1 for idx, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['<PAD>'] = 0
        self.aa_to_idx['<UNK>'] = len(self.amino_acids) + 1
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to integer indices."""
        sequence = sequence.upper()[:self.max_length]
        encoded = [self.aa_to_idx.get(aa, self.aa_to_idx['<UNK>']) for aa in sequence]
        # Pad sequence
        if len(encoded) < self.max_length:
            encoded += [self.aa_to_idx['<PAD>']] * (self.max_length - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.encode_sequence(self.sequences[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label


class ProteinCNN(nn.Module):
    """CNN model for protein sequence classification."""
    
    def __init__(self, vocab_size: int = 22, embedding_dim: int = 128, 
                 num_classes: int = 100, max_length: int = 1000):
        """
        Initialize CNN model.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Dimension of embeddings
            num_classes: Number of GO terms to predict
            max_length: Maximum sequence length
        """
        super(ProteinCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, 256, kernel_size=7, padding=3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(512)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # Convolutional layers
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        
        # Pooling
        x1 = self.pool(x1).squeeze(-1)
        x2 = self.pool(x2).squeeze(-1)
        x3 = self.pool(x3).squeeze(-1)
        
        # Concatenate
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)


class DeepLearningModel:
    """Wrapper class for deep learning models."""
    
    def __init__(self, model_type: str = 'cnn', num_classes: int = 100, 
                 max_length: int = 1000, device: str = None):
        """
        Initialize deep learning model.
        
        Args:
            model_type: Type of model ('cnn')
            num_classes: Number of GO terms to predict
            max_length: Maximum sequence length
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.go_terms = None
        
        if model_type == 'cnn':
            self.model = ProteinCNN(num_classes=num_classes, max_length=max_length)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        print(f"Initialized {model_type} model on {self.device}")
    
    def train(self, sequences: List[str], labels: np.ndarray, go_terms: List[str],
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              validation_split: float = 0.2):
        """
        Train the model.
        
        Args:
            sequences: List of protein sequences
            labels: Binary label matrix
            go_terms: List of GO term names
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
        """
        self.go_terms = go_terms
        
        # Split data
        split_idx = int(len(sequences) * (1 - validation_split))
        train_sequences = sequences[:split_idx]
        train_labels = labels[:split_idx]
        val_sequences = sequences[split_idx:]
        val_labels = labels[split_idx:]
        
        # Create datasets
        train_dataset = ProteinSequenceDataset(train_sequences, train_labels, self.max_length)
        val_dataset = ProteinSequenceDataset(val_sequences, val_labels, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for sequences_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                sequences_batch = sequences_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sequences_batch, labels_batch in val_loader:
                    sequences_batch = sequences_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    outputs = self.model(sequences_batch)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    def predict(self, sequences: List[str], batch_size: int = 32, threshold: float = 0.5) -> np.ndarray:
        """
        Predict GO terms for sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for prediction
            threshold: Threshold for binary prediction
            
        Returns:
            Binary predictions
        """
        self.model.eval()
        predictions = []
        
        # Create dummy labels for dataset
        dummy_labels = np.zeros((len(sequences), self.num_classes))
        dataset = ProteinSequenceDataset(sequences, dummy_labels, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for sequences_batch, _ in loader:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.vstack(predictions)
        return (predictions >= threshold).astype(int)
    
    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'go_terms': self.go_terms,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'model_type': self.model_type
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.go_terms = checkpoint['go_terms']
        self.num_classes = checkpoint['num_classes']
        self.max_length = checkpoint['max_length']
        self.model_type = checkpoint['model_type']
        
        if self.model_type == 'cnn':
            self.model = ProteinCNN(num_classes=self.num_classes, max_length=self.max_length)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = self.model.to(self.device)
        print(f"Model loaded from {filepath}")
