"""
Example script for training a deep learning model for GO term prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.data_loader import ProteinDataLoader
from src.models.deep_learning import DeepLearningModel


def generate_sample_data():
    """Generate sample protein sequences and GO annotations for demonstration."""
    
    # Sample protein sequences (using longer sequences for deep learning)
    sample_proteins = [
        ('P1', 'MAKTLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWE'),
        ('P2', 'MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKQWERLQWEKLQWEKLQWEKLQWE'),
        ('P3', 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWEKLQWE'),
        ('P4', 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWERLQWEKLQWEKLQWEKLQ'),
        ('P5', 'MAKTLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWEK'),
        ('P6', 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWERLQWEKLQWEKLQWEKLQW'),
        ('P7', 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWEKLQWEQ'),
        ('P8', 'MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKQWEKLQWEKLQWEKLQWEKLQWEL'),
        ('P9', 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWEKLQWEK'),
        ('P10', 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWERLQWEKLQWEKLQWEKLQ'),
        ('P11', 'MAKTLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWEK'),
        ('P12', 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWERLQWEKLQWEKLQWEKLQW'),
        ('P13', 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWEKLQWEQ'),
        ('P14', 'MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKQWEKLQWEKLQWEKLQWEKLQWEL'),
        ('P15', 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKLQWEKLQWEKLQWEKLQWEK'),
        ('P16', 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWERLQWEKLQWEKLQWEKLQ'),
    ]
    
    # Sample GO annotations
    sample_annotations = {
        'P1': ['GO:0003674', 'GO:0005575'],
        'P2': ['GO:0003674', 'GO:0008150'],
        'P3': ['GO:0005575', 'GO:0008150'],
        'P4': ['GO:0003674'],
        'P5': ['GO:0008150'],
        'P6': ['GO:0003674', 'GO:0005575', 'GO:0008150'],
        'P7': ['GO:0005575'],
        'P8': ['GO:0003674', 'GO:0008150'],
        'P9': ['GO:0005575', 'GO:0008150'],
        'P10': ['GO:0003674'],
        'P11': ['GO:0008150'],
        'P12': ['GO:0003674', 'GO:0005575'],
        'P13': ['GO:0005575'],
        'P14': ['GO:0003674', 'GO:0008150'],
        'P15': ['GO:0005575', 'GO:0008150'],
        'P16': ['GO:0003674'],
    }
    
    return sample_proteins, sample_annotations


def main():
    print("=" * 60)
    print("Training Deep Learning Model for GO Term Prediction")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Loading sample data...")
    protein_data, annotations = generate_sample_data()
    protein_ids = [p[0] for p in protein_data]
    sequences = [p[1] for p in protein_data]
    
    print(f"   Loaded {len(sequences)} protein sequences")
    
    # Create GO term matrix
    print("\n2. Creating GO term matrix...")
    data_loader = ProteinDataLoader()
    y, go_terms = data_loader.create_go_term_matrix(protein_ids, annotations)
    print(f"   Number of unique GO terms: {len(go_terms)}")
    print(f"   GO terms: {go_terms}")
    
    # Initialize deep learning model
    print("\n3. Initializing CNN model...")
    dl_model = DeepLearningModel(
        model_type='cnn',
        num_classes=len(go_terms),
        max_length=500
    )
    
    # Train model
    print("\n4. Training model...")
    print("   Note: Using small sample data for demonstration.")
    print("   For real training, use larger datasets and more epochs.")
    
    dl_model.train(
        sequences=sequences,
        labels=y,
        go_terms=go_terms,
        epochs=5,
        batch_size=4,
        learning_rate=0.001,
        validation_split=0.2
    )
    
    # Save model
    print("\n5. Saving model...")
    os.makedirs('models', exist_ok=True)
    dl_model.save('models/deep_learning_cnn_model.pt')
    
    # Make predictions
    print("\n6. Making predictions on sample sequences...")
    test_sequences = sequences[:3]
    predictions = dl_model.predict(test_sequences, batch_size=2)
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"\n   Sample predictions:")
    for i, seq_id in enumerate(protein_ids[:3]):
        print(f"\n   Protein {seq_id}:")
        predicted_terms = [go_terms[j] for j in range(len(go_terms)) if predictions[i, j] == 1]
        if predicted_terms:
            for term in predicted_terms:
                print(f"   - {term}")
        else:
            print("   - No terms predicted")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
