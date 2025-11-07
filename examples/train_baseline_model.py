"""
Example script for training a baseline ML model for GO term prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.data_loader import ProteinDataLoader
from src.utils.feature_extraction import ProteinFeatureExtractor
from src.models.baseline_ml import BaselineMLModel


def generate_sample_data():
    """Generate sample protein sequences and GO annotations for demonstration."""
    
    # Sample protein sequences (using random sequences for demo)
    sample_proteins = [
        ('P1', 'MAKTLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWE'),
        ('P2', 'MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKQWER'),
        ('P3', 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEK'),
        ('P4', 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQW'),
        ('P5', 'MAKTLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKL'),
        ('P6', 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWE'),
        ('P7', 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKQ'),
        ('P8', 'MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKQWEKL'),
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
    }
    
    return sample_proteins, sample_annotations


def main():
    print("=" * 60)
    print("Training Baseline ML Model for GO Term Prediction")
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
    
    # Extract features
    print("\n3. Extracting features from sequences...")
    feature_extractor = ProteinFeatureExtractor()
    X = feature_extractor.extract_features_batch(sequences, include_dipeptide=False)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Split data
    print("\n4. Splitting data into train and test sets...")
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train Random Forest model
    print("\n5. Training Random Forest model...")
    rf_model = BaselineMLModel(model_type='random_forest', n_estimators=50, max_depth=5)
    rf_model.train(X_train, y_train, go_terms)
    
    # Evaluate
    print("\n6. Evaluating model on test set...")
    metrics = rf_model.evaluate(X_test, y_test)
    print("   Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"   - {metric_name}: {metric_value:.4f}")
    
    # Save model
    print("\n7. Saving model...")
    os.makedirs('models', exist_ok=True)
    rf_model.save('models/baseline_rf_model.pkl')
    
    # Make predictions on test set
    print("\n8. Making predictions...")
    predictions = rf_model.predict(X_test)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample prediction for first test protein:")
    for i, go_term in enumerate(go_terms):
        if predictions[0, i] == 1:
            print(f"   - {go_term}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
