"""
Comprehensive example demonstrating the complete workflow:
1. Data preparation
2. Feature extraction
3. Model training
4. Evaluation
5. Prediction on new sequences
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.data_loader import ProteinDataLoader
from src.utils.feature_extraction import ProteinFeatureExtractor
from src.models.baseline_ml import BaselineMLModel


def main():
    print("=" * 70)
    print("CAFA6 Competition - Complete GO Term Prediction Workflow")
    print("=" * 70)
    
    # Step 1: Prepare sample data
    print("\n[Step 1] Preparing sample data...")
    
    # Sample proteins with diverse sequences
    proteins = {
        'P1': 'MAKTLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWE',
        'P2': 'MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKQWER',
        'P3': 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEK',
        'P4': 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQW',
        'P5': 'MAKTLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKL',
        'P6': 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWE',
        'P7': 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKQ',
        'P8': 'MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKQWEKL',
        'P9': 'MAKTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWEKL',
        'P10': 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQW',
        'P11': 'MAKTLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKL',
        'P12': 'MLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWEKLQWE',
    }
    
    # GO term annotations (training data)
    annotations = {
        'P1': ['GO:0003674', 'GO:0005575'],  # Molecular function, Cellular component
        'P2': ['GO:0003674', 'GO:0008150'],  # Molecular function, Biological process
        'P3': ['GO:0005575', 'GO:0008150'],  # Cellular component, Biological process
        'P4': ['GO:0003674'],
        'P5': ['GO:0008150'],
        'P6': ['GO:0003674', 'GO:0005575', 'GO:0008150'],
        'P7': ['GO:0005575'],
        'P8': ['GO:0003674', 'GO:0008150'],
        'P9': ['GO:0005575', 'GO:0008150'],
        'P10': ['GO:0003674'],
    }
    
    print(f"   Total proteins: {len(proteins)}")
    print(f"   Annotated proteins: {len(annotations)}")
    print(f"   Unannotated proteins: {len(proteins) - len(annotations)}")
    
    # Step 2: Create GO term matrix
    print("\n[Step 2] Creating GO term matrix...")
    data_loader = ProteinDataLoader()
    protein_ids = list(proteins.keys())
    sequences = list(proteins.values())
    
    y, go_terms = data_loader.create_go_term_matrix(protein_ids, annotations)
    print(f"   Unique GO terms: {len(go_terms)}")
    print(f"   GO terms: {go_terms}")
    print(f"   Label matrix shape: {y.shape}")
    
    # Step 3: Extract features
    print("\n[Step 3] Extracting features...")
    feature_extractor = ProteinFeatureExtractor()
    
    print("   Extracting amino acid composition...")
    X = feature_extractor.extract_features_batch(sequences, include_dipeptide=False)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Features per protein: {X.shape[1]}")
    print(f"     - 20 amino acid frequencies")
    print(f"     - 4 sequence properties (length, hydrophobic%, charged%, polar%)")
    
    # Step 4: Split data
    print("\n[Step 4] Splitting data...")
    # Use first 10 proteins for training, last 2 for testing
    train_size = 10
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"   Training set: {X_train.shape[0]} proteins")
    print(f"   Test set: {X_test.shape[0]} proteins")
    
    # Step 5: Train models
    print("\n[Step 5] Training models...")
    
    # Train Random Forest
    print("\n   5a. Training Random Forest...")
    rf_model = BaselineMLModel(
        model_type='random_forest',
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.train(X_train, y_train, go_terms)
    
    # Train Gradient Boosting
    print("\n   5b. Training Gradient Boosting...")
    gb_model = BaselineMLModel(
        model_type='gradient_boosting',
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    gb_model.train(X_train, y_train, go_terms)
    
    # Step 6: Evaluate models
    print("\n[Step 6] Evaluating models on test set...")
    
    print("\n   Random Forest Metrics:")
    rf_metrics = rf_model.evaluate(X_test, y_test)
    for metric, value in rf_metrics.items():
        print(f"      {metric}: {value:.4f}")
    
    print("\n   Gradient Boosting Metrics:")
    gb_metrics = gb_model.evaluate(X_test, y_test)
    for metric, value in gb_metrics.items():
        print(f"      {metric}: {value:.4f}")
    
    # Step 7: Make predictions
    print("\n[Step 7] Making predictions...")
    
    # Predict with Random Forest
    rf_predictions = rf_model.predict(X_test)
    rf_probabilities = rf_model.predict_proba(X_test)
    
    print(f"\n   Predictions for test proteins (Random Forest):")
    for i, protein_id in enumerate(protein_ids[train_size:]):
        print(f"\n   {protein_id}:")
        print(f"      Sequence: {sequences[train_size + i][:30]}...")
        print(f"      Predicted GO terms:")
        
        predicted_terms = []
        for j, go_term in enumerate(go_terms):
            if rf_predictions[i, j] == 1:
                prob = rf_probabilities[i, j]
                predicted_terms.append((go_term, prob))
        
        if predicted_terms:
            for term, prob in sorted(predicted_terms, key=lambda x: x[1], reverse=True):
                print(f"         - {term} (confidence: {prob:.3f})")
        else:
            print(f"         - No terms predicted")
        
        # Show true labels if available
        if protein_id in annotations:
            print(f"      True GO terms: {annotations[protein_id]}")
    
    # Step 8: Save models
    print("\n[Step 8] Saving models...")
    os.makedirs('models', exist_ok=True)
    
    rf_model.save('models/random_forest_model.pkl')
    gb_model.save('models/gradient_boosting_model.pkl')
    
    print("   ✓ Models saved successfully")
    
    # Step 9: Demonstrate loading and using saved model
    print("\n[Step 9] Loading and using saved model...")
    
    loaded_model = BaselineMLModel()
    loaded_model.load('models/random_forest_model.pkl')
    
    # Make prediction with loaded model
    test_seq = 'MAKTLKERTLQWEKLQKLEKWEKLQKLEKQEKQWLEKQWLEKQWLEKQWEKLQWE'
    test_features = feature_extractor.extract_all_features(test_seq).reshape(1, -1)
    loaded_pred = loaded_model.predict(test_features)
    
    print(f"   Loaded model prediction for new sequence:")
    print(f"      Sequence: {test_seq[:30]}...")
    predicted = [go_terms[j] for j in range(len(go_terms)) if loaded_pred[0, j] == 1]
    if predicted:
        print(f"      Predicted GO terms: {predicted}")
    else:
        print(f"      No terms predicted")
    
    print("\n" + "=" * 70)
    print("Workflow completed successfully!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - Trained 2 models (Random Forest, Gradient Boosting)")
    print(f"  - Best F1-score (micro): {max(rf_metrics['f1_micro'], gb_metrics['f1_micro']):.4f}")
    print(f"  - Models saved to: models/")
    print(f"  - Ready for production use!")
    print("=" * 70)


if __name__ == '__main__':
    main()
