"""
Classical machine learning models for GO term prediction.
Includes Random Forest, Gradient Boosting, and other ML approaches.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional
import joblib
import os


class BaselineMLModel:
    """Base class for classical ML models for GO term prediction."""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize baseline ML model.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting')
            **kwargs: Additional arguments for the underlying model
        """
        self.model_type = model_type
        self.model = None
        self.go_terms = None
        self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize the underlying ML model."""
        if self.model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Use MultiOutputClassifier for multi-label prediction
        self.model = MultiOutputClassifier(base_model, n_jobs=kwargs.get('n_jobs', -1))
    
    def train(self, X: np.ndarray, y: np.ndarray, go_terms: List[str]):
        """
        Train the model.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target matrix (n_samples x n_go_terms), binary labels
            go_terms: List of GO term names corresponding to columns in y
        """
        self.go_terms = go_terms
        print(f"Training {self.model_type} model...")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        self.model.fit(X, y)
        print("Training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict GO terms for new sequences.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            
        Returns:
            Binary predictions (n_samples x n_go_terms)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for GO terms.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            
        Returns:
            Probability matrix (n_samples x n_go_terms)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get probabilities for each output
        probas = []
        for estimator in self.model.estimators_:
            proba = estimator.predict_proba(X)
            # Get probability of positive class
            probas.append(proba[:, 1] if proba.shape[1] > 1 else proba[:, 0])
        
        return np.column_stack(probas)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'precision_micro': precision_score(y, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'go_terms': self.go_terms,
            'model_type': self.model_type
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.go_terms = data['go_terms']
        self.model_type = data['model_type']
        print(f"Model loaded from {filepath}")
