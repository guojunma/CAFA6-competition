"""
Feature extraction utilities for protein sequences.
Extracts various features from protein sequences for ML models.
"""

import numpy as np
from typing import List, Dict
from collections import Counter


class ProteinFeatureExtractor:
    """Extract features from protein sequences."""
    
    # Standard amino acids
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    def __init__(self):
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.AMINO_ACIDS)}
        
    def extract_amino_acid_composition(self, sequence: str) -> np.ndarray:
        """
        Extract amino acid composition features.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Vector of amino acid frequencies (20 dimensions)
        """
        sequence = sequence.upper()
        total = len(sequence)
        if total == 0:
            return np.zeros(20)
        
        composition = np.zeros(20)
        for aa in sequence:
            if aa in self.aa_to_idx:
                composition[self.aa_to_idx[aa]] += 1
        
        return composition / total
    
    def extract_dipeptide_composition(self, sequence: str) -> np.ndarray:
        """
        Extract dipeptide composition features.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Vector of dipeptide frequencies (400 dimensions)
        """
        sequence = sequence.upper()
        if len(sequence) < 2:
            return np.zeros(400)
        
        # Create all possible dipeptides
        dipeptides = [aa1 + aa2 for aa1 in self.AMINO_ACIDS for aa2 in self.AMINO_ACIDS]
        dipeptide_to_idx = {dp: idx for idx, dp in enumerate(dipeptides)}
        
        composition = np.zeros(400)
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            if all(aa in self.aa_to_idx for aa in dipeptide):
                composition[dipeptide_to_idx[dipeptide]] += 1
        
        total = len(sequence) - 1
        return composition / total if total > 0 else composition
    
    def extract_sequence_properties(self, sequence: str) -> np.ndarray:
        """
        Extract basic sequence properties.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Vector of sequence properties (molecular weight, charge, hydrophobicity, etc.)
        """
        sequence = sequence.upper()
        
        # Amino acid properties
        hydrophobic_aa = set('AILMFVPGW')
        charged_aa = set('DEKR')
        polar_aa = set('STNQCY')
        
        features = []
        
        # Length
        features.append(len(sequence))
        
        # Composition by property
        if len(sequence) > 0:
            features.append(sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence))
            features.append(sum(1 for aa in sequence if aa in charged_aa) / len(sequence))
            features.append(sum(1 for aa in sequence if aa in polar_aa) / len(sequence))
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def extract_all_features(self, sequence: str, include_dipeptide: bool = False) -> np.ndarray:
        """
        Extract all features from a sequence.
        
        Args:
            sequence: Protein sequence
            include_dipeptide: Whether to include dipeptide composition (increases dimensionality)
            
        Returns:
            Combined feature vector
        """
        features = []
        
        # Amino acid composition
        features.append(self.extract_amino_acid_composition(sequence))
        
        # Sequence properties
        features.append(self.extract_sequence_properties(sequence))
        
        # Dipeptide composition (optional, high dimensional)
        if include_dipeptide:
            features.append(self.extract_dipeptide_composition(sequence))
        
        return np.concatenate(features)
    
    def extract_features_batch(self, sequences: List[str], 
                              include_dipeptide: bool = False) -> np.ndarray:
        """
        Extract features from multiple sequences.
        
        Args:
            sequences: List of protein sequences
            include_dipeptide: Whether to include dipeptide composition
            
        Returns:
            Feature matrix (n_sequences x n_features)
        """
        features = []
        for seq in sequences:
            features.append(self.extract_all_features(seq, include_dipeptide))
        return np.array(features)
