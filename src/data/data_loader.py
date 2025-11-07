"""
Data preprocessing utilities for CAFA6 competition.
Handles loading and preprocessing of protein sequences and GO term annotations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq


class ProteinDataLoader:
    """Load and preprocess protein sequences and GO term annotations."""
    
    def __init__(self):
        self.proteins = []
        self.sequences = []
        self.go_terms = {}
        
    def load_fasta(self, fasta_file: str) -> List[Tuple[str, str]]:
        """
        Load protein sequences from FASTA file.
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            List of (protein_id, sequence) tuples
        """
        sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append((record.id, str(record.seq)))
        return sequences
    
    def load_go_annotations(self, annotation_file: str) -> Dict[str, List[str]]:
        """
        Load GO term annotations from file.
        
        Expected format: protein_id\tGO:term1,GO:term2,...
        
        Args:
            annotation_file: Path to annotation file
            
        Returns:
            Dictionary mapping protein IDs to lists of GO terms
        """
        annotations = {}
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    protein_id = parts[0]
                    go_terms = parts[1].split(',')
                    annotations[protein_id] = [term.strip() for term in go_terms]
        return annotations
    
    def create_go_term_matrix(self, proteins: List[str], 
                              annotations: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
        """
        Create binary matrix for GO term annotations.
        
        Args:
            proteins: List of protein IDs
            annotations: Dictionary mapping protein IDs to GO terms
            
        Returns:
            Tuple of (binary matrix, list of GO terms)
        """
        # Collect all unique GO terms
        all_go_terms = set()
        for go_list in annotations.values():
            all_go_terms.update(go_list)
        go_terms_list = sorted(list(all_go_terms))
        go_term_to_idx = {term: idx for idx, term in enumerate(go_terms_list)}
        
        # Create binary matrix
        matrix = np.zeros((len(proteins), len(go_terms_list)), dtype=np.int32)
        for i, protein in enumerate(proteins):
            if protein in annotations:
                for go_term in annotations[protein]:
                    if go_term in go_term_to_idx:
                        matrix[i, go_term_to_idx[go_term]] = 1
        
        return matrix, go_terms_list
