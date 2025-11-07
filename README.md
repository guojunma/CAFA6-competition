# CAFA6 Competition - GO Term Prediction

This repository contains machine learning models for predicting Gene Ontology (GO) terms for proteins, developed for the CAFA6 (Critical Assessment of Functional Annotation) competition.

## Overview

The CAFA challenge focuses on protein function prediction, specifically predicting Gene Ontology (GO) terms that describe molecular functions, biological processes, and cellular components. This project implements both classical machine learning and deep learning approaches for this task.

## Features

- **Data Loading & Preprocessing**: Utilities for loading protein sequences (FASTA) and GO annotations
- **Feature Extraction**: Multiple feature extraction methods including:
  - Amino acid composition
  - Dipeptide composition
  - Sequence properties (length, hydrophobicity, charge, etc.)
- **Classical ML Models**: 
  - Random Forest
  - Gradient Boosting
  - Multi-label classification support
- **Deep Learning Models**:
  - CNN-based sequence classifier
  - Automatic sequence encoding and padding
- **Model Evaluation**: Precision, recall, and F1-score metrics
- **Model Persistence**: Save and load trained models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/guojunma/CAFA6-competition.git
cd CAFA6-competition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
CAFA6-competition/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py          # Data loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_ml.py          # Classical ML models
│   │   └── deep_learning.py        # Deep learning models
│   ├── utils/
│   │   ├── __init__.py
│   │   └── feature_extraction.py   # Feature extraction utilities
│   └── __init__.py
├── examples/
│   ├── train_baseline_model.py     # Example: Train ML model
│   └── train_deep_learning_model.py # Example: Train DL model
├── requirements.txt
└── README.md
```

## Usage

### Training a Baseline ML Model

```python
from src.data.data_loader import ProteinDataLoader
from src.utils.feature_extraction import ProteinFeatureExtractor
from src.models.baseline_ml import BaselineMLModel

# Load data
data_loader = ProteinDataLoader()
sequences = data_loader.load_fasta('proteins.fasta')
annotations = data_loader.load_go_annotations('annotations.txt')

# Extract features
feature_extractor = ProteinFeatureExtractor()
protein_ids = [s[0] for s in sequences]
seqs = [s[1] for s in sequences]
X = feature_extractor.extract_features_batch(seqs)

# Create labels
y, go_terms = data_loader.create_go_term_matrix(protein_ids, annotations)

# Train model
model = BaselineMLModel(model_type='random_forest', n_estimators=100)
model.train(X, y, go_terms)

# Make predictions
predictions = model.predict(X)

# Save model
model.save('models/rf_model.pkl')
```

### Training a Deep Learning Model

```python
from src.data.data_loader import ProteinDataLoader
from src.models.deep_learning import DeepLearningModel

# Load data
data_loader = ProteinDataLoader()
sequences = data_loader.load_fasta('proteins.fasta')
annotations = data_loader.load_go_annotations('annotations.txt')

# Prepare data
protein_ids = [s[0] for s in sequences]
seqs = [s[1] for s in sequences]
y, go_terms = data_loader.create_go_term_matrix(protein_ids, annotations)

# Initialize and train model
model = DeepLearningModel(model_type='cnn', num_classes=len(go_terms))
model.train(seqs, y, go_terms, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(seqs)

# Save model
model.save('models/cnn_model.pt')
```

### Running Examples

Run the baseline ML model example:
```bash
python examples/train_baseline_model.py
```

Run the deep learning model example:
```bash
python examples/train_deep_learning_model.py
```

## Data Format

### FASTA File Format
Standard FASTA format for protein sequences:
```
>Protein1
MAKTLKERTLQWEKLQKLEKWEKLQKLEKQEKQ
>Protein2
MLKERLQWEKLQKLEKWEKLQKLEKQEKQWLEK
```

### GO Annotations Format
Tab-separated file with protein IDs and GO terms:
```
Protein1	GO:0003674,GO:0005575
Protein2	GO:0003674,GO:0008150
Protein3	GO:0005575,GO:0008150
```

## Models

### Baseline ML Models
- **Random Forest**: Ensemble of decision trees with multi-label support
- **Gradient Boosting**: Sequential ensemble method for better accuracy

### Deep Learning Models
- **CNN**: Convolutional Neural Network with multiple kernel sizes
  - Embedding layer for amino acid sequences
  - Multiple convolutional layers (kernel sizes: 3, 5, 7)
  - Global max pooling
  - Fully connected layers with dropout

## Evaluation Metrics

The models are evaluated using:
- Precision (micro and macro averages)
- Recall (micro and macro averages)
- F1-score (micro and macro averages)

## Dependencies

- numpy
- pandas
- scikit-learn
- biopython
- torch
- transformers
- scipy
- joblib
- tqdm

See `requirements.txt` for specific versions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research, please cite the CAFA competition:
```
Zhou, N., Jiang, Y., Bergquist, T.R. et al. The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens. Genome Biol 20, 244 (2019).
```

## Contact

For questions or issues, please open an issue on GitHub.