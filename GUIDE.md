# CAFA6 Competition - GO Term Prediction Models

## Quick Start Guide

This guide will help you get started with training and using machine learning models for GO term prediction.

## Installation

```bash
# Clone the repository
git clone https://github.com/guojunma/CAFA6-competition.git
cd CAFA6-competition

# Install dependencies
pip install -r requirements.txt

# For deep learning models, ensure PyTorch is installed
pip install torch
```

## Basic Usage

### 1. Using Pre-built Examples

Run the baseline ML model example:
```bash
python examples/train_baseline_model.py
```

Run the complete workflow:
```bash
python examples/complete_workflow.py
```

Run the deep learning model (requires PyTorch):
```bash
python examples/train_deep_learning_model.py
```

### 2. Using Your Own Data

#### Step 1: Prepare your data

Create a FASTA file with protein sequences (`proteins.fasta`):
```
>Protein1
MAKTLKERTLQWEKLQKLEKWEKLQ
>Protein2
MLKERLQWEKLQKLEKWEKLQKLEKQ
```

Create an annotation file (`annotations.txt`):
```
Protein1	GO:0003674,GO:0005575
Protein2	GO:0003674,GO:0008150
```

#### Step 2: Load and process data

```python
from src.data.data_loader import ProteinDataLoader
from src.utils.feature_extraction import ProteinFeatureExtractor
from src.models.baseline_ml import BaselineMLModel

# Load data
loader = ProteinDataLoader()
sequences = loader.load_fasta('proteins.fasta')
annotations = loader.load_go_annotations('annotations.txt')

# Extract features
extractor = ProteinFeatureExtractor()
protein_ids = [s[0] for s in sequences]
seqs = [s[1] for s in sequences]
X = extractor.extract_features_batch(seqs)

# Create labels
y, go_terms = loader.create_go_term_matrix(protein_ids, annotations)
```

#### Step 3: Train a model

```python
# Initialize model
model = BaselineMLModel(
    model_type='random_forest',
    n_estimators=100,
    max_depth=10
)

# Train
model.train(X, y, go_terms)

# Save
model.save('my_model.pkl')
```

#### Step 4: Make predictions

```python
# Load model
model = BaselineMLModel()
model.load('my_model.pkl')

# Predict on new sequences
new_sequences = ['MAKTLKERLQWEKLQKLEKWEKLQ']
new_features = extractor.extract_features_batch(new_sequences)
predictions = model.predict(new_features)

# Get probabilities
probabilities = model.predict_proba(new_features)
```

## Model Types

### Random Forest
- **Best for**: General-purpose classification, robust to overfitting
- **Parameters**: `n_estimators`, `max_depth`, `random_state`
- **Usage**: `BaselineMLModel(model_type='random_forest')`

### Gradient Boosting
- **Best for**: High accuracy when you have enough data
- **Parameters**: `n_estimators`, `max_depth`, `learning_rate`
- **Usage**: `BaselineMLModel(model_type='gradient_boosting')`

### CNN (Deep Learning)
- **Best for**: Large datasets with long protein sequences
- **Requires**: PyTorch
- **Usage**: `DeepLearningModel(model_type='cnn')`

## Feature Extraction Options

### Basic Features (Fast, 24 dimensions)
```python
features = extractor.extract_all_features(sequence, include_dipeptide=False)
```
Includes:
- 20 amino acid composition frequencies
- 4 sequence properties (length, hydrophobic%, charged%, polar%)

### Extended Features (Slower, 424 dimensions)
```python
features = extractor.extract_all_features(sequence, include_dipeptide=True)
```
Includes:
- Basic features (24)
- 400 dipeptide composition frequencies

## Evaluation Metrics

The models provide the following metrics:
- **Precision** (micro/macro): Fraction of predicted GO terms that are correct
- **Recall** (micro/macro): Fraction of actual GO terms that were predicted
- **F1-score** (micro/macro): Harmonic mean of precision and recall

```python
metrics = model.evaluate(X_test, y_test)
print(f"F1-score: {metrics['f1_micro']:.4f}")
```

## Tips for Better Performance

1. **Data Quality**
   - Ensure GO annotations are accurate and up-to-date
   - Remove redundant or obsolete GO terms
   - Use consistent protein sequence formats

2. **Feature Engineering**
   - Try different feature combinations
   - Consider sequence length when choosing features
   - Use dipeptide features for more discriminative power

3. **Model Selection**
   - Start with Random Forest for baseline
   - Use Gradient Boosting for better accuracy
   - Try deep learning for large datasets (>10,000 proteins)

4. **Hyperparameter Tuning**
   - Adjust `n_estimators` (more trees = better performance but slower)
   - Tune `max_depth` to control overfitting
   - Use cross-validation for optimal parameters

5. **Handling Imbalanced Data**
   - Some GO terms are rare - consider class weights
   - Use stratified sampling when splitting data
   - Focus on micro-averaged metrics for overall performance

## Troubleshooting

### Issue: Model predictions are all zeros
**Solution**: Check if your training data has enough positive examples for each GO term. Try increasing training data or reducing the number of GO terms.

### Issue: Out of memory errors
**Solution**: 
- Reduce batch size for deep learning models
- Use `include_dipeptide=False` for feature extraction
- Process data in chunks

### Issue: Poor performance on test set
**Solution**:
- Ensure train/test split is representative
- Check for data leakage
- Try hyperparameter tuning
- Collect more training data

## Advanced Topics

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# Convert to sklearn compatible format
model = BaselineMLModel().model
scores = cross_val_score(model, X, y, cv=5, scoring='f1_micro')
```

### Custom Feature Extraction
```python
class CustomFeatureExtractor(ProteinFeatureExtractor):
    def extract_custom_features(self, sequence):
        # Add your custom features here
        pass
```

### Ensemble Models
```python
# Train multiple models and combine predictions
rf_pred = rf_model.predict_proba(X_test)
gb_pred = gb_model.predict_proba(X_test)
ensemble_pred = (rf_pred + gb_pred) / 2
```

## Contributing

We welcome contributions! Areas for improvement:
- Additional feature extraction methods
- New model architectures
- Hyperparameter optimization
- Evaluation metrics
- Documentation

## Support

For questions or issues:
1. Check the examples in `examples/` directory
2. Review the API documentation in source files
3. Open an issue on GitHub

## References

- CAFA challenge: https://biofunctionprediction.org/cafa/
- Gene Ontology: http://geneontology.org/
- BioPython: https://biopython.org/

## License

This project is open source under the MIT License.
