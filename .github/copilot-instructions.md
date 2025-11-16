# CAFA6 Competition - AI Agent Instructions

## Project Overview
This repository contains code for the CAFA6 (Critical Assessment of Functional Annotation) competition, which focuses on predicting protein function annotations using Gene Ontology (GO) terms.

## Competition Context
- **Goal**: Predict GO term associations for protein sequences across three ontologies (Biological Process, Molecular Function, Cellular Component)
- **Data Format**: Protein sequences (FASTA), GO annotations, evaluation based on precision-recall curves
- **Key Challenge**: Multi-label classification with hierarchical ontology structure

## Development Workflow

### Data Pipeline
- Store raw competition data in `data/raw/`
- Processed features and embeddings in `data/processed/`
- Train/validation splits in `data/splits/`
- Keep original CAFA6 data files unchanged for reproducibility

### Model Development (PyTorch + ProtT5)
- Place model architectures in `models/` directory as `nn.Module` subclasses
- Use **ProtT5-XL-U50** (Rostlab/prot_t5_xl_uniref50) for sequence embeddings:
  - Load via `transformers` library: `T5EncoderModel` + `T5Tokenizer`
  - Extract per-residue embeddings (1024-dim) or use mean pooling for sequence-level features
  - Consider freezing ProtT5 weights initially, then fine-tune later if compute allows
- Architecture patterns:
  - Add projection layers after ProtT5 embeddings (e.g., `nn.Linear(1024, hidden_dim)`)
  - Use multi-head attention for capturing long-range dependencies
  - Implement separate prediction heads for each GO ontology (BPO, MFO, CCO)
- Consider GO term hierarchy in loss design:
  - Use sigmoid activation (not softmax) for multi-label prediction
  - Apply `BCEWithLogitsLoss` or implement hierarchical loss functions
  - Propagate predictions up DAG during inference
- Save checkpoints to `checkpoints/` with naming: `model_name_epoch_date.pth`
- Use `torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch})`

### Evaluation
- Implement CAFA evaluation metrics: Fmax, Smin, AUPR
- Create evaluation scripts in `evaluation/` that match official CAFA metrics
- Use `torch.no_grad()` context for inference to save memory
- Batch predictions efficiently to handle large test sets
- Track experiments using TensorBoard (built into PyTorch) or wandb

### Code Organization
```
data/               # Competition datasets
models/             # Model architectures  
training/           # Training scripts and utilities
evaluation/         # CAFA-compliant evaluation code
preprocessing/      # Feature extraction and data processing
notebooks/          # Exploratory analysis and visualization
configs/            # Model and experiment configurations
```

## Best Practices
- **Reproducibility**: Set random seeds, version dependencies, document hyperparameters
- **GO Hierarchy**: Leverage parent-child relationships in predictions (propagate predictions up the DAG)
- **Validation Strategy**: Use temporal splits to match competition setup (train on older annotations, validate on newer)
- **Submission Format**: Follow CAFA submission guidelines exactly - verify format before generating predictions

## Key Files to Create
- `requirements.txt`: Core dependencies - `torch`, `transformers`, `biopython`, `pandas`, `numpy`, `scikit-learn`, `sentencepiece`
- `train.py`: Main training entry point with CLI arguments (`argparse` or `hydra`)
- `predict.py`: Generate predictions in CAFA format using trained PyTorch models
- `config.yaml`: Centralized configuration for reproducibility (model params, learning rate, batch size)
- `models/prott5_classifier.py`: Base ProtT5 + classification head architecture

## Resources
- CAFA official website and evaluation code
- GO database and OBO format parsers (use libraries like `goatools`)
- ProtT5 model: `Rostlab/prot_t5_xl_uniref50` on HuggingFace
- PyTorch Lightning for cleaner training loops (optional but recommended)
