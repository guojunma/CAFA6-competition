# CAFA6 Competition - AI Agent Instructions

## Project Overview
PANDA-3D: A hybrid GVP-Transformer model for protein function prediction using **3D structure + ESM-1v embeddings**. This is NOT a ProtT5 project - it uses ESM-1v (esm1_t34_670M_UR50S) for sequence embeddings combined with geometric features from AlphaFold2 PDB structures.

## Architecture (Read Multiple Files to Understand)
**Three-stage pipeline**: Structure encoding → Transformer encoder → Multi-label GO term prediction

1. **GVP Encoder** (`code/gvp_encoder.py`, `code/features.py`)
   - Geometric Vector Perceptron processes 3D protein coordinates (N, CA, C, O atoms)
   - Extracts structural features: node embeddings (scalar + vector), edge embeddings, dihedral angles
   - Uses pLDDT scores to mask low-confidence regions (threshold: 0.9 by default in `args.json`)
   - Outputs graph embeddings that are rotation/translation equivariant

2. **Hybrid Encoder** (`code/gvp_transformer_encoder.py`)
   - Combines GVP structural features + ESM-1v sequence embeddings (1280-dim)
   - Projects ESM embeddings to 128-dim (`encoder_embed_dim` in `args.json`)
   - Applies Transformer self-attention (0 layers by default - relies on GVP)

3. **Transformer Decoder** (`code/transformer_decoder.py`, `code/gvp_transformer.py`)
   - Cross-attention between encoder output and GO term embeddings (512-dim)
   - Predicts all GO terms simultaneously (not autoregressive)
   - Uses sigmoid activation for multi-label classification
   - Filters predictions: only GO terms with scores 0.01-1.0 are output

## Critical Workflows

### Inference (Primary Use Case)
```bash
python run_inference.py <input_directory>
```
- **Input**: Directory with `.pdb` files (AlphaFold2 structures with pLDDT in B-factor column)
- **Output**: `<input_directory>/prediction.txt` in CAFA format
- **Model**: Pre-trained weights in `panda3dcode/trained.model` (loaded in `run_inference.py` line 108)
- **Config**: `code/args.json` - DO NOT modify paths; adjust `device` and `coords_mask_plddt_th` only

**Key Implementation Details** (`run_inference.py`):
- `CreateDataset_Server` (line 112): Lazily loads PDB files, extracts sequences, computes ESM embeddings on-the-fly
- `BatchGvpesmConverter` (line 114): Pads coordinates/embeddings, masks low-pLDDT regions (→ inf for coords, -1 for pLDDT)
- `predicate()` (line 39): Inference loop with progress bar, applies sigmoid, filters GO terms
- `prediction2text()` (line 60): Formats output per CAFA spec (AUTHOR/MODEL/KEYWORDS header, sorted by score)

### Data Format Conventions
**PDB files** (`code/model_util.py:CreateDataset_Server.pdb_fasta()`):
- Must contain `SEQRES` records (parsed by BioPython)
- pLDDT scores in B-factor column (residues 61-66 of ATOM lines)
- Single chain per file assumed

**Prediction format** (`data/example/prediction.txt`):
```
AUTHOR PANDA-3D
MODEL 1
KEYWORDS sequence embedding, geometric vector perceptron, transformer.
<protein_id>    <GO_term>    <score>
...
END
```

**GO term vocabulary** (`code/data.py:Alphabet_goclean`):
- Custom alphabet with GO terms as tokens (loaded from `train_w_go_greater_50.pkl`)
- `decoder_embed_tokens_mask` in `gvp_transformer.py` filters non-GO tokens before prediction

## Project-Specific Patterns

### Coordinate Handling (`code/model_util.py:BatchGvpesmConverter.mask_coord()`)
```python
# Low-confidence residues (pLDDT < 0.9):
coords[bad_mask] = np.inf  # GVP ignores infinite coords
plddts[bad_mask] = -1      # Negative pLDDT signals masking
```

### Module Paths (CRITICAL)
All code imports use `sys.path.append('panda3dcode/')` prefix (line 2 of `run_inference.py`). When creating new scripts:
```python
import sys
sys.path.append('panda3dcode/')  # Or adjust to 'code/' if refactoring
from gvp_transformer import GVPTransformerModel
```

### Batch Conversion Pipeline
`BatchGvpesmConverter.__call__()` orchestrates:
1. `batch_esm_converter()`: Pads ESM embeddings (0 for padding)
2. `batch_seq_converter()`: Tokenizes sequences (`<go>seq<eos><pad>...`)
3. `batch_coord_converter()`: Pads coords (inf for padding, nan for length mismatch)
4. `mask_coord()`: Applies pLDDT masking

### Model I/O
- **Loading**: `model.load_state_dict(torch.load('trained.model', map_location='cuda:0'))` (line 108)
- **Config**: `args.json` deserialized to `Namespace` object (line 99-100)
- **Device handling**: Multi-GPU training (e.g., `[2,3]` in args.json), single-GPU inference

## Dependencies (Existing Setup)
Core requirements already in conda environment:
- `torch` (with CUDA), `esm` (Facebook Research), `biotite` (structure parsing)
- `pandas`, `numpy`, `tqdm`, `biopython`, `scipy`

**DO NOT install ProtT5** - this project uses ESM-1v (`esm.pretrained.esm1_t34_670M_UR50S`)

## Common Pitfalls
1. **Path confusion**: `panda3dcode/` vs `code/` - existing scripts use former, workspace uses latter
2. **ESM device**: Model moved to GPU in `CreateDataset_Server.__init__()` - ensure CUDA available
3. **PDB format**: Missing SEQRES or pLDDT scores will crash inference silently
4. **Output filtering**: Predictions are truncated to 0.01-1.0 range, sorted descending
5. **Batch size**: Inference uses batch_size=1 (line 112) due to variable-length proteins

## Key Files Reference
- `run_inference.py`: Main entry point (125 lines)
- `code/args.json`: Model hyperparameters (77 lines, paths are hardcoded for original environment)
- `code/gvp_transformer.py`: Top-level model (91 lines, see `forward()` for prediction flow)
- `code/model_util.py`: Data loading and batch conversion (179 lines)
- `code/data.py`: Tokenization and alphabet management (531 lines, see `Alphabet_goclean`)
- `data/protein_list.txt`: Target proteins for competition (224,310 UniProt IDs)
