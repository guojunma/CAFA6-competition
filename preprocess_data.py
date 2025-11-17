import sys
tag_ = 'code/'
sys.path.append(tag_)

import numpy as np
import pandas as pd
import os
import pickle
import argparse
from tqdm import tqdm
import glob
import torch
import esm
from Bio import SeqIO

from util import load_coords


def parse_plddt(pdb_file):
    """Extract pLDDT scores from PDB file B-factor column"""
    plddts = []
    with open(pdb_file, 'r') as fh:
        for line in fh:
            if line[:4] == 'ATOM' and line[13:15] == 'CA':
                plddts.append(np.float32(line[61:66]))
    return np.array(plddts)


def pdb_to_fasta(pdb_file):
    """Extract sequence from PDB file (assumes single chain)"""
    for record in SeqIO.parse(pdb_file, "pdb-seqres"):
        return record.annotations["chain"], str(record.seq)
    return None, None


def compute_esm_embedding(sequence, model, batch_converter, device):
    """Compute ESM-1v embeddings for a sequence"""
    data = [('', sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[34])
    
    token_embeddings = results["representations"][34]
    # Remove BOS token, keep sequence embeddings
    pred = token_embeddings[0].cpu().numpy()[1:len(sequence)+1]
    return pred


def process_protein(pdb_file, protein_id, go_annotations, esm_model, batch_converter, device):
    """Process a single protein: extract coords, sequence, pLDDT, and compute ESM embeddings"""
    try:
        # Extract sequence and chain
        chain, seq = pdb_to_fasta(pdb_file)
        if seq is None:
            print(f"Warning: Could not extract sequence from {pdb_file}")
            return None
        
        # Load coordinates
        coords, _ = load_coords(pdb_file, chain)
        
        # Parse pLDDT scores
        plddts = parse_plddt(pdb_file)
        
        # Compute ESM embeddings
        esm_embedding = compute_esm_embedding(seq, esm_model, batch_converter, device)
        
        # Get GO annotations for this protein
        true_go = go_annotations.get(protein_id, [])
        
        # Create feature dictionary
        feat_dict = {
            'protein': protein_id,
            'seq': seq,
            'coords': coords,
            'plddt': plddts,
            'esm1v': esm_embedding,
            'true_go': true_go
        }
        
        return feat_dict
    
    except Exception as e:
        print(f"Error processing {protein_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Preprocess protein structures for PANDA-3D training')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory containing PDB files')
    parser.add_argument('--annotations_file', type=str, required=True, 
                        help='CSV/TSV file with protein_id and GO annotations')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed features')
    parser.add_argument('--output_df', type=str, required=True, help='Output dataframe pickle file name (e.g., train_df.pkl)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for ESM model')
    parser.add_argument('--annotation_col', type=str, default='GO_terms', 
                        help='Column name containing GO annotations (comma-separated)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ESM model
    print("Loading ESM-1v model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
    esm_model = esm_model.to(device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"Using device: {device}")
    
    # Load GO annotations
    print("Loading GO annotations...")
    if args.annotations_file.endswith('.csv'):
        annotations_df = pd.read_csv(args.annotations_file)
    elif args.annotations_file.endswith('.tsv') or args.annotations_file.endswith('.txt'):
        annotations_df = pd.read_csv(args.annotations_file, sep='\t')
    else:
        annotations_df = pd.read_pickle(args.annotations_file)
    
    # Create GO annotations dictionary
    go_annotations = {}
    for _, row in annotations_df.iterrows():
        protein_id = row['protein_id'] if 'protein_id' in row else row.iloc[0]
        go_terms = row[args.annotation_col]
        
        # Parse GO terms (assuming comma-separated string)
        if isinstance(go_terms, str):
            go_list = [go.strip() for go in go_terms.split(',') if go.strip()]
        elif isinstance(go_terms, list):
            go_list = go_terms
        else:
            go_list = []
        
        go_annotations[protein_id] = go_list
    
    print(f"Loaded annotations for {len(go_annotations)} proteins")
    
    # Find all PDB files
    pdb_files = glob.glob(os.path.join(args.pdb_dir, '*.pdb'))
    print(f"Found {len(pdb_files)} PDB files")
    
    # Process each protein
    processed_data = []
    failed_proteins = []
    
    for pdb_file in tqdm(pdb_files, desc="Processing proteins"):
        protein_id = os.path.basename(pdb_file).replace('.pdb', '')
        
        # Skip if no annotations
        if protein_id not in go_annotations:
            print(f"Skipping {protein_id}: no annotations found")
            continue
        
        feat_dict = process_protein(pdb_file, protein_id, go_annotations, 
                                    esm_model, batch_converter, device)
        
        if feat_dict is not None:
            # Save individual feature file
            # Use subdirectory based on first 2 characters of protein ID
            subdir = os.path.join(args.output_dir, protein_id[:2])
            os.makedirs(subdir, exist_ok=True)
            
            feat_path = os.path.join(subdir, f'{protein_id}.pkl')
            with open(feat_path, 'wb') as f:
                pickle.dump(feat_dict, f)
            
            processed_data.append({'protein': protein_id})
        else:
            failed_proteins.append(protein_id)
    
    # Create dataframe
    df = pd.DataFrame(processed_data)
    df_path = os.path.join(args.output_dir, args.output_df)
    df.to_pickle(df_path)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {len(processed_data)} proteins")
    print(f"Failed: {len(failed_proteins)} proteins")
    print(f"Dataframe saved to: {df_path}")
    print(f"Features saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    if failed_proteins:
        print(f"\nFailed proteins: {failed_proteins[:10]}" + 
              (f"... and {len(failed_proteins)-10} more" if len(failed_proteins) > 10 else ""))


if __name__ == "__main__":
    main()
