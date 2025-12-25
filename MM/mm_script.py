#!/usr/bin/env python3
"""CASF-2016 Scoring with MM Interaction Energies"""

import os
import sys
import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import warnings

# Suppress all warnings for clean progress bar
warnings.filterwarnings('ignore')

# Import MM calculation function
from mm_py import calculate_component_energies_in_complex


def parse_dat_file(dat_file):
    """Extract binding affinity data from CoreSet.dat."""
    data = {}
    with open(dat_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                data[parts[0]] = parts[4]
    
    return data


def convert_ka_to_pka(ka_string):
    """Convert Ka string (e.g., 'Kd=10nM') to pKa (-log10(Kd in M))."""
    ka_value = ka_string.split('=')[1] if '=' in ka_string else ka_string
    
    # Parse unit prefix
    unit_map = {'n': 1e-9, 'u': 1e-6, 'p': 1e-12, 'm': 1e-3, 'f': 1e-15}
    unit_prefix = ka_value[-2]
    
    if unit_prefix not in unit_map:
        raise ValueError(f"Unknown unit prefix in: {ka_string}")
    
    k = float(ka_value[:-2]) * unit_map[unit_prefix]
    return -np.log10(k)


def validate_files(casf_base_path):
    """Validate CASF directory structure."""
    dat_file = os.path.join(casf_base_path, "power_screening", "CoreSet.dat")
    coreset_dir = os.path.join(casf_base_path, "coreset")
    
    if not os.path.exists(dat_file):
        raise FileNotFoundError(f"CoreSet.dat not found at: {dat_file}")
    if not os.path.exists(coreset_dir):
        raise FileNotFoundError(f"Coreset directory not found at: {coreset_dir}")
    
    return dat_file, coreset_dir


def find_valid_complexes(ka_data, coreset_dir):
    """Find complexes with both protein and ligand files."""
    valid = []
    for code in ka_data.keys():
        code_dir = os.path.join(coreset_dir, code)
        protein_pdb = os.path.join(code_dir, f"{code}_protein.pdb")
        ligand_sdf = os.path.join(code_dir, f"{code}_ligand.sdf")
        
        if os.path.exists(protein_pdb) and os.path.exists(ligand_sdf):
            valid.append((code, protein_pdb, ligand_sdf))
    
    return valid


def process_complexes(valid_complexes, ka_data, optimize, optimization_steps):
    """Process all complexes and calculate energies."""
    results = {}
    failed = []
    
    for code, protein_pdb, ligand_sdf in tqdm(valid_complexes, 
                                               desc="Processing", 
                                               file=sys.stdout,
                                               ncols=80):
        try:
            # Calculate energies using mm_py function
            energy_results = calculate_component_energies_in_complex(
                protein_pdb=protein_pdb,
                ligand_sdf=ligand_sdf,
                optimization_steps=optimization_steps,
                optimize=optimize
            )
            
            # Store results with negative interaction energy as binding score
            results[code] = {
                'ka_string': ka_data[code],
                'mm_score': -energy_results['interaction_energy'],
                'complex_energy': energy_results['complex_energy'],
                'protein_energy': energy_results['protein_energy'],
                'ligand_energy': energy_results['ligand_energy'],
                'interaction_energy': energy_results['interaction_energy'],
                'protein_file': protein_pdb,
                'ligand_file': ligand_sdf
            }
            
        except Exception as e:
            failed.append((code, str(e)))
    
    return results, failed


def calculate_correlations(results):
    """Calculate Pearson and Spearman correlations."""
    pka_values = []
    mm_scores = []
    
    for code, data in results.items():
        try:
            pka = convert_ka_to_pka(data['ka_string'])
            pka_values.append(pka)
            mm_scores.append(data['mm_score'])
        except Exception as e:
            print(f"Warning: Could not convert Ka for {code}: {e}")
    
    if len(pka_values) < 3:
        return None
    
    pearson_corr, pearson_pval = pearsonr(pka_values, mm_scores)
    spearman_corr, spearman_pval = spearmanr(pka_values, mm_scores)
    rmse = np.sqrt(np.mean((np.array(pka_values) - np.array(mm_scores))**2))
    
    return {
        'n': len(pka_values),
        'pearson': pearson_corr,
        'pearson_p': pearson_pval,
        'spearman': spearman_corr,
        'spearman_p': spearman_pval,
        'rmse': rmse,
        'pka_values': pka_values,
        'mm_scores': mm_scores
    }


def save_results(results, failed, correlations, output_file, args):
    """Save detailed results to file."""
    with open(output_file, 'w') as f:
        # Header
        f.write(f"# CASF-2016 MM Scoring Analysis\n")
        f.write(f"# Optimization: {args.optimize}\n")
        f.write(f"# Steps: {args.steps}\n")
        f.write(f"# Processed: {len(results)}\n")
        f.write(f"# Failed: {len(failed)}\n")
        
        if failed:
            failed_codes = ','.join([c for c, _ in failed[:10]])
            f.write(f"# Failed codes: {failed_codes}\n")
            if len(failed) > 10:
                f.write(f"# ... and {len(failed)-10} more\n")
        
        # Data header
        f.write(f"\nCode\tKa_String\tpKa\tMM_Score\tInteraction_E\tComplex_E\tProtein_E\tLigand_E\n")
        
        for code, data in results.items():
            try:
                pka = convert_ka_to_pka(data['ka_string'])
                f.write(f"{code}\t{data['ka_string']}\t{pka:.4f}\t{data['mm_score']:.4f}\t")
                f.write(f"{data['interaction_energy']:.4f}\t{data['complex_energy']:.4f}\t")
                f.write(f"{data['protein_energy']:.4f}\t{data['ligand_energy']:.4f}\n")
            except:
                f.write(f"{code}\t{data['ka_string']}\tERROR\t{data['mm_score']:.4f}\t")
                f.write(f"{data['interaction_energy']:.4f}\t{data['complex_energy']:.4f}\t")
                f.write(f"{data['protein_energy']:.4f}\t{data['ligand_energy']:.4f}\n")
        
        # Statistics
        if correlations:
            f.write(f"\n# Statistics:\n")
            f.write(f"# N: {correlations['n']}\n")
            f.write(f"# Pearson:  {correlations['pearson']:.4f} (p={correlations['pearson_p']:.4e})\n")
            f.write(f"# Spearman: {correlations['spearman']:.4f} (p={correlations['spearman_p']:.4e})\n")
            f.write(f"# RMSE: {correlations['rmse']:.4f}\n")
    
    # Save simple version for plotting
    simple_file = output_file.replace('.txt', '_simple.txt')
    with open(simple_file, 'w') as f:
        f.write("Code\tpKa\tMM_Score\n")
        for code, data in results.items():
            try:
                pka = convert_ka_to_pka(data['ka_string'])
                f.write(f"{code}\t{pka:.4f}\t{data['mm_score']:.4f}\n")
            except:
                pass
    
    return simple_file


def main():
    parser = argparse.ArgumentParser(
        description="Calculate MM interaction energies for CASF-2016 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("casf_path", help="CASF-2016 base directory")
    parser.add_argument("--optimize", action="store_true", default=True,
                       help="Optimize structures (default: True)")
    parser.add_argument("--no-optimize", action="store_false", dest="optimize",
                       help="Skip optimization")
    parser.add_argument("--steps", type=int, default=10,
                       help="Optimization steps (default: 10)")
    parser.add_argument("--output", default="mm_scoring_results.txt",
                       help="Output file (default: mm_scoring_results.txt)")
    
    args = parser.parse_args()
    
    # Validate and load data
    print("Validating paths...")
    dat_file, coreset_dir = validate_files(args.casf_path)
    
    print("Parsing CoreSet.dat...")
    ka_data = parse_dat_file(dat_file)
    
    print("Finding valid complexes...")
    valid_complexes = find_valid_complexes(ka_data, coreset_dir)
    
    print(f"\nFound {len(valid_complexes)} valid complexes")
    print(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
    if args.optimize:
        print(f"Steps: {args.steps}")
    print("="*60 + "\n")
    
    # Process all complexes
    results, failed = process_complexes(valid_complexes, ka_data, 
                                       args.optimize, args.steps)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Successfully processed: {len(results)}/{len(valid_complexes)}")
    if failed:
        print(f"Failed: {len(failed)} complexes")
        for code, error in failed[:5]:
            print(f"  {code}: {error[:50]}...")
        if len(failed) > 5:
            print(f"  ... and {len(failed)-5} more")
    print("="*60)
    
    # Calculate correlations
    correlations = calculate_correlations(results)
    
    if correlations:
        print("\nCORRELATION ANALYSIS")
        print("="*60)
        print(f"N: {correlations['n']}")
        print(f"Pearson:  {correlations['pearson']:.4f} (p={correlations['pearson_p']:.4e})")
        print(f"Spearman: {correlations['spearman']:.4f} (p={correlations['spearman_p']:.4e})")
        print(f"RMSE: {correlations['rmse']:.4f}")
        print("="*60)
    else:
        print("\nInsufficient data for correlation analysis")
    
    # Save results
    simple_file = save_results(results, failed, correlations, args.output, args)
    print(f"\nDetailed results: {args.output}")
    print(f"Simple results: {simple_file}")


if __name__ == "__main__":
    main()