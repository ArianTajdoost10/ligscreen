#!/usr/bin/env python3
"""
CASF-2016 Scoring Analysis Script
Supports multiple scoring methods: vina, cnnscore, cnnaffinity
"""

import os
import subprocess
import re
import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def parse_dat_file(dat_file):
    """Parse the CoreSet.dat file to extract binding affinity data."""
    data = {}
    with open(dat_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                code = parts[0]
                ki_value = parts[4]
                data[code] = ki_value
    
    return data


def calculate_score(protein_pdb, ligand_sdf, gnina_path, score_type="cnnaffinity", minimize=False):
    """
    Calculate score using gnina.
    
    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file
        gnina_path: Path to gnina executable
        score_type: Type of scoring ('vina', 'cnnscore', or 'cnnaffinity')
        minimize: Whether to minimize the ligand before scoring
    
    Returns:
        dict: Dictionary containing requested scores
    """
    # Base command
    if score_type == "vina":
        command = [
            gnina_path,
            "-r", protein_pdb,
            "-l", ligand_sdf,
            "--scoring", "vina",
            "--score_only",
        ]
    else:
        command = [
            gnina_path,
            "-r", protein_pdb,
            "-l", ligand_sdf,
            "--cnn_scoring", "rescore",
            "--score_only",
        ]
    
    # Add minimize flag if requested
    if minimize:
        command.append("--minimize")
        command.remove("--score_only")
        
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Gnina failed: {result.stderr}")
    
    scores = {}
    
    # Parse output based on score type
    for line in result.stdout.split('\n'):
        if score_type == "vina" and 'Affinity:' in line:
            match = re.search(r'[-+]?\d+\.?\d*', line)
            if match:
                scores['vina'] = float(match.group())
        elif 'CNNscore:' in line:
            match = re.search(r'[-+]?\d+\.?\d*', line)
            if match:
                scores['cnnscore'] = float(match.group())
        elif 'CNNaffinity:' in line:
            match = re.search(r'[-+]?\d+\.?\d*', line)
            if match:
                scores['cnnaffinity'] = float(match.group())
    
    # Return requested score
    if score_type in scores:
        return scores[score_type]
    else:
        raise RuntimeError(f"Could not extract {score_type} from gnina output")


def convert_ki_to_pki(ki_string):
    """
    Convert ki string (e.g., 'Kd=10nM') to pki value.
    
    Args:
        ki_string: String containing dissociation constant
    
    Returns:
        float: -log10(Kd in M)
    """
    # Extract the numeric value and unit
    if ki_string[-2] == "n":
        k = float(ki_string.split("=")[1][:-2]) * 10**(-9)
    elif ki_string[-2] == "u":
        k = float(ki_string.split("=")[1][:-2]) * 10**(-6)
    elif ki_string[-2] == "p":
        k = float(ki_string.split("=")[1][:-2]) * 10**(-12)
    elif ki_string[-2] == "m":
        k = float(ki_string.split("=")[1][:-2]) * 10**(-3)
    elif ki_string[-2] == "f":
        k = float(ki_string.split("=")[1][:-2]) * 10**(-15)
    else:
        raise ValueError(f"Unknown unit prefix in: {ki_string}")
    
    # Convert to pki
    pki = -np.log10(k)
    return pki


def validate_files(casf_base_path):
    """
    Validate CASF directory structure and find all required files.
    
    Args:
        casf_base_path: Base path to CASF-2016 directory
    
    Returns:
        tuple: (dat_file_path, coreset_dir_path)
    """
    dat_file = os.path.join(casf_base_path, "power_screening", "CoreSet.dat")
    coreset_dir = os.path.join(casf_base_path, "coreset")
    
    if not os.path.exists(dat_file):
        raise FileNotFoundError(f"CoreSet.dat not found at: {dat_file}")
    
    if not os.path.exists(coreset_dir):
        raise FileNotFoundError(f"Coreset directory not found at: {coreset_dir}")
    
    return dat_file, coreset_dir


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and correlate binding affinity scores using gnina",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py /mnt/d/casf/data/CASF-2016 ./gnina --score_type cnnaffinity
  python script.py /mnt/d/casf/data/CASF-2016 ./gnina --score_type vina --minimize
  python script.py /mnt/d/casf/data/CASF-2016 ./gnina --score_type cnnscore --minimize --output results.txt
  python script.py /mnt/d/casf/data/CASF-2016 ./gnina --minimize  # Uses default cnnaffinity
        """
    )
    
    parser.add_argument(
        "casf_path",
        help="Path to CASF-2016 base directory"
    )
    parser.add_argument(
        "gnina_path",
        help="Path to gnina executable"
    )
    parser.add_argument(
        "--score_type",
        choices=["vina", "cnnscore", "cnnaffinity"],
        default="cnnaffinity",
        help="Type of scoring method to use (default: cnnaffinity)"
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize ligand before scoring"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file to save results (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    print("Validating paths...")
    dat_file, coreset_dir = validate_files(args.casf_path)
    
    if not os.path.exists(args.gnina_path):
        raise FileNotFoundError(f"Gnina executable not found at: {args.gnina_path}")
    
    # Parse binding affinity data
    print("Parsing CoreSet.dat...")
    ki_data = parse_dat_file(dat_file)
    
    # Find valid complexes
    print("Scanning for valid protein-ligand complexes...")
    valid_codes = []
    for code in ki_data.keys():
        code_dir = os.path.join(coreset_dir, code)
        protein_pdb = os.path.join(code_dir, f"{code}_protein.pdb")
        ligand_sdf = os.path.join(code_dir, f"{code}_ligand.sdf")
        if os.path.exists(protein_pdb) and os.path.exists(ligand_sdf):
            valid_codes.append(code)
    
    print(f"Found {len(valid_codes)} valid complexes")
    print(f"Using scoring method: {args.score_type}")
    if args.minimize:
        print(f"Minimization: ENABLED")
    else:
        print(f"Minimization: DISABLED")
    print("="*60)
    
    # Calculate scores
    results = {}
    failed = []
    
    for code in tqdm(valid_codes, desc="Processing complexes"):
        ki_value = ki_data[code]
        code_dir = os.path.join(coreset_dir, code)
        protein_pdb = os.path.join(code_dir, f"{code}_protein.pdb")
        ligand_sdf = os.path.join(code_dir, f"{code}_ligand.sdf")
        
        try:
            score = calculate_score(
                protein_pdb, 
                ligand_sdf, 
                args.gnina_path, 
                args.score_type,
                minimize=args.minimize
            )
            results[code] = {
                'ki_string': ki_value,
                'score': score
            }
        except Exception as e:
            print(f"\nError processing {code}: {e}")
            failed.append(code)
            continue
    
    print("\n" + "="*60)
    print(f"Successfully processed: {len(results)}/{len(valid_codes)} complexes")
    if failed:
        print(f"Failed: {len(failed)} complexes: {', '.join(failed)}")
    print("="*60)
    
    # Convert ki to pki and prepare for correlation
    pki_values = []
    scores = []
    
    for code, data in results.items():
        try:
            pki = convert_ki_to_pki(data['ki_string'])
            pki_values.append(pki)
            scores.append(data['score'])
        except Exception as e:
            print(f"Warning: Could not convert ki for {code}: {e}")
            continue
    
    # Calculate correlations
    if len(pki_values) >= 3:
        pearson_corr, pearson_pval = pearsonr(pki_values, scores)
        spearman_corr, spearman_pval = spearmanr(pki_values, scores)
        
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        print(f"Number of data points: {len(pki_values)}")
        print(f"\nPearson Correlation:  {pearson_corr:.4f} (p-value: {pearson_pval:.4e})")
        print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_pval:.4e})")
        print("="*60)
    else:
        print("\nInsufficient data for correlation analysis")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"# CASF-2016 Scoring Analysis\n")
            f.write(f"# Score Type: {args.score_type}\n")
            f.write(f"# Minimization: {'ENABLED' if args.minimize else 'DISABLED'}\n")
            f.write(f"# Complexes Processed: {len(results)}\n")
            f.write(f"#\n")
            f.write(f"# Code\tki_String\tpki\t{args.score_type.upper()}\n")
            
            for code, data in results.items():
                try:
                    pki = convert_ki_to_pki(data['ki_string'])
                    f.write(f"{code}\t{data['ki_string']}\t{pki:.4f}\t{data['score']:.4f}\n")
                except:
                    continue
            
            if len(pki_values) >= 3:
                f.write(f"\n# Pearson:  {pearson_corr:.4f} (p={pearson_pval:.4e})\n")
                f.write(f"# Spearman: {spearman_corr:.4f} (p={spearman_pval:.4e})\n")
        
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()