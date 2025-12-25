import openmm
from openmm import Platform, app, unit
from openmm.app import ForceField, Modeller, Simulation, PDBFile
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openff.toolkit import Molecule
from openff.units.openmm import to_openmm, from_openmm
from rdkit import Chem
from rdkit import RDLogger
from pdbfixer import PDBFixer
from typing import List, Tuple, Optional, Dict
import numpy as np
import warnings
import tempfile
import os

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


def rdmol_parametrizer(mol: Chem.Mol,
                       force_field_files: List[str] = ['amber14-all.xml',
                                                       'implicit/gbn2.xml']):
    """Create forcefield, topology, and positions from RDKit molecule."""
    molecule = Molecule.from_rdkit(mol, 
                                   hydrogens_are_explicit=True,
                                   allow_undefined_stereo=True)
    forcefield = ForceField(*force_field_files)
    molecule.assign_partial_charges("gasteiger")
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    topology = molecule.to_topology().to_openmm()
    positions = to_openmm(molecule.conformers[0])
    forcefield.registerTemplateGenerator(smirnoff.generator)

    return forcefield, topology, positions


def create_system_from_rdkit(forcefield: ForceField, mol: Chem.Mol):
    """Create system and modeller from RDKit molecule."""
    molecule = Molecule.from_rdkit(mol, 
                                   hydrogens_are_explicit=True,
                                   allow_undefined_stereo=True)
    topology = molecule.to_topology().to_openmm()
    positions = to_openmm(molecule.conformers[0])
    system = forcefield.createSystem(topology)
    modeller = Modeller(topology, positions)
    return system, modeller


def create_system(forcefield: ForceField, topology, positions):
    """Create system and modeller from topology and positions."""
    system = forcefield.createSystem(topology)
    modeller = Modeller(topology, positions)
    return system, modeller


def load_ligand_from_sdf(sdf_file: str) -> Molecule:
    """Load ligand from SDF with robust error handling."""
    # Try loading with RDKit first for better sanitization control
    supplier = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
    rdkit_mol = None
    
    for mol in supplier:
        if mol is not None:
            try:
                # Try sanitization without kekulization
                Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                rdkit_mol = mol
                break
            except:
                # Keep unsanitized if sanitization fails
                rdkit_mol = mol
                break
    
    if rdkit_mol is None:
        raise ValueError(f"Could not load molecule from {sdf_file}")
    
    # Convert to OpenFF Molecule
    ligand_molecule = Molecule.from_rdkit(
        rdkit_mol,
        hydrogens_are_explicit=True,
        allow_undefined_stereo=True
    )
    ligand_molecule.assign_partial_charges("gasteiger")
    
    return ligand_molecule


def fix_protein_pdb(input_pdb: str, output_pdb: str = None) -> str:
    """Clean and fix protein PDB file."""
    if output_pdb is None:
        output_pdb = input_pdb.replace('.pdb', '_fixed.pdb')
    
    # Remove hydrogens and SEQRES records
    temp_no_h = input_pdb.replace('.pdb', '_temp_no_h.pdb')
    
    with open(input_pdb, "r") as f:
        lines = [line for line in f 
                if not (" H  " in line or line[12:14] == " H" or line[13:15] == " H")
                and not line.startswith("SEQRES")]
    
    with open(temp_no_h, "w") as f:
        f.writelines(lines)
    
    # Fix with PDBFixer
    fixer = PDBFixer(filename=temp_no_h)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    os.remove(temp_no_h)
    return output_pdb


def create_simulation(system,
                      modeller: Modeller,
                      platform: Optional[openmm.Platform] = None) -> Simulation:
    """Create OpenMM simulation with Langevin integrator."""
    if platform is None:
        platform = Platform.getPlatformByName('CPU')
    
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picoseconds
    )
    
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    return simulation


def save_simulation_pdb(simulation: Simulation, file_name: str) -> str:
    """Save simulation state to PDB file."""
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    topology = simulation.topology
    
    output_filename = f'{file_name}.pdb'
    PDBFile.writeFile(topology, positions, open(output_filename, 'w'))
    
    return output_filename


def calculate_charges_mm(system, 
                         mm_indices: List[int]) -> Tuple[openmm.NonbondedForce, list[float]]:
    """Extract partial charges for specified atom indices."""
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise ValueError("No NonbondedForce found in the system")

    mm_charges = []
    for i in mm_indices:
        charge, _, _ = nonbonded_force.getParticleParameters(i)
        mm_charges.append(charge.value_in_unit(unit.elementary_charge))
    
    return nonbonded_force, mm_charges


def calculate_energy(simulation: Simulation) -> float:
    """Get potential energy in kcal/mol."""
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)


def optimize_structure(simulation: Simulation, 
                       max_iterations: int = 100,
                       tolerance: float = 10.0) -> Simulation:
    """Minimize energy and return simulation."""
    simulation.minimizeEnergy(maxIterations=max_iterations)
    return simulation


def create_complex_system(protein_pdb: str,
                          ligand_sdf: str,
                          force_field_files: List[str] = ['amber14-all.xml',
                                                          'implicit/gbn2.xml']):
    """Create protein-ligand complex system with implicit solvent."""
    
    fixed_protein = fix_protein_pdb(protein_pdb)
    
    # Load protein
    protein_pdb_obj = PDBFile(fixed_protein)
    protein_topology = protein_pdb_obj.topology
    protein_positions = protein_pdb_obj.positions
    
    # Load ligand with robust handling
    ligand_molecule = load_ligand_from_sdf(ligand_sdf)
    
    # Setup forcefield
    forcefield = ForceField(*force_field_files)
    smirnoff = SMIRNOFFTemplateGenerator(molecules=ligand_molecule)
    forcefield.registerTemplateGenerator(smirnoff.generator)
    
    # Create ligand topology
    ligand_topology = ligand_molecule.to_topology().to_openmm()
    ligand_positions = to_openmm(ligand_molecule.conformers[0])
    
    # Combine protein and ligand
    modeller = Modeller(protein_topology, protein_positions)
    modeller.add(ligand_topology, ligand_positions)
    
    # Create system with implicit solvent
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        implicitSolvent=app.GBn2
    )
    
    return system, modeller, forcefield, ligand_molecule, fixed_protein


def calculate_component_energies_in_complex(protein_pdb: str,
                                            ligand_sdf: str,
                                            optimization_steps: int = 10,
                                            force_field_files: List[str] = ['amber14-all.xml',
                                                                            'implicit/gbn2.xml'],
                                            platform: Optional[openmm.Platform] = None,
                                            optimize: bool = True) -> Dict[str, float]:
    """
    Calculate energies for complex and components.
    
    Returns:
        Dict with complex_energy, protein_energy, ligand_energy, interaction_energy
    """
    
    # Create complex system
    complex_system, complex_modeller, forcefield, ligand_molecule, fixed_protein = create_complex_system(
        protein_pdb, ligand_sdf, force_field_files
    )
    
    # Setup complex simulation
    complex_simulation = create_simulation(complex_system, complex_modeller, platform)
    
    # Optimize if requested
    if optimize:
        complex_simulation = optimize_structure(complex_simulation, max_iterations=optimization_steps)
    
    # Get complex energy and positions
    complex_state = complex_simulation.context.getState(getPositions=True, getEnergy=True)
    complex_energy = complex_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    complex_positions = complex_state.getPositions()
    
    # Get atom counts
    protein_pdb_obj = PDBFile(fixed_protein)
    protein_atom_count = protein_pdb_obj.topology.getNumAtoms()
    ligand_atom_count = ligand_molecule.n_atoms
    
    # Extract component positions directly
    all_positions = list(complex_positions)
    protein_positions = all_positions[:protein_atom_count]
    ligand_positions = all_positions[protein_atom_count:protein_atom_count + ligand_atom_count]
    
    # Calculate protein energy
    protein_system = forcefield.createSystem(
        protein_pdb_obj.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        implicitSolvent=app.GBn2
    )
    protein_modeller = Modeller(protein_pdb_obj.topology, protein_positions)
    protein_simulation = create_simulation(protein_system, protein_modeller, platform)
    protein_energy = calculate_energy(protein_simulation)
    
    # Calculate ligand energy
    ligand_topology = ligand_molecule.to_topology().to_openmm()
    ligand_system = forcefield.createSystem(
        ligand_topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        implicitSolvent=app.GBn2
    )
    ligand_modeller = Modeller(ligand_topology, ligand_positions)
    ligand_simulation = create_simulation(ligand_system, ligand_modeller, platform)
    ligand_energy = calculate_energy(ligand_simulation)
    
    # Calculate interaction energy
    interaction_energy = complex_energy - (protein_energy + ligand_energy)
    
    return {
        'complex_energy': complex_energy,
        'protein_energy': protein_energy,
        'ligand_energy': ligand_energy,
        'interaction_energy': interaction_energy
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate MM interaction energies")
    parser.add_argument("--protein", required=True, help="Protein PDB file")
    parser.add_argument("--ligand", required=True, help="Ligand SDF file")
    parser.add_argument("--steps", type=int, default=10, help="Optimization steps")
    parser.add_argument("--no-optimize", action="store_false", dest="optimize")
    
    args = parser.parse_args()
    
    print("Calculating energies...")
    print("="*50)
    
    results = calculate_component_energies_in_complex(
        protein_pdb=args.protein,
        ligand_sdf=args.ligand,
        optimization_steps=args.steps,
        optimize=args.optimize
    )
    
    print(f"\nResults:")
    print(f"Complex Energy: {results['complex_energy']:.2f} kcal/mol")
    print(f"Protein Energy: {results['protein_energy']:.2f} kcal/mol")
    print(f"Ligand Energy: {results['ligand_energy']:.2f} kcal/mol")
    print(f"Interaction Energy: {results['interaction_energy']:.2f} kcal/mol")