import ast
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from dscribe.descriptors import SOAP
from tqdm import tqdm


class SOAPFingerPrintGenerator:
    def __init__(self, r_cut, n_max, l_max, allowed_species):
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.allowed_species = allowed_species
        self.soap = None
        self.species = None
        
    def fit(self, atom_types_list):
        self.species = sorted(self.allowed_species)
        print(f"Using species: {self.species}")
        
        self.soap = SOAP(
            species=self.species,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            periodic=False,
            sparse=False,
            average='inner'
        )
    
    def _parse_input(self, atom_types_str, atom_coords_str):
        atom_types = ast.literal_eval(atom_types_str) if isinstance(atom_types_str, str) else atom_types_str
        atom_coords = ast.literal_eval(atom_coords_str) if isinstance(atom_coords_str, str) else atom_coords_str
        return atom_types, atom_coords
    
    def _filter_atoms(self, atom_types, atom_coords):
        filtered_symbols = []
        filtered_positions = []
            
        for symbol, coord in zip(atom_types, atom_coords):
            if symbol in self.species:
                filtered_symbols.append(symbol)
                filtered_positions.append([float(x) for x in coord])
        
        return filtered_symbols, filtered_positions
    
    def coords_and_types_to_atoms(self, atom_types_str, atom_coords_str):
        atom_types, atom_coords = self._parse_input(atom_types_str, atom_coords_str)
        filtered_symbols, filtered_positions = self._filter_atoms(atom_types, atom_coords)
        
        if len(filtered_symbols) == 0:
            return None
            
        atoms = Atoms(
            symbols=filtered_symbols,
            positions=np.array(filtered_positions)
        )
        return atoms
    
    def transform(self, atom_types_list, atom_coords_list):
        descriptors = []
        failed_count = 0
        n_features = self.soap.get_number_of_features()
        
        for atom_types, atom_coords in tqdm(
                zip(atom_types_list, atom_coords_list),
                total=len(atom_types_list),
                desc="Generating SOAP descriptors"):
            
            if pd.isna(atom_types) or pd.isna(atom_coords):
                descriptors.append(np.zeros(n_features))
                continue
            
            atoms = self.coords_and_types_to_atoms(atom_types, atom_coords)
            
            if atoms is None:
                failed_count += 1
                descriptors.append(np.zeros(n_features))
                continue
            
            try:
                soap_desc = self.soap.create(atoms)
                descriptors.append(soap_desc)
            except Exception:
                failed_count += 1
                descriptors.append(np.zeros(n_features))
        
        self._print_statistics(len(atom_types_list), failed_count)
        return np.array(descriptors)
    
    def _print_statistics(self, total, failed):
        success = total - failed
        rate = (failed / total * 100) if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"SOAP Generation Statistics:")
        print(f"  Total molecules: {total}")
        print(f"  Successful: {success}")
        print(f"  Failed (after filtering): {failed}")
        print(f"  Failure rate: {rate:.2f}%")
        print(f"{'='*70}\n")