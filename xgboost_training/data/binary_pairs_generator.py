import numpy as np
import pandas as pd


class BinaryPairsGenerator:
    def __init__(self, config):
        self.config = config.binary_pairs
        self.min_strength = self.config['min_oscillation_strength']
        self.target_prefix = self.config['target_prefix']
        
    def create_binary_f_targets(self, df, target_pairs):
        """Create binary classification targets for F values (oscillation strength)"""
        print(f"\nCreating binary pairs targets:")
        print(f"  Threshold: f >= {self.min_strength}")
        print(f"  Processing {len(target_pairs)} pairs")
        
        binary_f_columns = []
        
        for f_col, lambda_col in target_pairs:
            f_values = df[f_col].values
            binary_f = (f_values >= self.min_strength).astype(int)
            
            binary_col = f"{self.target_prefix}_{f_col}"
            df[binary_col] = binary_f
            binary_f_columns.append(binary_col)
            
            positive_count = binary_f.sum()
            print(f"  {binary_col}: {positive_count}/{len(df)} positive ({100*positive_count/len(df):.1f}%)")
        
        print(f"\nCreated {len(binary_f_columns)} binary F classification targets")
        return df, binary_f_columns
    
    def should_run_binary_pairs(self):
        return True