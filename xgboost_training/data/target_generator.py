import numpy as np
import pandas as pd


class TargetGenerator:
    def __init__(self, config):
        self.task = config.task
        self.custom_tasks = getattr(config, 'custom_tasks', {})
        self.num_pairs = config.targets['num_pairs']
        self.binary_config = getattr(config, 'binary', {})
        self.min_strength = self.binary_config.get('min_oscillation_strength', 0.01)
        self.wavelength_range = self.binary_config.get('wavelength_range', [350, 650])
        self.target_name = self.binary_config.get('target_name', 'has_uvvis_peak')
        
    def get_all_pairs(self, df):
        f_cols = [col for col in df.columns if col.startswith('f_') and col.endswith('_gasphase')]
        lambda_cols = [col for col in df.columns if col.startswith('lambda_') and col.endswith('_gasphase')]
        
        pairs = []
        max_available = min(len(f_cols), len(lambda_cols))
        
        for i in range(1, max_available + 1):
            f_col = f'f_{i}_gasphase'
            lambda_col = f'lambda_{i}_gasphase'
            
            if f_col in df.columns and lambda_col in df.columns:
                pairs.append((f_col, lambda_col))
        
        return pairs
    
    def get_limited_pairs(self, df):
        all_pairs = self.get_all_pairs(df)
        limited = all_pairs[:self.num_pairs]
        
        print(f"Found {len(all_pairs)} total pairs, using first {len(limited)} for pairs task")
        return limited
    
    def create_binary_target(self, df):
        all_pairs = self.get_all_pairs(df)
        lambda_min, lambda_max = self.wavelength_range
        
        has_valid_peak = np.zeros(len(df), dtype=int)
        
        for f_col, lambda_col in all_pairs:
            f_values = df[f_col].values
            lambda_values = df[lambda_col].values
            
            strong_enough = f_values >= self.min_strength
            in_range = (lambda_values >= lambda_min) & (lambda_values <= lambda_max)
            valid_peak = strong_enough & in_range
            
            has_valid_peak = has_valid_peak | valid_peak.astype(int)
        
        df_copy = df.copy()
        df_copy[self.target_name] = has_valid_peak
        
        positive_count = has_valid_peak.sum()
        print(f"\nBinary target '{self.target_name}' created:")
        print(f"  Checked {len(all_pairs)} oscillation pairs")
        print(f"  Conditions: f >= {self.min_strength} AND {lambda_min} <= lambda <= {lambda_max}")
        print(f"  Positive samples (has valid peak): {positive_count} ({100*positive_count/len(df):.1f}%)")
        print(f"  Negative samples (no valid peak): {len(df) - positive_count} ({100*(len(df)-positive_count)/len(df):.1f}%)")
        
        return df_copy, self.target_name
    
    def should_run_pairs(self):
        if self.task == "custom":
            return self.custom_tasks.get('pairs', False)
        return self.task in ["pairs", "binary-pairs", "all"]
    
    def should_run_binary(self):
        if self.task == "custom":
            return self.custom_tasks.get('binary', False)
        return self.task in ["binary", "all"]
    
    def should_run_binary_pairs(self):
        if self.task == "custom":
            return self.custom_tasks.get('binary_pairs', False)
        return self.task in ["binary-pairs", "all"]
    
    def should_run_multilabel(self):
        if self.task == "custom":
            return self.custom_tasks.get('multilabel', False)
        return self.task in ["multilabel", "all"]