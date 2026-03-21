import numpy as np
import pandas as pd


class MultilabelGenerator:
    def __init__(self, config):
        self.config = config.multilabel
        self.bucket_size = self.config['bucket_size_nm']
        self.min_wavelength = self.config['min_wavelength']
        self.max_wavelength = self.config['max_wavelength']
        self.min_strength = self.config['min_oscillation_strength']
        self.target_prefix = self.config['target_prefix']
        
        self.num_buckets = int((self.max_wavelength - self.min_wavelength) / self.bucket_size)
        self.bucket_edges = np.arange(self.min_wavelength, self.max_wavelength + self.bucket_size, self.bucket_size)
        
    def get_all_pairs(self, df):
        """Get all f and lambda pairs from dataframe"""
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
    
    def create_bucket_targets(self, df):
        """Create binary bucket targets for absorption spectrum reconstruction"""
        all_pairs = self.get_all_pairs(df)
        
        print(f"\nCreating multilabel bucket targets:")
        print(f"  Wavelength range: {self.min_wavelength}-{self.max_wavelength} nm")
        print(f"  Bucket size: {self.bucket_size} nm")
        print(f"  Number of buckets: {self.num_buckets}")
        print(f"  Min oscillation strength: {self.min_strength}")
        print(f"  Checking {len(all_pairs)} oscillation pairs")
        
        bucket_targets = np.zeros((len(df), self.num_buckets), dtype=int)
        
        for idx in range(len(df)):
            for f_col, lambda_col in all_pairs:
                f_value = df[f_col].iloc[idx]
                lambda_value = df[lambda_col].iloc[idx]
                
                if pd.isna(f_value) or f_value < self.min_strength:
                    continue
                
                if not (self.min_wavelength <= lambda_value < self.max_wavelength):
                    continue
                
                bucket_idx = int((lambda_value - self.min_wavelength) / self.bucket_size)
                
                if 0 <= bucket_idx < self.num_buckets:
                    bucket_targets[idx, bucket_idx] = 1
        
        bucket_columns = [f"{self.target_prefix}_{int(self.bucket_edges[i])}_{int(self.bucket_edges[i+1])}" 
                         for i in range(self.num_buckets)]
        
        df_copy = df.copy()
        for i, col_name in enumerate(bucket_columns):
            df_copy[col_name] = bucket_targets[:, i]
        
        bucket_counts = bucket_targets.sum(axis=0)
        molecules_with_absorption = (bucket_targets.sum(axis=1) > 0).sum()
        
        print(f"\nBucket statistics:")
        print(f"  Molecules with any absorption: {molecules_with_absorption}/{len(df)} ({100*molecules_with_absorption/len(df):.1f}%)")
        print(f"  Active buckets: {(bucket_counts > 0).sum()}/{self.num_buckets}")
        print(f"  Total absorption markers: {bucket_targets.sum()}")
        
        
        print(f"\nAll buckets:")
        for i in range(self.num_buckets):
            count = bucket_counts[i]
            pct = 100 * count / len(df)
            status = "ACTIVE" if count > 0 else "empty"
            print(f"  Bucket {i} ({int(self.bucket_edges[i])}-{int(self.bucket_edges[i+1])}nm): {count} molecules ({pct:.1f}%) [{status}]")
        
        return df_copy, bucket_columns
    
    def should_run_multilabel(self):
        return True