import pandas as pd
import numpy as np


class ValueCapper:
    def __init__(self, config):
        caps_config = config.data.get('value_caps', {})
        self.lambda_max = caps_config.get('lambda_max')
        self.f_max = caps_config.get('f_max')
        self.task = getattr(config, 'task', 'pairs')
        self.num_pairs = config.targets.get('num_pairs', 1)
        self.applied_caps = []
        
    def cap_values(self, df, target_pairs):
        """
        Apply value capping/filtering based on task type.
        
        For binary:
        - Cap f values > f_max to f_max
        - Remove examples where lambda > lambda_max
        
        For pairs:
        - Cap f values > f_max to f_max  
        - If lambda > lambda_max, skip that pair and take the next one
        - Always maintain num_pairs outputs by shifting pairs
        """
        if self.task in ['binary', 'all'] or (self.task == 'custom' and self._should_run_binary()):
            return self._cap_binary(df, target_pairs)
        else:
            return self._cap_pairs(df, target_pairs)
    
    def _should_run_binary(self):
        # Check if binary task is enabled in custom tasks
        # This is a simplified check - adjust based on your config structure
        return True  # Default to binary behavior for safety
    
    def _cap_binary(self, df, target_pairs):
        """Cap values for binary task: cap f, remove rows with lambda > lambda_max"""
        df_capped = df.copy()
        rows_to_remove = set()
        
        for f_col, lambda_col in target_pairs:
            # Cap f values at f_max
            if self.f_max is not None and f_col in df_capped.columns:
                count_before = (df_capped[f_col] > self.f_max).sum()
                df_capped[f_col] = df_capped[f_col].clip(upper=self.f_max)
                if count_before > 0:
                    self.applied_caps.append(f"{f_col}: capped {count_before} values at {self.f_max}")
            
            # Mark rows where lambda > lambda_max for removal
            if self.lambda_max is not None and lambda_col in df_capped.columns:
                invalid_mask = df_capped[lambda_col] > self.lambda_max
                invalid_indices = df_capped[invalid_mask].index.tolist()
                rows_to_remove.update(invalid_indices)
                if len(invalid_indices) > 0:
                    self.applied_caps.append(f"{lambda_col}: marked {len(invalid_indices)} rows for removal (lambda > {self.lambda_max})")
        
        # Remove rows with invalid lambda values
        if rows_to_remove:
            df_capped = df_capped.drop(index=list(rows_to_remove))
            print(f"\\nBinary value capping: removed {len(rows_to_remove)} rows with lambda > {self.lambda_max}")
        
        self._print_summary()
        return df_capped
    
    def _cap_pairs(self, df, target_pairs):
        """
        Cap values for pairs task: cap f, shift pairs if lambda > lambda_max.
        Always maintains num_pairs outputs per molecule.
        """
        df_capped = df.copy()
        all_pairs = target_pairs.copy()  # All available pairs
        
        # Get all possible pairs from dataframe (not just limited ones)
        f_cols_all = [col for col in df.columns if col.startswith('f_') and col.endswith('_gasphase')]
        lambda_cols_all = [col for col in df.columns if col.startswith('lambda_') and col.endswith('_gasphase')]
        
        all_available_pairs = []
        max_available = min(len(f_cols_all), len(lambda_cols_all))
        for i in range(1, max_available + 1):
            f_col = f'f_{i}_gasphase'
            lambda_col = f'lambda_{i}_gasphase'
            if f_col in df.columns and lambda_col in df.columns:
                all_available_pairs.append((f_col, lambda_col))
        
        print(f"\\nPairs capping: Found {len(all_available_pairs)} total available pairs")
        print(f"Need to maintain {self.num_pairs} output pairs per molecule")
        
        # For each row, determine which pairs to use
        new_pair_data = {}  # Will store new column values
        
        for idx in df_capped.index:
            valid_pairs_for_row = []
            
            # Find all valid pairs for this row (lambda <= lambda_max)
            for f_col, lambda_col in all_available_pairs:
                lambda_val = df_capped.loc[idx, lambda_col]
                
                # Check if lambda is valid
                if self.lambda_max is not None and pd.notna(lambda_val) and lambda_val <= self.lambda_max:
                    valid_pairs_for_row.append((f_col, lambda_col))
                elif self.lambda_max is None:
                    valid_pairs_for_row.append((f_col, lambda_col))
            
            # Take first num_pairs valid pairs
            selected_pairs = valid_pairs_for_row[:self.num_pairs]
            
            # Fill in the pair data for this row
            for i, (f_col, lambda_col) in enumerate(selected_pairs, 1):
                f_val = df_capped.loc[idx, f_col]
                lambda_val = df_capped.loc[idx, lambda_col]
                
                # Cap f value if needed
                if self.f_max is not None and pd.notna(f_val) and f_val > self.f_max:
                    f_val = self.f_max
                
                # Store in new columns
                new_f_col = f'f_{i}_gasphase_capped'
                new_lambda_col = f'lambda_{i}_gasphase_capped'
                
                if new_f_col not in new_pair_data:
                    new_pair_data[new_f_col] = {}
                    new_pair_data[new_lambda_col] = {}
                
                new_pair_data[new_f_col][idx] = f_val
                new_pair_data[new_lambda_col][idx] = lambda_val
            
            # Fill remaining pairs with NaN if not enough valid pairs
            for i in range(len(selected_pairs) + 1, self.num_pairs + 1):
                new_f_col = f'f_{i}_gasphase_capped'
                new_lambda_col = f'lambda_{i}_gasphase_capped'
                
                if new_f_col not in new_pair_data:
                    new_pair_data[new_f_col] = {}
                    new_pair_data[new_lambda_col] = {}
                
                new_pair_data[new_f_col][idx] = np.nan
                new_pair_data[new_lambda_col][idx] = np.nan
        
        # Add new columns to dataframe
        for col_name, values in new_pair_data.items():
            df_capped[col_name] = pd.Series(values)
        
        # Replace original columns with capped versions
        for i in range(1, self.num_pairs + 1):
            orig_f = f'f_{i}_gasphase'
            orig_lambda = f'lambda_{i}_gasphase'
            capped_f = f'f_{i}_gasphase_capped'
            capped_lambda = f'lambda_{i}_gasphase_capped'
            
            if capped_f in df_capped.columns:
                df_capped[orig_f] = df_capped[capped_f]
                df_capped = df_capped.drop(columns=[capped_f])
            if capped_lambda in df_capped.columns:
                df_capped[orig_lambda] = df_capped[capped_lambda]
                df_capped = df_capped.drop(columns=[capped_lambda])
        
        # Report statistics
        valid_counts = []
        for idx in df_capped.index:
            valid_count = sum(1 for i in range(1, self.num_pairs + 1) 
                            if pd.notna(df_capped.loc[idx, f'f_{i}_gasphase']))
            valid_counts.append(valid_count)
        
        print(f"  Average valid pairs per molecule: {np.mean(valid_counts):.2f}")
        print(f"  Molecules with all {self.num_pairs} pairs: {sum(1 for c in valid_counts if c == self.num_pairs)}/{len(df_capped)}")
        
        self._print_summary()
        return df_capped
    
    def _print_summary(self):
        if self.applied_caps:
            print(f"\\nValue capping applied:")
            for cap in self.applied_caps[:10]:  # Show first 10
                print(f"  {cap}")
            if len(self.applied_caps) > 10:
                print(f"  ... and {len(self.applied_caps) - 10} more")
        else:
            print(f"\\nNo value capping applied")
    
    def is_enabled(self):
        return self.lambda_max is not None or self.f_max is not None