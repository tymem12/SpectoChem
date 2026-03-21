import os
import json
import numpy as np
import pandas as pd


class ResultsSaver:
    def __init__(self, config):
        self.output_dir = config.output['results_dir']
        self.task = config.task
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_pairs(self, results, test_df, target_pairs):
        predictions_data = {'sample_idx': test_df.index.values}
        
        for col, preds in results['predictions'].items():
            predictions_data[f'{col}_pred'] = preds
        
        for col in results['predictions'].keys():
            if col in test_df.columns:
                predictions_data[f'{col}_actual'] = test_df[col].values
        
        predictions_df = pd.DataFrame(predictions_data)
        pred_path = os.path.join(self.output_dir, 'test_predictions_pairs.csv')
        predictions_df.to_csv(pred_path, index=False)
        print(f"\nPairs predictions saved to {pred_path}")
        
        metrics_df = pd.DataFrame(results['metrics']).T
        metrics_path = os.path.join(self.output_dir, 'model_metrics_pairs.csv')
        metrics_df.to_csv(metrics_path)
        print(f"Pairs metrics saved to {metrics_path}")
        
        self._save_hyperparameters(results, 'best_hyperparameters_pairs.json')
        self._print_pairs_summary(metrics_df)
        
        return predictions_df, metrics_df
    
    def save_binary(self, results, test_df, target_name):
        predictions_data = {
            'sample_idx': test_df.index.values,
            f'{target_name}_pred': results['predictions'],
            f'{target_name}_proba': results['probabilities'],
            f'{target_name}_actual': test_df[target_name].values
        }
        
        predictions_df = pd.DataFrame(predictions_data)
        pred_path = os.path.join(self.output_dir, 'test_predictions_binary.csv')
        predictions_df.to_csv(pred_path, index=False)
        print(f"\nBinary predictions saved to {pred_path}")
        
        metrics_df = pd.DataFrame([results['metrics']], index=[target_name])
        metrics_path = os.path.join(self.output_dir, 'model_metrics_binary.csv')
        metrics_df.to_csv(metrics_path)
        print(f"Binary metrics saved to {metrics_path}")
        
        print(f"\n{'='*70}")
        print("BINARY CLASSIFICATION METRICS SUMMARY")
        print(f"{'='*70}")
        print(metrics_df)
        
        return predictions_df, metrics_df
    
    def save_multilabel(self, results, test_df, bucket_columns):
        predictions_data = {'sample_idx': test_df.index.values}
        
        for bucket_col in bucket_columns:
            predictions_data[f'{bucket_col}_pred'] = results['predictions'][bucket_col]
            predictions_data[f'{bucket_col}_proba'] = results['probabilities'][bucket_col]
            predictions_data[f'{bucket_col}_actual'] = test_df[bucket_col].values
        
        predictions_df = pd.DataFrame(predictions_data)
        pred_path = os.path.join(self.output_dir, 'test_predictions_multilabel.csv')
        predictions_df.to_csv(pred_path, index=False)
        print(f"\nMultilabel predictions saved to {pred_path}")
        
        metrics_df = pd.DataFrame(results['metrics']).T
        metrics_path = os.path.join(self.output_dir, 'model_metrics_multilabel.csv')
        metrics_df.to_csv(metrics_path)
        print(f"Multilabel metrics saved to {metrics_path}")
        
        print(f"\n{'='*70}")
        print("MULTILABEL CLASSIFICATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total buckets: {len(bucket_columns)}")
        print(f"Average Accuracy: {metrics_df['accuracy'].mean():.4f}")
        print(f"Average Precision: {metrics_df['precision'].mean():.4f}")
        print(f"Average Recall: {metrics_df['recall'].mean():.4f}")
        print(f"Average F1: {metrics_df['f1'].mean():.4f}")
        print(f"Average ROC-AUC: {metrics_df['roc_auc'].mean():.4f}")
        
        best_bucket = metrics_df['f1'].idxmax()
        worst_bucket = metrics_df['f1'].idxmin()
        print(f"\nBest bucket (by F1): {best_bucket} (F1={metrics_df.loc[best_bucket, 'f1']:.4f})")
        print(f"Worst bucket (by F1): {worst_bucket} (F1={metrics_df.loc[worst_bucket, 'f1']:.4f})")
        
        return predictions_df, metrics_df
    
    def _save_hyperparameters(self, results, filename):
        serializable_params = self._make_serializable(results.get('best_params', {}))
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        print(f"Hyperparameters saved to {path}")
    
    def _make_serializable(self, params_dict):
        result = {}
        for model_name, params in params_dict.items():
            result[model_name] = {}
            for k, v in params.items():
                if isinstance(v, (np.integer, int)):
                    result[model_name][k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    result[model_name][k] = float(v)
                else:
                    result[model_name][k] = v
        return result
    
    def _print_pairs_summary(self, metrics_df):
        print(f"\n{'='*70}")
        print("PAIRS REGRESSION METRICS SUMMARY")
        print(f"{'='*70}")
        print(metrics_df)
        
        print(f"\n{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Average MAE: {metrics_df['MAE'].mean():.6f}")
        print(f"Average MSE: {metrics_df['MSE'].mean():.6f}")
        print(f"Average R2:  {metrics_df['R2'].mean():.6f}")
        
    def save_binary_pairs(self, results, test_df, binary_f_columns):
    
        predictions_data = {'sample_idx': test_df.index.values}
        
        for f_col in binary_f_columns:
            predictions_data[f'{f_col}_pred'] = results['predictions'][f_col]
            predictions_data[f'{f_col}_proba'] = results['probabilities'][f_col]
    
            orig_f_col = f_col.replace('f_binary_', '')
            predictions_data[f'{f_col}_actual_binary'] = test_df[f_col].values
            predictions_data[f'{orig_f_col}_actual'] = test_df[orig_f_col].values
        
        predictions_df = pd.DataFrame(predictions_data)
        pred_path = os.path.join(self.output_dir, 'test_predictions_binary_pairs.csv')
        predictions_df.to_csv(pred_path, index=False)
        print(f"\nBinary-pairs predictions saved to {pred_path}")
        
    
        metrics_df = pd.DataFrame(results['metrics']).T
        metrics_path = os.path.join(self.output_dir, 'model_metrics_binary_pairs.csv')
        metrics_df.to_csv(metrics_path)
        print(f"Binary-pairs metrics saved to {metrics_path}")
        
    
        print(f"\n{'='*70}")
        print("BINARY-PAIRS CLASSIFICATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total F classifiers: {len(binary_f_columns)}")
        print(f"Average Accuracy: {metrics_df['accuracy'].mean():.4f}")
        print(f"Average Precision: {metrics_df['precision'].mean():.4f}")
        print(f"Average Recall: {metrics_df['recall'].mean():.4f}")
        print(f"Average F1: {metrics_df['f1'].mean():.4f}")
        print(f"Average ROC-AUC: {metrics_df['roc_auc'].mean():.4f}")
        
        best_f = metrics_df['f1'].idxmax()
        worst_f = metrics_df['f1'].idxmin()
        print(f"\nBest classifier (by F1): {best_f} (F1={metrics_df.loc[best_f, 'f1']:.4f})")
        print(f"Worst classifier (by F1): {worst_f} (F1={metrics_df.loc[worst_f, 'f1']:.4f})")
        
        return predictions_df, metrics_df