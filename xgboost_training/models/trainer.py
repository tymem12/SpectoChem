import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.model_factory import get_model, get_model_params, get_cpu_fallback_model, check_gpu_available
from models.tuning_factory import (get_tuning_config, get_param_distributions, 
                                   setup_tuning_search, is_tuning_enabled, get_tuning_model_for_gpu)


class XGBoostTrainer:
    def __init__(self, config):
        self.output_dir = config.output['models_dir']
        self.config = config
        self.model_type = getattr(config, 'model_type', 'xgboost')
        self.random_state = getattr(config, 'random_state', 42)
        self.results = {
            'models': {},
            'metrics': {},
            'scalers': {},
            'predictions': {},
            'best_params': {}
        }
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self, data_reader, descriptors, target_pairs):
        X_train, X_test, X_val = data_reader.get_feature_splits(descriptors)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"\\nFeature shape: {X_train_scaled.shape}")
        print(f"Model type: {self.model_type}")
        print(f"Random state: {self.random_state}")
        
        X_train_val = np.vstack([X_train_scaled, X_val_scaled])
        
        for idx, (f_col, lambda_col) in enumerate(target_pairs):
            self._train_pair(idx, f_col, lambda_col, data_reader, 
                           X_train_val, X_test_scaled, X_val_scaled, scaler, len(target_pairs))
        
        return self.results
    
    def _train_pair(self, idx, f_col, lambda_col, data_reader, 
                    X_train_val, X_test_scaled, X_val_scaled, feature_scaler, total_pairs):
        print(f"\\n{'='*70}")
        print(f"Training Model {2*idx+1} and {2*idx+2}/{2*total_pairs}")
        print(f"{f_col} & {lambda_col}")
        print(f"{'='*70}")
        
        self._train_target(f_col, data_reader.train_df, data_reader.test_df, 
                          data_reader.val_df, X_train_val, X_test_scaled, 
                          X_val_scaled, feature_scaler)
        
        self._train_target(lambda_col, data_reader.train_df, data_reader.test_df, 
                          data_reader.val_df, X_train_val, X_test_scaled, 
                          X_val_scaled, feature_scaler)
    
    def _train_target(self, target_col, train_df, test_df, val_df,
                      X_train_val, X_test_scaled, X_val_scaled, feature_scaler):
        print(f"\\nTraining {target_col}...")
        
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        y_val = val_df[target_col].values
        
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_train_val = np.concatenate([y_train_scaled, y_val_scaled])
        
        ModelClass, is_cuml = get_model(self.config, task_type='regression')
        params = get_model_params(self.config, task_type='regression')
        
        if is_tuning_enabled(self.config, 'pairs'):
            print(f"  Running hyperparameter tuning...")
            best_params = self._tune_hyperparameters(X_train_val, y_train_val, ModelClass, is_cuml)
            params = {**params, **best_params}
        
        model = ModelClass(**params)
        model.fit(X_train_val, y_train_val)
        
        y_pred = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        
        metrics = self._calculate_metrics(y_test, y_pred)
        self._print_metrics(metrics)
        
        self._store_results(target_col, model, metrics, y_pred, 
                           feature_scaler, target_scaler, params)
        self._save_model(model, target_col)
    
    def _tune_hyperparameters(self, X, y, ModelClass, is_cuml):
        """Run hyperparameter tuning using GPU if available"""
        tuning_config = get_tuning_config(self.config, 'pairs')
        param_dist = get_param_distributions(self.config, task_type='regression')
        
        use_gpu = getattr(self.config, 'gpu', {}).get('use_if_available', True)
        gpu_available = check_gpu_available() if use_gpu else False
        
        if gpu_available and self.model_type in ['random_forest', 'svm']:
            TuningModelClass, tuning_params, _ = get_tuning_model_for_gpu(self.config, 'regression')
            print(f"  Using GPU-accelerated tuning for {self.model_type}")
        else:
            TuningModelClass, tuning_params = get_cpu_fallback_model(self.config, 'regression')
            print(f"  Using CPU tuning for {self.model_type}")
        
        search = setup_tuning_search(
            TuningModelClass, param_dist, tuning_config, tuning_params,
            random_state=self.random_state,
            use_gpu=(gpu_available and self.model_type in ['random_forest', 'svm']),
            model_type=self.model_type,
            task_type='regression'
        )
        search.fit(X, y)
        
        print(f"  Best score: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
        
        return search.best_params_
    
    def _calculate_metrics(self, y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    
    def _print_metrics(self, metrics):
        print(f"MAE: {metrics['MAE']:.6f}")
        print(f"MSE: {metrics['MSE']:.6f}")
        print(f"R2 : {metrics['R2']:.6f}")
    
    def _store_results(self, target_col, model, metrics, predictions,
                      feature_scaler, target_scaler, params):
        self.results['models'][target_col] = model
        self.results['metrics'][target_col] = metrics
        self.results['predictions'][target_col] = predictions
        self.results['best_params'][target_col] = params
        self.results['scalers'][f'{target_col}_feature'] = feature_scaler
        self.results['scalers'][f'{target_col}_target'] = target_scaler
    
    def _save_model(self, model, target_col):
        joblib.dump(model, os.path.join(self.output_dir, f'model_{target_col}.pkl'))