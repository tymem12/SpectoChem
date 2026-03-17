import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.model_factory import get_model, get_model_params, get_cpu_fallback_model, check_gpu_available
from models.tuning_factory import (get_tuning_config, get_param_distributions, 
                                   setup_tuning_search, get_tuning_model_for_gpu)


class XGBoostClassifier:
    def __init__(self, config):
        self.output_dir = config.output['models_dir']
        self.target_name = config.binary.get('target_name', 'has_uvvis_peak')
        self.config = config
        self.model_type = getattr(config, 'model_type', 'xgboost')
        self.random_state = getattr(config, 'random_state', 42)
        self.results = {}
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self, data_reader, descriptors):
        X_train, X_test, X_val = data_reader.get_feature_splits(descriptors)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"\\nFeature shape: {X_train_scaled.shape}")
        print(f"Model type: {self.model_type}")
        print(f"Random state: {self.random_state}")
        
        X_train_val = np.vstack([X_train_scaled, X_val_scaled])
        
        y_train = data_reader.train_df[self.target_name].values
        y_test = data_reader.test_df[self.target_name].values
        y_val = data_reader.val_df[self.target_name].values
        
        y_train_val = np.concatenate([y_train, y_val])
        
        print(f"\\n{'='*70}")
        print(f"Training Binary Classification Model: {self.target_name}")
        print(f"{'='*70}")
        print(f"Training samples: {len(y_train_val)}")
        print(f"Positive class: {y_train_val.sum()} ({100*y_train_val.sum()/len(y_train_val):.1f}%)")
        
        ModelClass, is_cuml = get_model(self.config, task_type='classification')
        params = get_model_params(self.config, task_type='classification')
        
        tuning_config = get_tuning_config(self.config, 'binary')
        if tuning_config:
            print(f"  Running hyperparameter tuning with RandomizedSearchCV...")
            best_params = self._tune_hyperparameters(X_train_val, y_train_val, ModelClass, is_cuml)
            params = {**params, **best_params}
        
        model = ModelClass(**params)
        model.fit(X_train_val, y_train_val)
        
        y_pred = model.predict(X_test_scaled)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.decision_function(X_test_scaled)
            y_proba = 1 / (1 + np.exp(-y_proba))
        
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        self._print_metrics(metrics)
        
        self.results = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'scaler': scaler,
            'best_params': params
        }
        
        self._save_model(model)
        
        return self.results
    
    def _tune_hyperparameters(self, X, y, ModelClass, is_cuml):
        """Run hyperparameter tuning using GPU if available"""
        tuning_config = get_tuning_config(self.config, 'binary')
        param_dist = get_param_distributions(self.config, task_type='classification')
        
        use_gpu = getattr(self.config, 'gpu', {}).get('use_if_available', True)
        gpu_available = check_gpu_available() if use_gpu else False
        
        if gpu_available and self.model_type in ['random_forest', 'svm', 'logistic_regression']:
            TuningModelClass, tuning_params, _ = get_tuning_model_for_gpu(self.config, 'classification')
            print(f"  Using GPU-accelerated tuning for {self.model_type}")
        else:
            if is_cuml or self.model_type == 'xgboost':
                TuningModelClass, tuning_params = get_cpu_fallback_model(self.config, 'classification')
            else:
                TuningModelClass = ModelClass
                tuning_params = get_model_params(self.config, 'classification')
            print(f"  Using CPU tuning for {self.model_type}")
        
        search = setup_tuning_search(
            TuningModelClass, param_dist, tuning_config, tuning_params,
            random_state=self.random_state,
            use_gpu=(gpu_available and self.model_type in ['random_forest', 'svm', 'logistic_regression']),
            model_type=self.model_type,
            task_type='classification'
        )
        search.fit(X, y)
        
        print(f"  Best score: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
        
        return search.best_params_
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
    
    def _print_metrics(self, metrics):
        print(f"\\nClassification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    def _save_model(self, model):
        joblib.dump(model, os.path.join(self.output_dir, f'classifier_{self.target_name}.pkl'))