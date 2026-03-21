import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.model_factory import get_model, get_model_params, get_cpu_fallback_model, check_gpu_available
from models.tuning_factory import (get_tuning_config, get_param_distributions, 
                                   setup_tuning_search, is_tuning_enabled, get_tuning_model_for_gpu)


class MultilabelTrainer:
    def __init__(self, config):
        self.output_dir = config.output['models_dir']
        self.config = config
        self.model_type = getattr(config, 'model_type', 'xgboost')
        self.random_state = getattr(config, 'random_state', 42)
        self.results = {
            'models': {},
            'metrics': {},
            'predictions': {},
            'probabilities': {},
        }
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self, data_reader, descriptors):
        X_train, X_test, X_val = data_reader.get_feature_splits(descriptors)
        bucket_columns = data_reader.get_bucket_columns()
        
        print(f"\\nFeature shape: {X_train.shape}")
        print(f"Model type: {self.model_type}")
        print(f"Random state: {self.random_state}")
        print(f"Training {len(bucket_columns)} bucket classifiers...")
        
        X_train_val = np.vstack([X_train, X_val])
        
        for idx, bucket_col in enumerate(bucket_columns):
            self._train_bucket(idx, bucket_col, data_reader, 
                              X_train_val, X_test, X_train, X_val, len(bucket_columns))
        
        return self.results
    
    def _train_bucket(self, idx, bucket_col, data_reader, 
                     X_train_val, X_test, X_train_raw, X_val_raw):
        print(f"\\n{'='*70}")
        print(f"Training Bucket {idx+1}/{data_reader.get_bucket_columns().__len__()}: {bucket_col}")
        print(f"{'='*70}")
        
        y_train = data_reader.train_df[bucket_col].values
        y_test = data_reader.test_df[bucket_col].values
        y_val = data_reader.val_df[bucket_col].values
        
        y_train_val = np.concatenate([y_train, y_val])
        
        print(f"Training samples: {len(y_train_val)}")
        print(f"Positive class: {y_train_val.sum()} ({100*y_train_val.sum()/len(y_train_val):.1f}%)")
        
        ModelClass, is_cuml = get_model(self.config, task_type='classification')
        base_params = get_model_params(self.config, task_type='classification')
        
        if is_tuning_enabled(self.config, 'multilabel'):
            print(f"  Running hyperparameter tuning with Pipeline...")
            best_params = self._tune_hyperparameters(X_train_raw, y_train, ModelClass, base_params.copy())
            final_params = {**base_params, **best_params}
        else:
            final_params = base_params
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ModelClass(**final_params))
        ])
        
        pipeline.fit(X_train_val, y_train_val)
        
        y_pred = pipeline.predict(X_test)
        
        if hasattr(pipeline.named_steps['model'], 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            decision = pipeline.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-decision))
        
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        self._print_metrics(metrics)
        
        self._store_results(bucket_col, pipeline, metrics, y_pred, y_proba, final_params)
        self._save_model(pipeline, bucket_col)
    
    def _tune_hyperparameters(self, X, y, ModelClass, base_params):
        """Run hyperparameter tuning using Pipeline to prevent data leakage."""
        tuning_config = get_tuning_config(self.config, 'multilabel')
        param_dist = get_param_distributions(self.config, task_type='classification')
        
        pipeline_param_dist = {}
        for param_name, values in param_dist.items():
            pipeline_param_dist[f'model__{param_name}'] = values
        
        TuningModelClass, tuning_params = get_cpu_fallback_model(self.config, 'classification')
        
        for key in list(tuning_params.keys()):
            if f'model__{key}' in pipeline_param_dist:
                tuning_params.pop(key, None)
        
        print(f"  Using CPU tuning with Pipeline to prevent data leakage")
        
        tuning_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', TuningModelClass(**tuning_params))
        ])
        
        search = setup_tuning_search(
            tuning_pipeline,
            pipeline_param_dist,
            tuning_config, 
            {},
            random_state=self.random_state,
            use_gpu=False,
            model_type=self.model_type,
            task_type='classification'
        )
        search.fit(X, y)
        
        print(f"  Best score: {search.best_score_:.4f}")
        best_params_raw = search.best_params_
        best_params = {k.replace('model__', ''): v for k, v in best_params_raw.items()}
        print(f"  Best params: {best_params}")
        
        return best_params
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
    
    def _print_metrics(self, metrics):
        print(f"\\nBucket Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    def _store_results(self, bucket_col, pipeline, metrics, predictions, probabilities, params):
        self.results['models'][bucket_col] = pipeline
        self.results['metrics'][bucket_col] = metrics
        self.results['predictions'][bucket_col] = predictions
        self.results['probabilities'][bucket_col] = probabilities
        self.results['best_params'] = params
    
    def _save_model(self, pipeline, bucket_col):
        joblib.dump(pipeline, os.path.join(self.output_dir, f'multilabel_{bucket_col}.pkl'))