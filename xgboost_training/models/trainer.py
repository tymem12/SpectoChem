import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
            'predictions': {},
            'best_params': {}
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, data_reader, descriptors, target_pairs):
        X_train, X_test, X_val = data_reader.get_feature_splits(descriptors)

        print(f"\\nFeature shape: {X_train.shape}")
        print(f"Model type: {self.model_type}")
        print(f"Random state: {self.random_state}")

        for idx, (f_col, lambda_col) in enumerate(target_pairs):
            self._train_pair(idx, f_col, lambda_col, data_reader, 
                           X_train, X_test, X_val, len(target_pairs))

        return self.results

    def _train_pair(self, idx, f_col, lambda_col, data_reader, 
                    X_train, X_test, X_val, total_pairs):
        print(f"\\n{'='*70}")
        print(f"Training Model {2*idx+1} and {2*idx+2}/{2*total_pairs}")
        print(f"{f_col} & {lambda_col}")
        print(f"{'='*70}")

        self._train_target(f_col, data_reader.train_df, data_reader.test_df, 
                          data_reader.val_df, X_train, X_test, X_val)

        self._train_target(lambda_col, data_reader.train_df, data_reader.test_df, 
                          data_reader.val_df, X_train, X_test, X_val)

    def _train_target(self, target_col, train_df, test_df, val_df,
                      X_train, X_test, X_val):
        print(f"\\nTraining {target_col}...")

        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        y_val = val_df[target_col].values

        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        ModelClass, is_cuml = get_model(self.config, task_type='regression')
        base_params = get_model_params(self.config, task_type='regression')

        if is_tuning_enabled(self.config, 'pairs'):
            print(f"  Running hyperparameter tuning with proper CV...")
            best_params = self._tune_hyperparameters(
                X_train, y_train,  
                X_val, y_val,       
                ModelClass, base_params.copy()
            )
            final_params = {**base_params, **best_params}
        else:
            final_params = base_params

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ModelClass(**final_params))
        ])

        pipeline.fit(X_train_val, y_train_val)

        y_pred = pipeline.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)
        self._print_metrics(metrics)

        self._store_results(target_col, pipeline, metrics, y_pred, final_params)
        self._save_model(pipeline, target_col)

    def _tune_hyperparameters(self, X_train, y_train, X_val, y_val, ModelClass, base_params):
        """
        Run hyperparameter tuning using Pipeline to prevent data leakage.
        
        The Pipeline ensures that StandardScaler is fit ONLY on training folds
        during cross-validation, preventing any information leak from validation/test.
        """
        tuning_config = get_tuning_config(self.config, 'pairs')
        param_dist = get_param_distributions(self.config, task_type='regression')

        pipeline_param_dist = {}
        for param_name, values in param_dist.items():
            pipeline_param_dist[f'model__{param_name}'] = values

        use_gpu = getattr(self.config, 'gpu', {}).get('use_if_available', True)
        gpu_available = check_gpu_available() if use_gpu else False

        TuningModelClass, tuning_params = get_cpu_fallback_model(self.config, 'regression')
        
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
            task_type='regression'
        )

        search.fit(X_train, y_train)

        print(f"  Best CV score: {search.best_score_:.4f}")
        best_params_raw = search.best_params_
        best_params = {k.replace('model__', ''): v for k, v in best_params_raw.items()}
        print(f"  Best params: {best_params}")

        return best_params

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

    def _store_results(self, target_col, pipeline, metrics, predictions, params):
        self.results['models'][target_col] = pipeline
        self.results['metrics'][target_col] = metrics
        self.results['predictions'][target_col] = predictions
        self.results['best_params'][target_col] = params

    def _save_model(self, pipeline, target_col):
        joblib.dump(pipeline, os.path.join(self.output_dir, f'model_{target_col}.pkl'))