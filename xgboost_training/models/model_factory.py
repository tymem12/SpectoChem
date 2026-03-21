import numpy as np
import warnings
from typing import Tuple, Any, Dict


def check_gpu_available():
    """Check if GPU is available for cuML"""
    try:
        import cupy as cp
        gpu_count = cp.cuda.runtime.getDeviceCount()
        return gpu_count > 0
    except Exception:
        return False


def get_model(config, task_type='regression'):
    """
    Factory function to get appropriate model based on config.

    Args:
        config: Config object
        task_type: 'regression' or 'classification'

    Returns:
        (ModelClass, is_cuml): Tuple of model class and whether it's cuML
    """
    model_type = getattr(config, 'model_type', 'xgboost').lower()
    use_gpu = getattr(config, 'gpu', {}).get('use_if_available', True)
    gpu_available = check_gpu_available() if use_gpu else False

    if model_type == 'xgboost':
        return _get_xgboost_model(task_type), False

    elif model_type == 'random_forest':
        return _get_random_forest_model(task_type, gpu_available)

    elif model_type == 'svm':
        return _get_svm_model(task_type, gpu_available)

    elif model_type == 'dummy':
        return _get_dummy_model(task_type), False

    elif model_type == 'logistic_regression':
        return _get_logistic_regression_model(task_type, gpu_available)

    elif model_type == 'mlp':
        return _get_mlp_model(task_type), False

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _get_xgboost_model(task_type):
    """Get XGBoost model"""
    if task_type == 'regression':
        from xgboost import XGBRegressor
        return XGBRegressor
    else:
        from xgboost import XGBClassifier
        return XGBClassifier


def _get_random_forest_model(task_type, gpu_available):
    """Get Random Forest model (cuML if GPU available, else sklearn)"""
    if gpu_available:
        try:
            from cuml.ensemble import RandomForestRegressor, RandomForestClassifier
            print(f"Using cuML Random Forest on GPU")
            if task_type == 'regression':
                return RandomForestRegressor, True
            else:
                return RandomForestClassifier, True
        except ImportError:
            warnings.warn("cuML not available, falling back to sklearn")
            gpu_available = False

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    print(f"Using sklearn Random Forest on CPU")
    if task_type == 'regression':
        return RandomForestRegressor, False
    else:
        return RandomForestClassifier, False


def _get_svm_model(task_type, gpu_available):
    """Get SVM model (cuML if GPU available, else sklearn)"""
    if gpu_available:
        try:
            from cuml.svm import SVR, SVC
            print(f"Using cuML SVM on GPU")
            if task_type == 'regression':
                return SVR, True
            else:
                return SVC, True
        except ImportError:
            warnings.warn("cuML not available, falling back to sklearn")
            gpu_available = False

    from sklearn.svm import SVR, SVC
    print(f"Using sklearn SVM on CPU")
    if task_type == 'regression':
        return SVR, False
    else:
        return SVC, False


def _get_dummy_model(task_type):
    """
    Get Dummy model.

    IMPORTANT: DummyRegressor and DummyClassifier have DIFFERENT strategies!
    - DummyRegressor (regression): 'mean', 'median', 'quantile', 'constant'
    - DummyClassifier (classification): 'stratified', 'most_frequent', 'uniform', 'constant'
    """
    if task_type == 'regression':
        from sklearn.dummy import DummyRegressor
        return DummyRegressor
    else:
        from sklearn.dummy import DummyClassifier
        return DummyClassifier


def _get_logistic_regression_model(task_type, gpu_available):
    """Get Logistic Regression model (cuML if GPU available, else sklearn)"""
    if task_type == 'regression':
        if gpu_available:
            try:
                from cuml.linear_model import LinearRegression
                print(f"Using cuML Linear Regression on GPU")
                return LinearRegression, True
            except ImportError:
                warnings.warn("cuML not available, falling back to sklearn")
                gpu_available = False
        from sklearn.linear_model import LinearRegression
        print(f"Using sklearn Linear Regression on CPU")
        return LinearRegression, False
    else:
        if gpu_available:
            try:
                from cuml.linear_model import LogisticRegression
                print(f"Using cuML Logistic Regression on GPU")
                return LogisticRegression, True
            except ImportError:
                warnings.warn("cuML not available, falling back to sklearn")
                gpu_available = False
        from sklearn.linear_model import LogisticRegression
        print(f"Using sklearn Logistic Regression on CPU")
        return LogisticRegression, False


def _get_mlp_model(task_type):
    """Get Simple MLP model (sklearn MLPClassifier/MLPRegressor)"""
    if task_type == 'regression':
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor
    else:
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier


def get_model_params(config, task_type='regression') -> Dict[str, Any]:
    """Get model parameters from config"""
    model_type = getattr(config, 'model_type', 'xgboost').lower()

    if model_type == 'xgboost':
        params = config.xgboost.copy()
        return params

    elif model_type == 'random_forest':
        params = config.random_forest.copy()
        is_cuml = check_gpu_available() and params.get('use_if_available', True)
        if is_cuml and 'n_jobs' in params:
            del params['n_jobs']
        return params

    elif model_type == 'svm':
        params = config.svm.copy()
        if task_type != 'regression':
            params['probability'] = True
        return params

    elif model_type == 'dummy':
        dummy_config = getattr(config, 'dummy', {})

        if task_type == 'regression':
            valid_strategies = ['mean', 'median', 'quantile', 'constant']
            strategy = dummy_config.get('strategy', 'mean')

            if strategy not in valid_strategies:
                print(f"WARNING: strategy '{strategy}' is not valid for regression (DummyRegressor).")
                print(f"         Valid strategies: {valid_strategies}")
                print(f"         Using default: 'mean'")
                strategy = 'mean'

            params = {'strategy': strategy}

            if strategy == 'constant':
                params['constant'] = dummy_config.get('constant', None)
            if strategy == 'quantile':
                params['quantile'] = dummy_config.get('quantile', 0.5)

            return params

        else:
            valid_strategies = ['stratified', 'most_frequent', 'uniform', 'constant', 'prior']
            strategy = dummy_config.get('strategy', 'stratified')

            if strategy not in valid_strategies:
                print(f"WARNING: strategy '{strategy}' is not valid for classification (DummyClassifier).")
                print(f"         Valid strategies: {valid_strategies}")
                print(f"         Using default: 'stratified'")
                strategy = 'stratified'

            params = {
                'strategy': strategy,
                'random_state': dummy_config.get('random_state', 42)
            }

            if strategy == 'constant':
                params['constant'] = dummy_config.get('constant', None)

            return params

    elif model_type == 'logistic_regression':
        lr_config = getattr(config, 'logistic_regression', {})

        use_gpu = getattr(config, 'gpu', {}).get('use_if_available', True)
        gpu_available = check_gpu_available() if use_gpu else False

        if gpu_available and task_type != 'regression':
            import cuml
            return {
                'C': lr_config.get('C', 1.0),
                'max_iter': lr_config.get('max_iter', 1000),
                'tol': lr_config.get('tol', 0.0001),
                'random_state': lr_config.get('random_state', 42)
            }

        if task_type == 'regression':
            return {
                'n_jobs': lr_config.get('n_jobs', -1),
                'positive': lr_config.get('positive', False)
            }
        else:
            return {
                'C': lr_config.get('C', 1.0),
                'max_iter': lr_config.get('max_iter', 1000),
                'random_state': lr_config.get('random_state', 42),
                'n_jobs': lr_config.get('n_jobs', -1),
            }

    elif model_type == 'mlp':
        mlp_config = getattr(config, 'mlp', {})
        return {
            'hidden_layer_sizes': mlp_config.get('hidden_layer_sizes', [128, 64]),
            'activation': mlp_config.get('activation', 'relu'),
            'solver': mlp_config.get('solver', 'adam'),
            'alpha': mlp_config.get('alpha', 0.0001),
            'batch_size': mlp_config.get('batch_size', 'auto'),
            'learning_rate': mlp_config.get('learning_rate', 'constant'),
            'learning_rate_init': mlp_config.get('learning_rate_init', 0.001),
            'max_iter': mlp_config.get('max_iter', 500),
            'shuffle': mlp_config.get('shuffle', True),
            'random_state': mlp_config.get('random_state', 42),
            'early_stopping': mlp_config.get('early_stopping', True),
            'validation_fraction': mlp_config.get('validation_fraction', 0.1),
            'n_iter_no_change': mlp_config.get('n_iter_no_change', 10),
            'verbose': mlp_config.get('verbose', False)
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_cpu_fallback_model(config, task_type='regression'):
    """Get CPU version of model for tuning (to avoid GPU memory issues)."""
    model_type = getattr(config, 'model_type', 'xgboost').lower()

    if model_type == 'xgboost':
        from xgboost import XGBRegressor, XGBClassifier
        params = config.xgboost.copy()
        params['tree_method'] = 'hist'
        params['device'] = 'cpu'
        if task_type == 'regression':
            return XGBRegressor, params
        else:
            return XGBClassifier, params

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        params = config.random_forest.copy()
        if task_type == 'regression':
            return RandomForestRegressor, params
        else:
            return RandomForestClassifier, params

    elif model_type == 'svm':
        from sklearn.svm import SVR, SVC
        params = config.svm.copy()
        if task_type != 'regression':
            params['probability'] = True
        if task_type == 'regression':
            return SVR, params
        else:
            return SVC, params

    elif model_type == 'dummy':
        """
        Dummy model CPU fallback - same strategy validation as main function.
        """
        if task_type == 'regression':
            from sklearn.dummy import DummyRegressor
            return DummyRegressor, {'strategy': 'mean'}
        else:
            from sklearn.dummy import DummyClassifier
            return DummyClassifier, {'strategy': 'stratified', 'random_state': 42}

    elif model_type == 'logistic_regression':
        if task_type == 'regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression, {
                'n_jobs': getattr(config, 'logistic_regression', {}).get('n_jobs', -1),
                'positive': getattr(config, 'logistic_regression', {}).get('positive', False)
            }
        else:
            from sklearn.linear_model import LogisticRegression
            params = getattr(config, 'logistic_regression', {})
            return LogisticRegression, {
                'C': params.get('C', 1.0),
                'max_iter': params.get('max_iter', 1000),
                'random_state': params.get('random_state', 42)
            }

    elif model_type == 'mlp':
        if task_type == 'regression':
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor, get_model_params(config, task_type)
        else:
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier, get_model_params(config, task_type)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")