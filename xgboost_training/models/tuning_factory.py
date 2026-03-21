import warnings
import numpy as np
import os

def check_gpu_available():
    """Check if GPU is available for cuML"""
    try:
        import cupy as cp
        gpu_count = cp.cuda.runtime.getDeviceCount()
        return gpu_count > 0
    except Exception:
        return False


def get_tuning_config(config, task_name=None):
    """
    Get tuning configuration for current model type and task.
    
    Args:
        config: Config object
        task_name: Optional task name for legacy fallback ('pairs', 'binary', etc.)
    
    Returns:
        tuning_config: Dictionary with tuning settings or None if disabled
    """
    model_type = getattr(config, 'model_type', 'xgboost').lower()
    
    model_tuning = getattr(config, 'tuning', {}).get(model_type, {})
    
    if model_tuning and model_tuning.get('enabled', False):
        return model_tuning
    
    if task_name:
        task_tuning = getattr(config, 'tuning', {}).get(task_name, {})
        if task_tuning and task_tuning.get('enabled', False):
            warnings.warn(f"Using legacy '{task_name}' tuning config. Consider migrating to '{model_type}' tuning.")
            return task_tuning
    
    return None


def is_tuning_enabled(config, task_name=None):
    """Check if tuning is enabled for current model"""
    return get_tuning_config(config, task_name) is not None


def get_param_distributions(config, task_type='regression'):
    """
    Get parameter distributions for RandomizedSearchCV.
    Automatically selects appropriate params based on task type (regression vs classification).
    """
    tuning_config = get_tuning_config(config)
    
    if tuning_config is None:
        return {}
    
    param_dist = tuning_config.get('param_distributions', {})
    model_type = getattr(config, 'model_type', 'xgboost').lower()
    
    if model_type == 'logistic_regression':
        if task_type == 'regression':
            return param_dist.get('regression', param_dist)
        else:
            return param_dist.get('classification', param_dist)
    
    return param_dist


def convert_param_distributions_for_cuml(param_distributions, model_type):
    """
    Convert sklearn parameter distributions to cuML-compatible format.
    cuML has some limitations compared to sklearn.
    """
    cuml_params = {}
    
    for param_name, values in param_distributions.items():
        # Handle both 'param' and 'model__param' formats
        clean_name = param_name.replace('model__', '')
        
        if model_type == 'random_forest':
            if clean_name in ['max_features', 'bootstrap', 'min_samples_split', 'min_samples_leaf']:
                continue  
            cuml_params[param_name] = values
            
        elif model_type == 'svm':
            if clean_name == 'kernel':
                cuml_params[param_name] = [v for v in values if v in ['rbf', 'poly', 'sigmoid']]
            elif clean_name == 'gamma':
                numeric_gammas = [v for v in values if isinstance(v, (int, float))]
                if numeric_gammas:
                    cuml_params[param_name] = numeric_gammas
            else:
                cuml_params[param_name] = values
                
        elif model_type == 'logistic_regression':
            if clean_name in ['max_iter', 'C', 'tol']:
                cuml_params[param_name] = values
            
        else:
            cuml_params[param_name] = values
    
    return cuml_params


def setup_tuning_search(model_class, param_distributions, tuning_config, base_params, 
                        random_state=42, use_gpu=False, model_type='xgboost', task_type='regression'):
    """
    Setup RandomizedSearchCV with appropriate parameters for the model type.
    Uses cuML for GPU-accelerated tuning when available and appropriate.
    
    Args:
        model_class: Model class to tune OR an already instantiated Pipeline
        param_distributions: Parameter distributions
        tuning_config: Tuning configuration dictionary
        base_params: Base model parameters (used only if model_class is a class)
        random_state: Random state for reproducibility
        use_gpu: Whether to use GPU for tuning
        model_type: Type of model (xgboost, random_forest, svm, etc.)
        task_type: 'regression' or 'classification'
    
    Returns:
        RandomizedSearchCV or cuML RandomizedSearchCV instance
    """
    n_iter = tuning_config.get('n_iter', 20)
    cv = tuning_config.get('cv', 3)
    n_jobs = tuning_config.get('n_jobs', -1)
    scoring = tuning_config.get('scoring', None)
    
    if isinstance(model_class, type):
        is_pipeline = False
        base_estimator = model_class(**base_params)
    else:
        is_pipeline = True
        base_estimator = model_class
    
    if use_gpu and model_type in ['random_forest', 'svm', 'logistic_regression']:
        try:
            from cuml.model_selection import RandomizedSearchCV as cuMLRandomizedSearchCV
            
            cuml_param_dist = convert_param_distributions_for_cuml(param_distributions, model_type)
            
            if cuml_param_dist: 
                print(f"  Using cuML RandomizedSearchCV on GPU for {model_type}")
                
                search = cuMLRandomizedSearchCV(
                    estimator=base_estimator,
                    param_distributions=cuml_param_dist,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    random_state=random_state,
                    verbose=1,
                    refit=True
                )
                return search
            else:
                print(f"  No cuML-compatible parameters for {model_type}, falling back to sklearn")
        except ImportError:
            print(f"  cuML not available for tuning, using sklearn")
        except Exception as e:
            print(f"  Could not use cuML for tuning: {e}, falling back to sklearn")
    
    from sklearn.model_selection import RandomizedSearchCV
    
    print(f"  Using sklearn RandomizedSearchCV on CPU for {model_type}")
    
    if n_jobs == -1:
        n_jobs = min(4, os.cpu_count() or 1)  
    
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        refit=True
    )
    
    return search


def get_tuning_model_for_gpu(config, task_type='regression'):
    """
    Get the appropriate model class and parameters for GPU tuning.
    Returns (model_class, base_params, is_gpu) tuple.
    """
    from models.model_factory import get_model, get_model_params, check_gpu_available
    
    model_type = getattr(config, 'model_type', 'xgboost').lower()
    use_gpu = getattr(config, 'gpu', {}).get('use_if_available', True)
    gpu_available = check_gpu_available() if use_gpu else False
    
    ModelClass, is_cuml = get_model(config, task_type)
    
    base_params = get_model_params(config, task_type)
    
    if is_cuml and model_type == 'random_forest':
        for param in ['n_jobs', 'max_features', 'bootstrap', 'min_samples_split', 'min_samples_leaf']:
            base_params.pop(param, None)
    elif is_cuml and model_type == 'svm':
        base_params.pop('probability', None)
    elif is_cuml and model_type == 'logistic_regression':
        for param in ['n_jobs', 'solver']:
            base_params.pop(param, None)
    
    return ModelClass, base_params, is_cuml and gpu_available