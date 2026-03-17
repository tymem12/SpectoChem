import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    @property
    def task(self) -> str:
        return self._config.get('task', 'pairs')

    @property
    def model_type(self) -> str:
        return self._config.get('model_type', 'xgboost')

    @property
    def random_state(self) -> int:
        """Get global random state from config."""
        return self._config.get('random_state', 42)

    @property
    def experiments(self) -> List[Dict[str, Any]]:
        """Get experiments list from config."""
        return self._config.get('experiments', [])

    @property
    def hydra_overrides(self) -> List[str]:
        """Get Hydra overrides list from config."""
        return self._config.get('hydra_overrides', [])

    @property
    def hydra_config_dir(self) -> str:
        """Get Hydra config directory from config."""
        return self._config.get('hydra_config_dir', './config')

    @property
    def task_to_hydra_exp(self) -> Dict[str, str]:
        """Get task to Hydra experiment mapping."""
        return self._config.get('task_to_hydra_exp', {})

    @property
    def gpu(self) -> Dict[str, Any]:
        return self._config.get('gpu', {'use_if_available': True})

    @property
    def custom_tasks(self) -> Dict[str, Any]:
        return self._config.get('custom_tasks', {})

    @property
    def data(self) -> Dict[str, Any]:
        return self._config['data']

    @property
    def binary(self) -> Dict[str, Any]:
        return self._config.get('binary', {})

    @property
    def binary_pairs(self) -> Dict[str, Any]:
        return self._config.get('binary_pairs', {})

    @property
    def multilabel(self) -> Dict[str, Any]:
        return self._config.get('multilabel', {})

    @property
    def targets(self) -> Dict[str, Any]:
        return self._config['targets']

    @property
    def descriptor(self) -> Dict[str, Any]:
        return self._config.get('descriptor', {'type': 'soap'})

    @property
    def soap(self) -> Dict[str, Any]:
        return self._config.get('soap', {})

    @property
    def acsf(self) -> Dict[str, Any]:
        return self._config.get('acsf', {})

    @property
    def tuning(self) -> Dict[str, Any]:
        return self._config.get('tuning', {})

    @property
    def xgboost(self) -> Dict[str, Any]:
        """Get XGBoost params with random_state injected."""
        params = self._config.get('xgboost', {}).copy()
        params['random_state'] = self.random_state
        return params

    @property
    def random_forest(self) -> Dict[str, Any]:
        """Get Random Forest params with random_state injected."""
        params = self._config.get('random_forest', {}).copy()
        params['random_state'] = self.random_state
        return params

    @property
    def svm(self) -> Dict[str, Any]:
        return self._config.get('svm', {})

    @property
    def dummy(self) -> Dict[str, Any]:
        """Get Dummy model params with random_state injected."""
        params = self._config.get('dummy', {}).copy()
        params['random_state'] = self.random_state
        return params

    @property
    def logistic_regression(self) -> Dict[str, Any]:
        """Get Logistic Regression params with random_state injected."""
        params = self._config.get('logistic_regression', {}).copy()
        params['random_state'] = self.random_state
        return params

    @property
    def mlp(self) -> Dict[str, Any]:
        """Get MLP params with random_state injected."""
        params = self._config.get('mlp', {}).copy()
        params['random_state'] = self.random_state
        return params

    @property
    def output(self) -> Dict[str, Any]:
        return self._config['output']

    def get(self, key: str, default=None):
        return self._config.get(key, default)


def load_config(config_path: str = "config.yaml") -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)