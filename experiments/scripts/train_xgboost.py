import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from gjepa.utils.config import resolve_config
from gjepa.config import GraphLevelExperimentConfig
from gjepa.datasets.graph_level import GraphLevelDataModule
from gjepa.utils.feature_extraction import extract_graph_features_from_dataloader

from gjepa.utils.misc import import_from_string
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from experiments.training_utils import save_metrics
from sklearn.preprocessing import StandardScaler


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    raw_config = resolve_config(cfg)
    config = GraphLevelExperimentConfig.from_raw_config(raw_config)

    if config.pos_encoding is None:
        raise RuntimeError("pos_encoding must be set in config to compute descriptors for XGBoost")

    if config.dataset.task_type != "binary":
        raise NotImplementedError("Only binary dataset.task_type is supported by train_xgboost.py")

    datamodule = GraphLevelDataModule(
        dataset_config=config.dataset,
        batch_size=config.training.batch_size,
        pos_enc_path=config.pos_encoding.file if config.pos_encoding else None,
    )

    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    pooling = config.model.backbone.get("pooling", "mean")

    X_train, y_train = extract_graph_features_from_dataloader(
        train_loader, encoding_field="positional_encoding", pooling=pooling
    )
    
    X_val, y_val = extract_graph_features_from_dataloader(
        val_loader, encoding_field="positional_encoding", pooling=pooling
    )
    
    X_test, y_test = extract_graph_features_from_dataloader(
        test_loader, encoding_field="positional_encoding", pooling=pooling
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   
    X_val = scaler.transform(X_val)           
    X_test = scaler.transform(X_test)

    xgb_cls_path = config.model.backbone.get("gnn_cls", "gjepa.models.backbones.xgboost.XGBoostBinaryBackbone")
    XGBoostCls = import_from_string(xgb_cls_path)
    backbone_config = dict(config.model.backbone) if config.model and config.model.backbone else {}

    backbone = XGBoostCls(backbone_config)

    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([np.full(len(X_train), -1, dtype=int), np.zeros(len(X_val), dtype=int)])
    ps = PredefinedSplit(test_fold)

    cfg_search_space = config.model.backbone.get("search_space", None)
    search_n_iter = int(config.model.backbone.get("search_n_iter", 25))

    if cfg_search_space is None:
        backbone.fit(X_train, y_train)
    else:
        search_space = {k: list(v) for k, v in dict(cfg_search_space).items()}
        estimator = backbone.model

        n_iter_search = min(search_n_iter, sum(len(v) for v in search_space.values()))
        rs = RandomizedSearchCV(
            estimator,
            param_distributions=search_space,
            n_iter=n_iter_search,
            scoring="roc_auc",
            cv=ps,
            verbose=1,
            n_jobs=-1,
            refit=True,
            random_state=0,
        )

        rs.fit(X_search, y_search)
        print("Best params:", rs.best_params_)

        backbone.model = rs.best_estimator_

    y_train_pred = backbone.predict(X_train)
    y_val_pred = backbone.predict(X_val)
    y_test_pred = backbone.predict(X_test)

    y_train_prob = backbone.model.predict_proba(X_train)[:, 1]
    y_val_prob = backbone.model.predict_proba(X_val)[:, 1]
    y_test_prob = backbone.model.predict_proba(X_test)[:, 1]

    metrics = {}
    metrics["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
    metrics["val_accuracy"] = float(accuracy_score(y_val, y_val_pred))
    metrics["test_accuracy"] = float(accuracy_score(y_test, y_test_pred))

    metrics["train_auc"] = float(roc_auc_score(y_train, y_train_prob))
    metrics["val_auc"] = float(roc_auc_score(y_val, y_val_prob))
    metrics["test_auc"] = float(roc_auc_score(y_test, y_test_prob))

    print("Metrics:", metrics)

    log_dir = Path(config.training.experiment_dir)
    os.makedirs(log_dir, exist_ok=True)

    model_path = log_dir / "xgboost_model.json"
    backbone.model.save_model(str(model_path))

    save_metrics(metrics, log_dir)

    df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_test_pred,
    })
    
    if y_test_prob is not None:
        df["y_score"] = y_test_prob
    df.to_csv(log_dir / f"xgboost_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()