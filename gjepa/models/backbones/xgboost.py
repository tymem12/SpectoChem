import xgboost as xgb
from sklearn.metrics import accuracy_score
from gjepa.utils.feature_extraction import extract_graph_features_from_dataloader
import torch

class XGBoostBinaryBackbone:
    
    handles_pos_encoding = True
    
    def __init__(self, config):
        self.config = config
        
        params = self.config.get('params', {})
        self.model = xgb.XGBClassifier(**params)
        
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def fit_dataloader(self, dataloader):
        X, y = extract_graph_features_from_dataloader(
                dataloader, 
                encoding_field="positional_encoding", 
                pooling=self.config.get('pooling', 'mean')
            )
        self.fit(X, y)
        
        preds = self.predict(X)
        return torch.tensor(preds, dtype=torch.float32)