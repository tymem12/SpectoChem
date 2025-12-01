import torch
import numpy as np
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

def extract_graph_features_from_dataloader(
    dataloader,
    encoding_field: str = "positional_encoding",
    pooling: str = "mean"
):
    """
    Extract graph-level feature vectors from a PyG dataloader.
    Used for feature extraction for classic ML models like XGBoost.
    
    Args:
        dataloader: torch_geometric.loader.DataLoader
        encoding_field: attribute name on batch (e.g., "positional_encoding", "vector")
        pooling: 'mean', 'sum', or 'max' for node->graph aggregation
    
    Returns:
        X: numpy array [num_graphs, feature_dim]
        y: numpy array [num_graphs, ...] (labels)
    """
    X_list, y_list = [], []
    
    for batch in dataloader:        
        if not hasattr(batch, encoding_field):
            raise AttributeError(f"Batch missing '{encoding_field}'. Available: {batch.__dict__.keys()}")
        
        encoding = getattr(batch, encoding_field)  
        graph_idx = batch.batch  
        
        
        if encoding.dim() == 2 and encoding.size(0) == batch.batch.size(0):
            graph_idx = batch.batch
            if pooling == "mean":
                graph_features = global_mean_pool(encoding, graph_idx)
            elif pooling == "sum":
                graph_features = global_add_pool(encoding, graph_idx)
            elif pooling == "max":
                graph_features = global_max_pool(encoding, graph_idx)
            else:
                raise ValueError("Bad pooling")
        elif encoding.dim() == 2 and encoding.size(0) == batch.num_graphs:
            graph_features = encoding  
        else:
            raise RuntimeError(
                f"Unexpected encoding shape {encoding.shape}: "
                "not node-level nor graph-level"
            )

        X_list.append(graph_features.detach().cpu().numpy())
        y_list.append(batch.y.detach().cpu().numpy())
    
    X = np.concatenate(X_list, axis=0)  
    y = np.concatenate(y_list, axis=0)
    
    y = y.reshape(-1) # Flatten y to 1D 
    
    return X, y