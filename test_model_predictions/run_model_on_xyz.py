# @TODO: Add block3 in output.json


### MODELS

import torch
from torch import nn
from torch_geometric.nn import radius_graph, global_mean_pool
from torch_geometric.data import Data

try:
    from torch_geometric.nn.models import SchNet
except ImportError:
    from torch_geometric.nn.models.schnet import SchNet

class SchNetEncoder(SchNet):
    handles_pos_encoding = True

    def __init__(self,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0,
                 max_num_neighbors: int = 64,
                 readout: str = 'add',
                 **kwargs):
        super().__init__(hidden_channels=hidden_channels,
                         num_filters=num_filters,
                         num_interactions=num_interactions,
                         num_gaussians=num_gaussians,
                         cutoff=cutoff,
                         max_num_neighbors=max_num_neighbors,
                         readout=readout

                         )
        self.kwargs = kwargs
        self.out_channels = self.hidden_channels

    def forward(self, batch: Data):
        pos = batch.pos
        z = batch.z
        batch = batch.batch
        batch = torch.zeros_like(z) if batch is None else batch
        h = self.embedding(z)

        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h 


from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT

class GATEncoder(nn.Module):
    handles_pos_encoding = False

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0,
        heads: int = 4,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs

        self.out_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels)

        self.gnn = GAT(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout,
            heads=heads
        )

    def forward(self, batch: Data):
        x = self.embedding(batch.z)
        h = self.gnn(x, batch.edge_index, edge_weight=batch.edge_weight)
        return h

from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv


class GINEEncoder(nn.Module):
    handles_pos_encoding = False

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs
        self.out_channels = hidden_channels
        self.dropout = dropout

        self.embedding = nn.Embedding(100, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

            self.convs.append(GINEConv(
                nn=mlp, edge_dim=1
            ))

    def forward(self, batch: Data):
        x = self.embedding(batch.z)

        edge_attr = batch.edge_weight.view(-1, 1)

        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN

class GCNEncoder(nn.Module):
    handles_pos_encoding = False

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs
        self.out_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels)

        self.gnn = GCN(
            hidden_channels, hidden_channels, num_layers, hidden_channels, dropout
        )

    def forward(self, batch: Data):
        x = self.embedding(batch.z)
        h = self.gnn(x, batch.edge_index, edge_weight=batch.edge_weight)
        return h


#######################

import argparse
import glob
import json
import logging
import os
import traceback
from typing import List, Dict, Any, Tuple

import torch
from torch_cluster import radius_graph
# Assuming the user has PyTorch Geometric installed based on the radius_graph import
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ==========================================
# 0. LOGGING CONFIGURATION
# ==========================================

import os
import sys

# Enable ANSI escape code support on Windows 10+ legacy consoles
if os.name == 'nt':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11, STD_ERROR_HANDLE = -12
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

class ColoredFormatter(logging.Formatter):
    """Custom logging formatter to add colors based on the log level."""
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[90m',       # Gray
        'INFO': '\033[96m',        # Cyan
        'WARNING': '\033[93m',     # Yellow
        'ERROR': '\033[91m',       # Red
        'CRITICAL': '\033[1;91m',  # Bold Red
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        format_str = f"%(asctime)s | {log_color}%(levelname)-8s{self.RESET} | %(message)s"
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# ==========================================
# 1. GRAPH TRANSFORMATION (From User)
# ==========================================

def calc_edge_weight(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    diff = pos[row] - pos[col]             # [E, 3]
    edge_weight = diff.norm(dim=-1).clamp(min=1e-12)        # [E]
    return edge_weight

class AddEdgesAndDistances(object):
    def __init__(self, cutoff=10.0):
        self.cutoff = cutoff
    
    def __call__(self, data: Data) -> Data:
        edge_index = radius_graph(data.pos, r=self.cutoff, loop=False, max_num_neighbors=64)
        edge_weight = calc_edge_weight(data.pos, edge_index)
        data.edge_index = edge_index
        data.edge_weight = edge_weight
        return data

# ==========================================
# 2. PLACEHOLDER FUNCTIONS (To be modified)
# ==========================================
PERIODIC_TABLE = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def parse_xyz(filepath: str) -> Data:
    """Parses an XYZ file into 3D coordinates (pos) and atomic numbers (z)."""
    coords = []
    atomic_numbers = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            raise ValueError(f"File {filepath} is too short to be a valid XYZ.")
        
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 4:
                elem = parts[0].capitalize()
                # Parse symbol or fallback to standard atomic number lookup
                z_val = PERIODIC_TABLE[elem]
                atomic_numbers.append(z_val)
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    if not coords:
        raise ValueError(f"No valid coordinates found in {filepath}.")
        
    pos = torch.tensor(coords, dtype=torch.float)
    z = torch.tensor(atomic_numbers, dtype=torch.long)
    
    return Data(pos=pos, z=z)


class GraphModelWrapper(nn.Module):
    """Combines an Encoder with graph pooling and a prediction head."""
    def __init__(self, encoder: nn.Module, hidden_channels: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, batch: Data) -> torch.Tensor:
        # Encoder returns node representations [N, hidden_channels]
        h_node = self.encoder(batch)
        
        batch_idx = getattr(batch, 'batch', None)
        if batch_idx is None:
            batch_idx = torch.zeros(h_node.size(0), dtype=torch.long, device=h_node.device)
            
        # Global pooling to convert node embeddings into a single graph embedding
        h_graph = global_mean_pool(h_node, batch_idx)
        
        # Scalar output per graph
        return self.head(h_graph)

def load_lightning_model(ckpt_path: str, model_name: str) -> nn.Module:
    """
    Loads PyTorch Lightning checkpoint and maps it to the designated encoder model.
    """
    logging.info(f"Loading model checkpoint from {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Map target string to pre-imported Encoder classes
    model_mapping = {
        "schnet": SchNetEncoder,
        "gat": GATEncoder,
        "gine": GINEEncoder,
        "gcn": GCNEncoder,
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model string: {model_name}. Options: {list(model_mapping.keys())}")

    # 1. Load checkpoint payload
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    hparams = checkpoint["hyper_parameters"]["config"]["model"]["backbone"]

    # 2. Instantiate Encoder and Full Graph Wrapper
    EncoderClass = model_mapping[model_name]
    
    encoder = EncoderClass(**hparams)

    model = GraphModelWrapper(encoder, hidden_channels=hparams["hidden_channels"])

    # 3. Remap state_dict prefixes (e.g. 'gnn.gnn.' -> 'encoder.', 'predictor.mlp.' -> 'head.')
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Strip optional 'model.' prefix
        if new_key.startswith("model."):
            new_key = new_key[6:]
            
        # Map encoder keys
        if new_key.startswith("gnn.gnn."):
            new_key = "encoder." + new_key[len("gnn.gnn."):]
        elif new_key.startswith("gnn."):
            new_key = "encoder." + new_key[len("gnn."):]
            
        # Map head/predictor keys
        if new_key.startswith("predictor.mlp."):
            new_key = "head." + new_key[len("predictor.mlp."):]
        elif new_key.startswith("predictor."):
            new_key = "head." + new_key[len("predictor."):]

        cleaned_state_dict[new_key] = value

    # 4. Load weights into the architecture
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
        logging.info("Successfully loaded state_dict with exact key match.")
    except Exception:
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        logging.warning(
            f"Loaded state_dict with non-strict match.\n"
            f"  Missing keys: {missing_keys}\n"
            f"  Unexpected keys: {unexpected_keys}"
        )

    model.eval()
    return model

# ==========================================
# 3. CORE LOGIC & PREDICTIONS
# ==========================================

def update_json_results(
    json_path: str, 
    task: str, 
    model_name: str, 
    state: str, 
    metadata: Dict[str, Any], 
    predictions: Dict[str, Any]
):
    """Updates the JSON file safely without nuking other states or models."""
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Failed to read {json_path}. It will be overwritten.")
            
    if task not in data:
        data[task] = {}
        
    if model_name not in data[task]:
        # Brand new model entry
        data[task][model_name] = {
            "metadata": metadata,
            "predictions": {}
        }
    else:
        # Update metadata, but keep existing predictions
        data[task][model_name]["metadata"] = metadata
        
        # If the task is binary, we overwrite the whole predictions block for this model
        if task == "binary":
            data[task][model_name]["predictions"] = {}

    # Merge in new predictions
    for filename, pred_data in predictions.items():
        if task == "binary":
            data[task][model_name]["predictions"][filename] = pred_data
        else:
            # For regression, ensure nested state structure exists
            if filename not in data[task][model_name]["predictions"]:
                data[task][model_name]["predictions"][filename] = {}
            # Overwrite only the specific state digit
            data[task][model_name]["predictions"][filename][state] = pred_data

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    logging.info(f"Results successfully saved to {json_path}.")

def process_state(args, state: str, xyz_files: List[str], output_json_path: str):
    """Processes all files for a specific checkpoint (either binary, or a specific state)."""
    # 1. Resolve Checkpoint Path
    if args.task == "binary":
        ckpt_path = os.path.join("models", str(args.seed), args.task, args.model, f"block_3_{args.block3}.ckpt")
    else:
        ckpt_path = os.path.join("models", str(args.seed), args.task, args.model, state, f"block_3_{args.block3}.ckpt")

    try:
        model = load_lightning_model(ckpt_path, args.model)
    except Exception as e:
        logging.error(f"Failed to load model for state {state}.")
        logging.error(traceback.format_exc())
        return

    # 2. Parse Data
    dataset = []
    file_mapping = [] # Keep track of which file corresponds to which graph
    graph_transform = AddEdgesAndDistances() if args.model != "schnet" else None
    
    success_count = 0
    for fpath in xyz_files:
        try:
            data = parse_xyz(fpath)
            if graph_transform:
                data = graph_transform(data)
            
            dataset.append(data)
            file_mapping.append(os.path.basename(fpath))
            success_count += 1
        except Exception as e:
            logging.warning(f"Error parsing {fpath}: {e}")

    if success_count == 0:
        logging.error("No files were successfully parsed. Aborting inference.")
        return

    # 3. Batch Inference
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    raw_predictions = []
    
    logging.info(f"Running inference on {success_count} files (batch size 32)...")
    with torch.no_grad():
        for batch in loader:
            try:
                preds = model(batch).squeeze(-1) # Assuming shape [Batch, 1]
                raw_predictions.extend(preds.tolist())
            except Exception as e:
                logging.error(f"Error during model forward pass.")
                logging.error(traceback.format_exc())
                return

    # 4. Transform Predictions and Format Dictionary
    metadata = {
        "xyz_dir": args.xyz_dir,
        "scanned_files": len(xyz_files),
        "successful_files": success_count
    }
    
    predictions_dict = {}
    
    for filename, raw_val in zip(file_mapping, raw_predictions):
        if args.task == "binary":
            proba = torch.sigmoid(torch.tensor(raw_val)).item()
            cls = 1 if proba > 0.5 else 0
            predictions_dict[filename] = {
                "proba": proba, 
                "class": cls
            }
        elif args.task == "f":
            f_val = 10 ** raw_val
            predictions_dict[filename] = {
                "log10f": raw_val,
                "f": f_val
            }
        elif args.task == "lambda":
            # Guard against division by zero
            safe_val = raw_val if abs(raw_val) > 1e-12 else 1e-12
            lam_val = 1239.8419843320026224 / safe_val
            predictions_dict[filename] = {
                "energy": raw_val,
                "lambda": lam_val
            }

    # 5. Save/Update JSON
    update_json_results(output_json_path, args.task, args.model, state, metadata, predictions_dict)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ML Models on XYZ coordinates.")
    
    # Required strict arguments
    parser.add_argument("--model", type=str, required=True, choices=["schnet", "gat", "gcn", "gine"], help="Model architecture.")
    parser.add_argument("--seed", type=int, required=True, choices=[1234, 2137, 42], help="Random seed used for the run.")
    parser.add_argument("--block3", type=str, required=True, choices=["null", "test"], help="Block 3 setting.")
    parser.add_argument("--task", type=str, required=True, choices=["binary", "f", "lambda"], help="Task target.")
    parser.add_argument("--xyz-dir", type=str, required=True, help="Path to directory containing .xyz files.")
    
    # Optional / Conditional arguments
    parser.add_argument("--state", type=str, default="all", help="Excited state number (0-9) or 'all'. Used only if task is not binary.")
    parser.add_argument("--output-dir", type=str, default="output_model_predictions", help="Directory to save the resulting seed_<SEED>.json file.")

    args = parser.parse_args()

    # Setup Colored Logging
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    # Clear existing handlers just in case, then apply ours
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    logging.info("=== Starting Model Evaluation ===")
    logging.info(f"Parameters: Model={args.model}, Task={args.task}, Seed={args.seed}, Block3={args.block3}")
    logging.info(f"Data Directory: {args.xyz_dir}")

    # Discover Files
    search_pattern = os.path.join(args.xyz_dir, "*.xyz")
    xyz_files = glob.glob(search_pattern)
    logging.info(f"Found {len(xyz_files)} .xyz files.")

    if not xyz_files:
        logging.error("No XYZ files found. Exiting.")
        return

    # Setup Output Directory and File Path
    os.makedirs(args.output_dir, exist_ok=True)
    output_json_path = os.path.join(args.output_dir, f"seed_{args.seed}.json")
    logging.info(f"Output will be saved to: {output_json_path}")

    # Determine which states to process
    if args.task == "binary":
        states_to_process = ["none"] # Dummy state for binary
    else:
        if args.state.lower() == "all":
            states_to_process = [str(i) for i in range(10)]
        elif args.state.isdigit() and 0 <= int(args.state) <= 9:
            states_to_process = [args.state]
        else:
            logging.error("Invalid --state argument. Must be 'all' or a digit 0-9.")
            return

    # Process each state
    for state in states_to_process:
        if args.task != "binary":
            logging.info(f"--- Processing State: {state} ---")
        process_state(args, state, xyz_files, output_json_path)

    logging.info("=== Evaluation Complete ===")

if __name__ == "__main__":
    main()