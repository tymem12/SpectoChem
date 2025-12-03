import torch
import numpy as np
from torch import nn
from ase import Atoms
from torch_geometric.data import Batch, Data
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.base import HydraModel

from experiments.training_utils import DEVICE

task_name = "omol"

def pyg_to_ase(data):
    pos = data.pos.cpu().numpy()
    numbers = data.z.cpu().numpy()
    return Atoms(
        numbers=numbers,
        positions=pos,
        cell=np.eye(3),        # non-periodic
        pbc=[False, False, False]
    )

def pyg_to_atomicdata(data: Data):
    task = task_name

    if data.edge_index is None:
        task = [task]

    # `task_name` has to be specified
    atomicdata = AtomicData.from_ase(
        pyg_to_ase(data),
        r_edges=True,          # enable radius-based edges
        radius=6.0,            # typical UMA cutoff
        max_neigh=32,          # max neighbors per atom
        task_name=task
    )

    return atomicdata

# TODO: instead of pooling over heads we could also concatenate last two dims and have one larger embedding
def pool_over_heads(atom_embeddings: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    Pool UMA per-head embeddings into a single per-atom embedding.
    
    Args:
        atom_embeddings: Tensor of shape [N_atoms, n_heads, hidden_dim]
        method: Pooling method, "mean", "sum", or "max"
        
    Returns:
        Tensor of shape [N_atoms, hidden_dim] (head dimension pooled)
    """
    if method == "mean":
        return atom_embeddings.mean(dim=1)  # pool over heads
    elif method == "sum":
        return atom_embeddings.sum(dim=1)
    elif method == "max":
        return atom_embeddings.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling method: {method}")

# TODO: check if UMA interprets the batch correctly (if the result is similar to predicting on individual data points
# (`torch.cat(list(map(predict, batch.to_data_list())))`))
def batch_to_atomicdata(batch: Batch) -> AtomicData:
    """
    Converts a PyG Batch to a UMA-ready AtomicData batch.
    
    Args:
        batch: PyG Batch containing multiple graphs/molecules
        task_name: str, required by AtomicData.from_ase
    
    Returns:
        AtomicData: batched UMA input with proper per-molecule embeddings
    """
    # Step 1: split batch into list of Data objects (1 per molecule)
    data_list = batch.to_data_list()

    # Step 3: convert ASE Atoms → AtomicData (UMA expects one molecule per AtomicData)
    atomicdata_list = list(map(
        pyg_to_atomicdata, data_list
    ))

    # Step 4: batch all AtomicData together
    atomicdata_batch = atomicdata_list_to_batch(atomicdata_list)

    return atomicdata_batch

class UMAEncoder(nn.Module):
    handles_pos_encoding = True

    def mlip_model_predict_embedding(self, data: Data):
        # `mlip.model` is HydraModel` (fairchem.core.models.base.HydraModel) (wrapped in `torch.optim.swa_utils.AveragedModel`,
        # with `fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone` as the backbone)
        # embedding.shape = [num_atoms_in_molecule, num_embeddings_per_atom, embedding_dim]
        # (e.g. = [2,               9,                   128]) 
        if isinstance(data, Batch):
            data = batch_to_atomicdata(data)
        else:
            data = pyg_to_atomicdata(data)

        data = data.to(DEVICE)

        # if using `mlip.model` (`HydraModel` wrapped in `AveragedModel`):
        # emb = self.mlip_model_backbone(data)[f"{task_name}_embeddings"]["embeddings"]
        emb = self.mlip_model_backbone(data)["node_embedding"]
        emb = pool_over_heads(emb)

        return emb

    def __init__(self, predictor_name: str = "uma-s-1p1"):
        """
        predictor_name: name of the pretrained UMA model to use
        DEVICE: DEVICE to load the predictor on
        """
        super().__init__()
        self.predictor_name = predictor_name

        mlip = pretrained_mlip.get_predict_unit(predictor_name, device=DEVICE)

        # TODO: maybe this isn't necessary by default?
        for param in mlip.model.parameters():
            param.requires_grad = False

        mlip_model_module: HydraModel = mlip.model.module

        # without this we get the "some tensors were on cpu while others on gpu" error
        mlip_model_module.to(DEVICE)

        # TODO: if `AveragedModel` "is a wrapper around a model that keeps a running average of the parameters during training"
        # then maybe we should use `mlip.model` instead of using the backbone directly?
        #
        # we could also use `mlip_model_module` directly since it calls the backbone, but it also returns other unnecessary information
        self.mlip_model_backbone: eSCNMDMoeBackbone = mlip_model_module.backbone

        # Two atoms within cutoff distance
        pos_dummy = torch.tensor([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]], dtype=torch.float)  # 1 Å apart
        z_dummy = torch.tensor([1, 6], dtype=torch.long)  # H and C
        batch_dummy = torch.zeros(2, dtype=torch.long)

        dummy_data = Data(pos=pos_dummy, z=z_dummy, batch=batch_dummy)
        with torch.no_grad():
            h_dummy = self(dummy_data)

        out_channels = h_dummy.shape[-1]

        self.out_channels = out_channels

    def forward(self, batch: Data):
        return self.mlip_model_predict_embedding(batch)
