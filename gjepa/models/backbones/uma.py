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
    )

def pyg_to_atomicdata(data: Data):
    task = task_name

    if data.edge_index is None:
        task = [task]

    # `task_name` has to be specified
    atomicdata = AtomicData.from_ase(
        pyg_to_ase(data),
        # `eSCNMDBackbone::_generate_graph` generates the molecue graph - it doesn't have edge_index and creates it on the fly
        # (based on `self.cutoff`=6, `self.max_neighbors`=300); the question is what is "otf_graph"
        # (with this the`cell` parameter from `Atoms` structure is ignored)
        # ^ could also use get_molecule instead get_structure in AtomsData.from_ase
        #r_edges=False,          # enable radius-based edges
        #radius=6.0,            # typical UMA cutoff
        #max_neigh=32,          # max neighbors per atom
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
        result = self.mlip_model_backbone(data)
        emb = result["node_embedding"]
        emb = pool_over_heads(emb)

        return emb

    def __init__(self, predictor_name: str = "uma-s-1p1"):
        """
        predictor_name: name of the pretrained UMA model to use
        DEVICE: DEVICE to load the predictor on
        """
        super().__init__()
        self.predictor_name = predictor_name

        mlip = pretrained_mlip.get_predict_unit(
            predictor_name,
            device=DEVICE,
            # default backbone config:
            # {'num_experts': 32, 'sphere_channels': 128, 'max_neighbors': 300, 'edge_channels': 128, 'regress_stress': True,
            # 'norm_type': 'rms_norm_sh', 'cs_emb_grad': True, 'moe_layer_type': 'pytorch', 'max_num_elements': 100,
            # 'otf_graph': True, 'cutoff': 6, 'regress_forces': True, 'hidden_channels': 128, 'chg_spin_emb_type': 'rand_emb',
            # 'moe_dropout': 0.05, 'use_global_embedding': False, 'mmax': 2, 'use_pbc_single': True, 'num_distance_basis': 64,
            # 'num_layers': 4, 'ff_type': 'spectral', 'model': 'fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone',
            # 'use_composition_embedding': True, 'lmax': 2, 'use_pbc': True, 'distance_function': 'gaussian',
            # 'direct_forces': False, 'act_type': 'gate', 'dataset_list': ['oc20', 'omol', 'omat', 'odac', 'omc'],
            # 'model_version': 1.1, 'always_use_pbc': False, 'activation_checkpointing': True, 'radius_pbc_version': 2}
            overrides=dict(
                freeze_backbone=True,
                backbone=dict(
                    # without these overrides `eSCNMDBackbone::_get_displacement_and_cell` makes `data_dict["pos"]` and
                    # `displacement` require grad which causes a long delay (due to optimizer step) after each train step in
                    # the trainer (it takes about 2x as much as uma forward on cpu)
                    regress_forces=False,
                    regress_stress=False
                )
            )
        )

        mlip_model_module: HydraModel = mlip.model.module

        # without this we get the "some tensors were on cpu while others on gpu" error
        mlip_model_module.to(DEVICE)

        # TODO: if `AveragedModel` "is a wrapper around a model that keeps a running average of the parameters during training"
        # then maybe we should use `mlip.model` instead of using the backbone directly?
        #
        # we could also use `mlip_model_module` directly since it calls the backbone, but it also returns other unnecessary information -
        # a dict with the following keys: ['oc20_energy', 'oc20_embeddings', 'oc20_forces', 'oc20_stress', 'odac_energy',
        # 'odac_embeddings', 'odac_forces', 'odac_stress', 'omat_energy', 'omat_embeddings', 'omat_forces', 'omat_stress',
        # 'omc_energy', 'omc_embeddings', 'omc_forces', 'omc_stress', 'omol_energy', 'omol_embeddings', 'omol_forces', 'omol_stress']
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
