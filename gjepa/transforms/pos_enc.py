# POSTIONAL ENCODING
import math
from typing import Literal, Optional
from scipy.special import sph_harm

import torch
import numpy as np
from torch_geometric.transforms import BaseTransform, AddLaplacianEigenvectorPE
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import (
    to_undirected,
    get_laplacian,
    to_scipy_sparse_matrix,
)
from gjepa.utils.graph_level import calc_edge_index, calc_edge_weight
try:
    from dscribe.descriptors import SOAP
except ModuleNotFoundError:
    SOAP = None  # dscribe optional; only needed for SOAP positional encoding
from ase import Atoms



class DummyOnesPE(BaseTransform):
    def __init__(self, dim_size: int, attr_name: str = "positional_encoding"):
        if dim_size <= 0:
            raise ValueError("dim_size must be a positive integer.")
        self.dim_size = dim_size
        self.attr_name = attr_name

    def __call__(self, data):
        num_nodes = data.num_nodes
        pe = torch.ones((num_nodes, self.dim_size))
        setattr(data, self.attr_name, pe)
        return data

    def __str__(self):
        return 'DummyOnesPE'

    

class FourierFrequencyPE(BaseTransform):
    """
    https://arxiv.org/pdf/2006.10739
    """
    def __init__(
        self,
        num_freqs: int = 32,
        sigma: float = 1.0,
        include_input: bool = False,
        random: bool = True,
        attr_name: str = "positional_encoding",
    ):
        self.num_freqs = int(num_freqs)
        self.sigma = float(sigma)
        self.include_input = bool(include_input)
        self.random = bool(random)
        self.attr_name = attr_name

        g = torch.Generator(device="cpu")

        if self.random:
            B = torch.normal(mean=0.0, std=self.sigma, size=(self.num_freqs, 3), generator=g)
        else:
            per_axis = [self.num_freqs // 3] * 3
            for i in range(self.num_freqs % 3):
                per_axis[i] += 1
            rows = []
            for f in (2.0 ** torch.arange(per_axis[0], dtype=torch.float32)):
                rows.append(torch.tensor([f, 0.0, 0.0]))
            for f in (2.0 ** torch.arange(per_axis[1], dtype=torch.float32)):
                rows.append(torch.tensor([0.0, f, 0.0]))
            for f in (2.0 ** torch.arange(per_axis[2], dtype=torch.float32)):
                rows.append(torch.tensor([0.0, 0.0, f]))
            B = torch.stack(rows, dim=0) if rows else torch.zeros((1, 3))

        self.B = B

    def __repr__(self):
        mode = "RFF" if self.random else "NeRF"
        return (f"{self.__class__.__name__}(K={self.num_freqs}, mode={mode}, "
                f"sigma={self.sigma}, include_input={self.include_input}, "
                f"attr='{self.attr_name}')")

    def _ensure_device_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if self.B.device != x.device or self.B.dtype != x.dtype:
            self.B = self.B.to(device=x.device, dtype=x.dtype)
        return self.B

    def __call__(self, data):
        if not hasattr(data, "pos"):
            raise AttributeError("Data.pos is required: [num_nodes, 3]).")
        pos = data.pos
        if pos.size(-1) != 3:
            raise ValueError(f"data.pos requires shape [N,3], obtained {tuple(pos.shape)}")

        B = self._ensure_device_dtype(pos)
        proj = (2.0 * math.pi) * (pos @ B.T) 
        enc = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        if self.include_input:
            enc = torch.cat([pos, enc], dim=-1)

        setattr(data, self.attr_name, enc)
        return data
    
    def __str__(self):
        return 'FourierEncoding'


class Raw3DCoordinatesPE(BaseTransform):
    def __init__(self, attr_name="positional_encoding"):
        self.attr_name = attr_name

    def __call__(self, data):
        if not hasattr(data, "pos"):
            raise AttributeError("Data.pos is required: [num_nodes, 3]).")
        pos = data.pos
        setattr(data, self.attr_name, pos)
        return data
    
    def __str__(self):
        return 'Raw3DCoordinates'



class SphericalHarmonicsPE(BaseTransform):
    def __init__(self, l_max: int = 3, attr_name: str = "positional_encoding",
                 center: bool = True, include_input: bool = False, eps: float = 1e-8):
        self.l_max = int(l_max)
        self.attr_name = attr_name
        self.center = bool(center)
        self.include_input = bool(include_input)
        self.eps = float(eps)

    def __repr__(self):
        return (f"{self.__class__.__name__}(l_max={self.l_max}, center={self.center}, "
                f"attr='{self.attr_name}', include_input={self.include_input})")

    def __call__(self, data):
        if not hasattr(data, "pos"):
            raise AttributeError("Data.pos is required:  [N, 3]).")
        pos = data.pos
        if pos.size(-1) != 3:
            raise ValueError(f"data.pos was required with the size [N,3]. Obtained {tuple(pos.shape)}")

        if self.center:
            pos = pos - pos.mean(dim=0, keepdim=True)

        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = torch.sqrt(x * x + y * y + z * z) + self.eps

        theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))
        phi = torch.atan2(y, x)

        theta_np = theta.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()

        enc_list = []
        for l in range(self.l_max + 1):
            Y_l0 = sph_harm(0, l, phi_np, theta_np)
            enc_list.append(np.real(Y_l0).reshape(-1, 1))

            if l > 0:
                for m in range(1, l + 1):
                    Y_lm = sph_harm(m, l, phi_np, theta_np)
                    phase = (-1.0) ** m
                    Y_posm = math.sqrt(2.0) * phase * np.real(Y_lm)  # corresponds to +m (cos-like)
                    Y_negm = math.sqrt(2.0) * phase * np.imag(Y_lm)  # corresponds to -m (sin-like)
                    enc_list.append(Y_posm.reshape(-1, 1))
                    enc_list.append(Y_negm.reshape(-1, 1))

        Y_real = np.concatenate(enc_list, axis=1)  # [N, (L+1)^2]
        Y_real = torch.from_numpy(Y_real).to(dtype=pos.dtype, device=pos.device)

        if self.include_input:
            Y_real = torch.cat([pos, Y_real], dim=-1)

        setattr(data, self.attr_name, Y_real)
        return data



class NerfPE(BaseTransform):

    """
    Wraps nerfstudio.field_components.encodings.NeRFEncoding for 3D (x,y,z) positions.

    Args:
        num_frequencies (int): Number of frequency bands per axis.
        min_freq_exp (float): Minimum frequency exponent (2**min_freq_exp).
        max_freq_exp (float): Maximum frequency exponent (2**max_freq_exp).
        include_input (bool): If True, append raw (x,y,z) to the encoding.
        attr_name (str): Attribute name to store on Data (default: "positional_encoding").
        normalize (bool): If True, normalize positions to [0,1] before encoding.
        aabb (tuple|Tensor|None): Axis-aligned bounds for normalization:
            ((xmin,ymin,zmin), (xmax,ymax,zmax)) or tensor shape [2,3].
            If None and normalize=True, per-graph min/max are used.
        implementation (str): "torch" or "tcnn" (if tiny-cuda-nn is available).
    """
    def __init__(self,
                 num_frequencies: int = 10,
                 min_freq_exp: float = 0.0,
                 max_freq_exp: float = 9.0,
                 include_input: bool = True,
                 attr_name: str = "positional_encoding",
                 normalize: bool = True,
                 implementation: str = "torch"):
        self.attr_name = attr_name
        self.normalize = normalize
        self.aabb = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0]], dtype=torch.float32)

        from nerfstudio.field_components.encodings import NeRFEncoding
        self.encoder = NeRFEncoding(
            in_dim=3,
            num_frequencies=num_frequencies,
            min_freq_exp=min_freq_exp,
            max_freq_exp=max_freq_exp,
            include_input=include_input,
            implementation=implementation,
        )

    @property
    def output_dim(self) -> int:
        return int(self.encoder.get_out_dim())

    def _normalize_to_01(self, pos: torch.Tensor) -> torch.Tensor:
        if self.aabb is not None:
            aabb = torch.as_tensor(self.aabb, device=pos.device, dtype=pos.dtype)
            pmin, pmax = aabb[0], aabb[1]
        else:
            pmin = pos.min(dim=0).values
            pmax = pos.max(dim=0).values
        scale = (pmax - pmin).clamp(min=1e-12)
        return (pos - pmin) / scale

    def __call__(self, data):
        if not hasattr(data, "pos"):
            raise AttributeError("Data.pos is required and must have shape [N, 3].")
        pos = data.pos
        if pos.dim() != 2 or pos.size(-1) != 3:
            raise ValueError(f"Expected Data.pos of shape [N,3], got {tuple(pos.shape)}")

        in_tensor = self._normalize_to_01(pos) if self.normalize else pos
        pe = self.encoder(in_tensor)
        setattr(data, self.attr_name, pe)
        return data

    def __str__(self):
        return f"NerfstudioNeRFPosEnc(out_dim={self.output_dim})"



class SOAPPE(BaseTransform):
    """
    Args:
        l_max: maximum SH degree L (uses real SH, size per L is (2L+1))
        num_radial: number of Gaussian radial basis channels K
        r_max: neighbor cutoff (Å). Also sets RBF support / centers range [0, r_max].
        radial_width: Gaussian width (σ). If None, uses spacing = r_max / max(1, K-1).
        use_cutoff: multiply by smooth cosine cutoff f_c(r) on [0, r_max]
        center: if True, center positions per-graph before building the neighbor graph
        use_existing_edges: if True and `data.edge_index` exists, uses it; otherwise builds radius graph
        attr_name: where to store the node descriptor (e.g., "positional_encoding")
        include_input: if True, concatenates raw `pos` (3 dims) to the output (not recommended for invariance)
        eps: numerical epsilon for safe divisions
    """
    def __init__(self,
                 l_max: int = 3,
                 num_radial: int = 10,
                 r_max: float = 5.0,
                 radial_width: Optional[float] = None,
                 use_cutoff: bool = True,
                 center: bool = False,
                 use_existing_edges: bool = False,
                 attr_name: str = "positional_encoding",
                 include_input: bool = False,
                 eps: float = 1e-8):
        self.l_max = int(l_max)
        self.num_radial = int(num_radial)
        self.r_max = float(r_max)
        self.radial_width = float(radial_width) if radial_width is not None else (
            (self.r_max / max(1, self.num_radial - 1)) if self.num_radial > 1 else self.r_max
        )
        self.use_cutoff = bool(use_cutoff)
        self.center = bool(center)
        self.use_existing_edges = bool(use_existing_edges)
        self.attr_name = attr_name
        self.include_input = bool(include_input)
        self.eps = float(eps)

        self._sh_slices = []
        start = 0
        for l in range(self.l_max + 1):
            m_sz = 2 * l + 1
            self._sh_slices.append((l, start, start + m_sz))
            start += m_sz
        self._S = (self.l_max + 1) ** 2

    def __repr__(self):
        D = self.output_dim()
        return (f"{self.__class__.__name__}(l_max={self.l_max}, num_radial={self.num_radial}, "
                f"r_max={self.r_max}, radial_width={self.radial_width:.3g}, "
                f"use_cutoff={self.use_cutoff}, center={self.center}, "
                f"use_existing_edges={self.use_existing_edges}, "
                f"attr='{self.attr_name}', include_input={self.include_input}, "
                f"out_dim={D})")

    def output_dim(self) -> int:
        # D = (L+1) * K*(K+1)/2
        K = self.num_radial
        return (self.l_max + 1) * (K * (K + 1) // 2) + (3 if self.include_input else 0)

    @torch.no_grad()
    def __call__(self, data):
        if not hasattr(data, "pos"):
            raise AttributeError("Data.pos is required with shape [N, 3].")
        pos = data.pos
        if pos.size(-1) != 3:
            raise ValueError(f"data.pos must have shape [N, 3]. Got {tuple(pos.shape)}")
        device, dtype = pos.device, pos.dtype
        N = pos.size(0)

        # Optional per-graph centering (uses data.batch if present)
        if self.center:
            if hasattr(data, "batch"):
                mean_pos = global_mean_pool(pos, data.batch)  # [num_graphs, 3]
                pos = pos - mean_pos[data.batch]
            else:
                pos = pos - pos.mean(dim=0, keepdim=True)

        # Build / use neighbor graph
        if self.use_existing_edges and hasattr(data, "edge_index"):
            edge_index = data.edge_index
        else:
            if hasattr(data, "batch"):
                edge_index = radius_graph(pos, r=self.r_max, batch=data.batch, loop=False, max_num_neighbors=128)
            else:
                edge_index = radius_graph(pos, r=self.r_max, loop=False, max_num_neighbors=128)
        row, col = edge_index[0], edge_index[1]

        rij = pos[col] - pos[row]  # [E,3]
        x, y, z = rij[:, 0], rij[:, 1], rij[:, 2]
        r = torch.linalg.norm(rij, dim=-1)  # [E]
        r_safe = torch.clamp(r, min=self.eps)
        cos_theta = torch.clamp(z / r_safe, -1.0, 1.0)
        theta = torch.acos(cos_theta)            # [E]
        phi = torch.atan2(y, x)                  # [E]

        # Optional smooth cutoff f_c(r) on [0, r_max]
        if self.use_cutoff:
            fc = 0.5 * (torch.cos(math.pi * torch.clamp(r, max=self.r_max) / self.r_max) + 1.0)
            fc = fc * (r <= self.r_max).to(dtype)
        else:
            fc = torch.ones_like(r, dtype=dtype, device=device)

        # --- Radial basis G (E x K) ---
        K = self.num_radial
        centers = torch.linspace(0.0, self.r_max, K, device=device, dtype=dtype) if K > 1 else torch.tensor([0.0], device=device, dtype=dtype)
        width = torch.tensor(self.radial_width, device=device, dtype=dtype)
        diff = (r.unsqueeze(-1) - centers.unsqueeze(0)) / torch.clamp(width, min=self.eps)
        G = torch.exp(-0.5 * diff * diff) * fc.unsqueeze(-1)  # [E,K]

        # --- Real spherical harmonics Y
        theta_np = theta.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()
        enc = []
        for l in range(self.l_max + 1):
            Y_l0 = sph_harm(0, l, phi_np, theta_np)  # complex [E]
            enc.append(np.real(Y_l0).reshape(-1, 1))
            if l > 0:
                for m in range(1, l + 1):
                    Y_lm = sph_harm(m, l, phi_np, theta_np)
                    phase = (-1.0) ** m
                    Y_posm = math.sqrt(2.0) * phase * np.real(Y_lm)  # +m (cos-like)
                    Y_negm = math.sqrt(2.0) * phase * np.imag(Y_lm)  # -m (sin-like)
                    enc.append(Y_posm.reshape(-1, 1))
                    enc.append(Y_negm.reshape(-1, 1))
        Y_real = np.concatenate(enc, axis=1)  # [E, S], S=(L+1)^2
        Y_real = torch.from_numpy(Y_real).to(device=device, dtype=dtype)

        F = G.unsqueeze(-1) * Y_real.unsqueeze(1)  # [E,K,S]
        C = rij.new_zeros((N, K, self._S))         # [N,K,S]
        C.index_add_(0, row, F) 

        tri_idx = torch.triu_indices(K, K, offset=0, device=device)
        pieces = []
        for l, s0, s1 in self._sh_slices:
            C_l = C[:, :, s0:s1]                   # [N,K,M_l] with M_l = 2l+1
            P_l = torch.matmul(C_l, C_l.transpose(1, 2))  # [N,K,K]
            P_ut = P_l[:, tri_idx[0], tri_idx[1]]         # [N, K*(K+1)/2]
            pieces.append(P_ut)

        soap = torch.cat(pieces, dim=-1)           # [N, (L+1)*K*(K+1)/2]

        norm = torch.linalg.norm(soap, dim=-1, keepdim=True).clamp_min(self.eps)
        soap = soap / norm

        if self.include_input:
            soap = torch.cat([pos, soap], dim=-1)

        setattr(data, self.attr_name, soap)
        return data



class NoisePosition(BaseTransform):
    def __init__(self, attr_name="positional_encoding"):

        self.attr_name = attr_name

    def __call__(self, data):
        if not hasattr(data, "pos"):
            raise AttributeError("Data.pos is required: [num_nodes, 3]).")
        
        pos = data.pos
        n, d = pos.size()

        std = pos.std(dim=0, keepdim=True)
        noise = torch.randn(n, d) * std
        
        setattr(data, self.attr_name, noise)
        return data

    def __str__(self):
        return f'NoisePosition(scale={self.scale})'


class AddLaplacianPE(AddLaplacianEigenvectorPE):
    def __init__(
        self,
        k: int,
        attr_name: str = "positional_encoding",
        cutoff: float = 5.0,
        max_num_neighbors: int = 64,
    ):
        super().__init__(k=k, attr_name=attr_name, is_undirected=True, normalization="sym")
        self.cutoff = float(cutoff)
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, data):
        N = data.num_nodes
        k = int(self.k)
        if not N or k <= 0:
            return data

        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            pos = getattr(data, "pos", None)
            if pos is None:
                raise ValueError("data.pos not present")
            edge_index = radius_graph(
                pos, r=self.cutoff, loop=False, max_num_neighbors=self.max_num_neighbors
            )
        edge_index = to_undirected(edge_index, num_nodes=N)

        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None:
            pos = getattr(data, "pos", None)
            if pos is None:
                raise ValueError("data.pos not present")
            edge_weight = calc_edge_weight(
                pos=pos, edge_index=edge_index
            )

        ei_L, ew_L = get_laplacian(
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalization="sym",
            num_nodes=N,
        )
        L = to_scipy_sparse_matrix(ei_L, ew_L, N)

        L_dense = L.toarray()
        evals, evecs = np.linalg.eigh(L_dense)   
        eps = 1e-9
        c = int((evals <= eps).sum())
        start = c
        end = min(start + k, N)
        U = evecs[:, start:end]                 

        if U.shape[1] < k:
            U = np.pad(U, ((0, 0), (0, k - U.shape[1])), mode="constant")

        pe = torch.from_numpy(U).to(dtype=torch.float)
        sign = torch.randint(0, 2, (k,)) * 2 - 1  # {0,1}→{-1,1}
        pe = pe * sign
        setattr(data, self.attr_name, pe)

        return data
    

class SOAP_Dsribe(BaseTransform):
    def __init__(self, species, rcut=5.0, nmax=8, lmax=6, sigma=0.5, sparse: str=True, attr_name: str = "positional_encoding"):
        self.soap = SOAP(
            species=species,
            r_cut=rcut,
            n_max=nmax,
            l_max=lmax,
            sigma=sigma,
            sparse=True,
            periodic=False
        )
        self.attr_name = attr_name
        self.species = set(species)

    @torch.no_grad()
    def __call__(self, data):
        pos = data.pos.cpu().numpy()
        Z = data.z.cpu().numpy()

        mask = [i for i, z in enumerate(Z) if z in self.species]
        if len(mask) == 0:
            soap_tensor = torch.zeros((0, self.soap.get_number_of_features()), dtype=torch.float32)
            setattr(data, self.attr_name, soap_tensor)
            return data

        pos_filtered = pos[mask]
        Z_filtered = Z[mask]

        atoms = Atoms(numbers=Z_filtered, positions=pos_filtered)
        desc = self.soap.create(atoms)  
        rows = [torch.tensor(desc[i].todense() if self.soap.sparse else desc[i], dtype=torch.float32)
                for i in range(desc.shape[0])]
        soap_tensor = torch.stack(rows)
        
        

        setattr(data, self.attr_name, soap_tensor)
        return data
    
    
class CoulombMatrixPE(BaseTransform):

    def __init__(self, max_atoms: int = 50, attr_name: str = "positional_encoding", sorting: str = "row_norm"):
        self.max_atoms = max_atoms
        self.attr_name = attr_name
        self.sorting = sorting

    @staticmethod
    def compute_coulomb_matrix(Z: torch.Tensor, R: torch.Tensor):
        N = Z.shape[0]
        C = torch.zeros((N, N), dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                if i == j:
                    C[i, j] = 0.5 * Z[i] ** 2.4  
                else:
                    dist = torch.norm(R[i] - R[j])
                    C[i, j] = Z[i] * Z[j] / dist
        return C

    def __call__(self, data):
        if not hasattr(data, "pos") or not hasattr(data, "z"):
            raise AttributeError("Data must have pos [N,3] and z [N] attributes.")

        Z = data.z
        R = data.pos
        C = self.compute_coulomb_matrix(Z, R)  
        
        if self.sorting == "row_norm":
            row_norms = torch.linalg.norm(C, dim=1)
            idx = torch.argsort(row_norms, descending=True)
            C = C[idx][:, idx]

        N = C.shape[0]
        if N < self.max_atoms:
            pad = self.max_atoms - N
            C = torch.nn.functional.pad(C, (0, pad, 0, pad))
        else:
            C = C[:self.max_atoms, :self.max_atoms]


        triu = torch.triu_indices(self.max_atoms, self.max_atoms)
        vec = C[triu[0], triu[1]].unsqueeze(0) 

        setattr(data, self.attr_name, vec)       
        data.__dict__["_graph_level_pe"] = True  
        return data