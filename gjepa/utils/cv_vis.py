
# custom_chemical_dataset.py
from __future__ import annotations
import os
import json
import ast
import hashlib
from typing import List, Optional, Sequence, Union, Any, Dict

import torch
from torch import Tensor
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data

_SYMBOL2Z = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,"Al":13,
    "Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,
    "Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Rb":37,
    "Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,
    "Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,"Pr":59,"Nd":60,"Pm":61,
    "Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,
    "W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,"Bi":83,"Po":84,"At":85,
    "Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,"Pa":91,"U":92,"Np":93,"Pu":94,"Am":95,"Cm":96,"Bk":97,
    "Cf":98,"Es":99,"Fm":100,"Md":101,"No":102,"Lr":103,"Rf":104,"Db":105,"Sg":106,"Bh":107,"Hs":108,
    "Mt":109,"Ds":110,"Rg":111,"Cn":112,"Nh":113,"Fl":114,"Mc":115,"Lv":116,"Ts":117,"Og":118
}

def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def _to_list(s: Any) -> List:
    """Parse list-like input safely from JSON, Python literal, or CSV-ish strings."""
    if isinstance(s, list):
        return s
    if isinstance(s, tuple):
        return list(s)
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    if isinstance(s, str):
        txt = s.strip()
        if not txt:
            return []
        # JSON first
        try:
            v = json.loads(txt)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            pass
        # Python literal
        try:
            v = ast.literal_eval(txt)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            pass
        # Semicolon-separated triples like: "x y z; x y z; ..."
        if ";" in txt and any(c.isdigit() for c in txt):
            parts = [p.strip() for p in txt.split(";") if p.strip()]
            triples = []
            for p in parts:
                nums = [float(t) for t in p.replace(",", " ").split()]
                triples.extend(nums)
            return triples
        # Fallback comma/space separated flat list
        sep = "," if "," in txt else " "
        return [t for t in (x.strip() for x in txt.split(sep)) if t]
    # numpy/pandas scalars or arrays
    try:
        return list(s)
    except Exception:
        return [s]

def _parse_coords(val: Any, expected_n: Optional[int] = None) -> Tensor:
    """
    Convert coords to (N,3) float32. Accepts:
    - nested list [[x,y,z], ...]
    - flat list [x1,y1,z1, x2,y2,z2, ...]
    - "x y z; x y z; ..." or whitespace-separated flat numbers
    """
    lst = _to_list(val)
    # If nested already
    if len(lst) > 0 and isinstance(lst[0], (list, tuple)):
        coords = [[float(c) for c in row] for row in lst]
        pos = torch.tensor(coords, dtype=torch.float32)
    else:
        # flat list of numbers or strings
        flat = [float(x) for x in lst]
        if len(flat) % 3 != 0:
            raise ValueError(f"atom_coords length {len(flat)} is not divisible by 3")
        if expected_n is not None and len(flat) // 3 != int(expected_n):
            # allow but warn via exception path—better to fail fast in processing
            raise ValueError(f"num_atoms={expected_n} but coords imply {len(flat)//3} atoms")
        pos = torch.tensor([flat[i:i+3] for i in range(0, len(flat), 3)], dtype=torch.float32)

    if pos.ndim != 2 or pos.size(-1) != 3:
        raise ValueError(f"pos must be (N,3), got {tuple(pos.shape)}")
    return pos



def _parse_atom_types(val: Any) -> Tensor:

    lst = _to_list(val)
    if not lst:
        return torch.empty(0, dtype=torch.long)

    if all(_is_number(x) for x in lst):
        return torch.tensor([int(float(x)) for x in lst], dtype=torch.long)

    z_list = []
    for s in lst:
        sym = str(s).strip()
        if sym == "":
            continue
        if sym not in _SYMBOL2Z:
            raise ValueError(f"Unknown element symbol '{sym}' in atom_types")
        z_list.append(_SYMBOL2Z[sym])
    return torch.tensor(z_list, dtype=torch.long)

def _to_float_vec(vals: Sequence[Any]) -> Tensor:
    return torch.tensor([float(v) for v in vals], dtype=torch.float32)