from __future__ import annotations
import os
import json
import ast
import hashlib
from typing import List, Optional, Sequence, Union, Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch
from torch import Tensor
from gjepa.datasets.cv_vis.standarizer_singleton import StandarizerSingletonF, StandarizerSingletonLambda

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





def _plot_pairs_graph_with_predictions(
    data: pd.DataFrame,
    save_path: str,
    range_x: Tuple[float, float],
    num_samples: int = 50
) -> None:
    out_dir = os.path.join(save_path, "saved_plots")
    os.makedirs(out_dir, exist_ok=True)

    target_cols = [c for c in data.columns if c.startswith("target_")]
    prediction_cols = [c for c in data.columns if c.startswith("prediction_")]

    def _sort_key(col_name: str) -> int:
        return int(col_name.split("_")[1])

    target_cols = sorted(target_cols, key=_sort_key)
    prediction_cols = sorted(prediction_cols, key=_sort_key)

    num_targets = len(target_cols)
    num_preds = len(prediction_cols)

    if num_preds != num_targets:
        raise ValueError()

    if num_targets % 2 != 0:
        raise ValueError()
    
    standarized_lambda_val = StandarizerSingletonLambda.get_values()
    standarized_f_val = StandarizerSingletonF.get_values()

    num_pairs = num_targets // 2
    num_samples = min(len(data), num_samples)
    for _, row in list(data.iterrows())[:num_samples]:
        origin_id = row["origin_id"]

        x_targets = []
        y_targets = []
        x_preds = []
        y_preds = []

        for pair_idx in range(num_pairs):
            even_idx = 2 * pair_idx
            odd_idx = 2 * pair_idx + 1

            t_x = row[target_cols[even_idx]]
            t_y = row[target_cols[odd_idx]]
            p_x = row[prediction_cols[even_idx]]
            p_y = row[prediction_cols[odd_idx]]
            if standarized_f_val['standarize']:
                t_y = (t_y * standarized_f_val['std_f']) + standarized_f_val['mean_f']
                p_y = (p_y * standarized_f_val['std_f']) + standarized_f_val['mean_f']
            
            if standarized_lambda_val['standarize']:
                t_x = (t_x * standarized_lambda_val['std_lambda']) + standarized_lambda_val['mean_lambda']
                p_x = (p_x * standarized_lambda_val['std_lambda']) + standarized_lambda_val['mean_lambda']


            x_targets.append(t_x)
            y_targets.append(t_y)
            x_preds.append(p_x)
            y_preds.append(p_y)

        if len(x_targets) == 0:
            continue

        plt.figure(figsize=(8, 4))
        for i, (x, y) in enumerate(zip(x_targets, y_targets)):
            plt.vlines(
                x,
                0,
                y,
                color="red",
                alpha=0.7,
                linewidth=2,
                label="target" if i == 0 else None,
            )

        for i, (x, y) in enumerate(zip(x_preds, y_preds)):
            plt.vlines(
                x,
                0,
                y,
                color="blue",
                alpha=0.7,
                linewidth=2,
                label="prediction" if i == 0 else None,
            )

        x_min, x_max = range_x
        plt.xlim(x_min, x_max)

        all_y = np.array(y_targets + y_preds)
        y_max = float(all_y.max()) if all_y.size > 0 else 1.0
        plt.ylim(0, y_max * 1.05)

        plt.xlabel("x")
        plt.ylabel("value (y)")
        plt.title(f"Pairs plot for {origin_id}")
        plt.legend()

        filename = f"{origin_id}.png"
        filepath = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        print("saves to:", filepath)


def _plot_lambda_binary_predictions(
    data: pd.DataFrame,
    save_path: str,
    range_x: Tuple[float, float],
    num_samples: int = 50
) -> None:
    out_dir = os.path.join(save_path, "saved_plots")
    os.makedirs(out_dir, exist_ok=True)

    target_cols = [c for c in data.columns if c.startswith("target_")]
    prediction_cols = [c for c in data.columns if c.startswith("prediction_")]

    def _sort_key(col_name: str) -> int:
        return int(col_name.split("_")[1])

    target_cols = sorted(target_cols, key=_sort_key)
    prediction_cols = sorted(prediction_cols, key=_sort_key)

    num_targets = len(target_cols)
    num_preds = len(prediction_cols)
    standarized_lambda_val = StandarizerSingletonLambda.get_values()

    if num_preds != num_targets:
        raise ValueError()

    num_samples = min(len(data), num_samples)
    for _, row in list(data.iterrows())[:num_samples]:
        origin_id = row["origin_id"]

        x_targets = []
        y_targets = []
        x_preds = []
        y_preds = []

        for pair_idx in range(num_targets):

            t_x = row[target_cols[pair_idx]]
            p_x = row[prediction_cols[pair_idx]]

            if standarized_lambda_val['standarize']:
                t_x = (t_x * standarized_lambda_val['std_lambda']) + standarized_lambda_val['mean_lambda']
                p_x = (p_x * standarized_lambda_val['std_lambda']) + standarized_lambda_val['mean_lambda']

            x_targets.append(t_x)
            y_targets.append(1)
            x_preds.append(p_x)
            y_preds.append(1)

        if len(x_targets) == 0:
            continue

        plt.figure(figsize=(8, 4))
        for i, (x, y) in enumerate(zip(x_targets, y_targets)):
            plt.vlines(
                x,
                0,
                y,
                color="red",
                alpha=0.7,
                linewidth=2,
                label="target" if i == 0 else None,
            )

        for i, (x, y) in enumerate(zip(x_preds, y_preds)):
            plt.vlines(
                x,
                0,
                y,
                color="blue",
                alpha=0.7,
                linewidth=2,
                label="prediction" if i == 0 else None,
            )

        x_min, x_max = range_x
        plt.xlim(x_min, x_max)

        all_y = np.array(y_targets + y_preds)
        y_max = float(all_y.max()) if all_y.size > 0 else 1.0
        plt.ylim(0, y_max * 1.05)

        plt.xlabel("x")
        plt.ylabel("value (y)")
        plt.title(f"Pairs plot for {origin_id}")
        plt.legend()

        filename = f"{origin_id}.png"
        filepath = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        print("saves to:", filepath)

def _plot_vector_graph_with_predictions(
    data: pd.DataFrame,
    save_path: str,
    range_x: Tuple[float, float],
    num_samples: int = 50
) -> None:
    out_dir = os.path.join(save_path, "saved_plots")
    os.makedirs(out_dir, exist_ok=True)

    target_cols = [c for c in data.columns if c.startswith("target_")]
    prediction_cols = [c for c in data.columns if c.startswith("prediction_")]

    def _sort_key(col_name: str) -> int:
        return int(col_name.split("_")[1])

    target_cols = sorted(target_cols, key=_sort_key)
    prediction_cols = sorted(prediction_cols, key=_sort_key)

    num_targets = len(target_cols)
    num_preds = len(prediction_cols)

    if num_preds != num_targets:
        raise ValueError()

    x_min, x_max = range_x
    x_values = np.linspace(x_min, x_max, num_targets)
    num_samples = min(len(data), num_samples)
    for _, row in list(data.iterrows())[:num_samples]:
        origin_id = row["origin_id"]

        y_targets = []
        y_preds = []

        for t_col, p_col in zip(target_cols, prediction_cols):
            t_y = row[t_col]
            p_y = row[p_col]

            if pd.isna(t_y) or pd.isna(p_y):
                y_targets.append(np.nan)
                y_preds.append(np.nan)
            else:
                y_targets.append(t_y)
                y_preds.append(p_y)

        y_targets = np.array(y_targets, dtype=float)
        y_preds = np.array(y_preds, dtype=float)

        plt.figure(figsize=(8, 4))
        first_target_drawn = False
        for x, y in zip(x_values, y_targets):
            if np.isnan(y):
                continue
            plt.vlines(
                x,
                0,
                y,
                color="red",
                alpha=0.7,
                linewidth=1.5,
                label="target" if not first_target_drawn else None,
            )
            first_target_drawn = True

        first_pred_drawn = False
        for x, y in zip(x_values, y_preds):
            if np.isnan(y):
                continue
            plt.vlines(
                x,
                0,
                y,
                color="blue",
                alpha=0.7,
                linewidth=1.5,
                label="prediction" if not first_pred_drawn else None,
            )
            first_pred_drawn = True

        plt.xlim(x_min, x_max)

        all_y = np.concatenate(
            [y_targets[~np.isnan(y_targets)], y_preds[~np.isnan(y_preds)]]
        )
        if all_y.size > 0:
            y_max = float(all_y.max())
        else:
            y_max = 1.0
        if y_max <= 0:
            y_max = 1.0
        plt.ylim(0, y_max * 1.05)

        plt.xlabel("x")
        plt.ylabel("value (y)")
        plt.title(f"Vector plot for {origin_id}")
        plt.legend()

        filename = f"{origin_id}_vector.png"
        filepath = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        print("saves to:", filepath)


def _plot_binary_classification(data: pd.DataFrame, save_path: str) -> None:
    out_dir = os.path.join(save_path, "saved_plots")
    os.makedirs(out_dir, exist_ok=True)

    if 'target_0' in data.columns:
        targets = data['target_0']
    else:
        targets = data['taget_0'] 
    
    predictions = data['prediction_0']

    probs = 1.0 / (1.0 + np.exp(-predictions.values))  # sigmoid
    y_pred = (probs >= 0.5).astype(int)
    y_true = targets.values.astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)

    fig_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(y_true, y_pred, digits=4)
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)




def plot_graph_with_predictions(data: pd.DataFrame, output_type: str, save_path: str, range: tuple):
    if output_type == 'vector':
        _plot_vector_graph_with_predictions(data, save_path, range)

    elif output_type == 'pairs':
        _plot_pairs_graph_with_predictions(data, save_path, range)

    elif output_type == 'lambda_binary':
        _plot_lambda_binary_predictions(data, save_path, range)
    elif output_type == 'binary_classification':
        _plot_binary_classification(data, save_path)

