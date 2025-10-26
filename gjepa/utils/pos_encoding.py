import torch
from tqdm import tqdm

def save_pe_from_transformed_dataset(dataset, out_path,
                                     field_name="positional_encoding",
                                     dtype=torch.float32):
    slices = [0]
    chunks = []
    dim = None

    for i in tqdm(range(len(dataset))):
        d = dataset[i]
        if not hasattr(d, field_name):
            raise RuntimeError(f"Sample {i} does not have'{field_name}' field.")

        pe = getattr(d, field_name).to(dtype=dtype, device="cpu")
        if dim is None:
            dim = pe.size(1)
        elif pe.size(1) != dim:
            raise RuntimeError(f"Inconsitent PE dimension in sample {i}: {pe.size(1)} != {dim}")

        chunks.append(pe)
        slices.append(slices[-1] + pe.size(0))

    cat = torch.cat(chunks, dim=0) 
    blob = {
        "cat": cat,  
        "slices": torch.tensor(slices, dtype=torch.long),
        "meta": {
            "num_graphs": len(dataset),
            "dim": dim,
            "field": field_name,
            "dtype": str(dtype),
        },
    }
    torch.save(blob, out_path)
    print(f"Saved {out_path}: cat={tuple(cat.shape)}, graphs={len(dataset)}, dim={dim}")



def attach_pe_to_dataset_inplace(dataset, pe_path, field_name="positional_encoding",
                                 strict=True, map_location="cpu"):

    blob = torch.load(pe_path, map_location=map_location)
    cat = blob["cat"]                 
    pe_slices = blob["slices"].long()
    n_graphs = pe_slices.numel() - 1

    if strict and n_graphs != len(dataset):
        raise ValueError(f"PE graphs ({n_graphs}) != dataset graphs ({len(dataset)})")

    if strict:
        base_key = "pos" if "pos" in dataset.slices else ("x" if "x" in dataset.slices else None)
        if base_key is not None:
            ds_slices = dataset.slices[base_key].cpu().long()
            nodes_ds = ds_slices[1:] - ds_slices[:-1]
            nodes_pe = (pe_slices[1:] - pe_slices[:-1]).cpu().long()
            if not torch.equal(nodes_ds, nodes_pe):
                raise ValueError("Inconsistent number of"
                                 "nodes per graph between dataset and PE slices.")

    data_list = []
    for i in range(len(dataset)):
        s = int(pe_slices[i])
        e = int(pe_slices[i + 1])
        pe = cat[s:e]
        d = dataset.get(i)
        if strict and hasattr(d, "num_nodes") and d.num_nodes != pe.size(0):
            raise ValueError(f"idx={i}: data.num_nodes={d.num_nodes} != pe={pe.size(0)}")
        setattr(d, field_name, pe) 
        data_list.append(d)

    dataset._data_list = data_list
    dataset._len = len(data_list)
    if hasattr(dataset, "_indices"):
        dataset._indices = None

    return dataset
