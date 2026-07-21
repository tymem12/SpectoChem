import os
import json
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
MODELS = ["SchNet", "GCN", "GAT", "GINE", "Null Model"]
MODEL_MAP = {"dummy": "Null Model", "schnet": "SchNet", "gcn": "GCN", "gat": "GAT", "gine": "GINE"}
TEX_DIR = "TEX_DIR"

def format_val(mean, std, scale=1.0, rank=None):
    """Formats Mean and Std Dev with optional LaTeX styling for ranking."""
    if pd.isna(mean):
        return "---"
    m = mean * scale
    s = std * scale if pd.notna(std) else 0.0
    val_str = f"{m:.3f} $\\pm$ {s:.3f}"
    
    if rank == 1:
        return f"\\textbf{{{val_str}}}"
    elif rank == 2:
        return f"\\underline{{{val_str}}}"
    return val_str

def format_time_with_std(mean_sec: float, std_sec: float, dataset: str = "", col_name: str = "") -> str:
    """Custom time formatting featuring subseconds for binary/regression and full seconds for spectral."""
    if pd.isna(mean_sec): return "---"
    
    is_test = "test" in col_name.lower() or col_name == "time_seconds"
    ds_lower = dataset.lower()
    
    # Configure granularity based on table type
    if "binary" in ds_lower or "regression" in ds_lower:
        sec_decimals = 1  # Seconds and subseconds (1 decimal place)
    elif "spectral" in ds_lower:
        sec_decimals = 0 if not is_test else 1  # Train: show seconds too (no decimals); Test: 1 decimal
    else:
        sec_decimals = 0

    def sec_to_str(val: float) -> str:
        if pd.isna(val) or val == 0: return "0s"
            
        d = int(val // 86400)
        rem = val % 86400
        h = int(rem // 3600)
        rem = rem % 3600
        m = int(rem // 60)
        s = rem % 60
        
        parts = []
        if d > 0: parts.append(f"{d}d")
        if h > 0: parts.append(f"{h}h")
        if m > 0: parts.append(f"{m}m")
        
        if sec_decimals > 0:
            s_str = f"{s:.{sec_decimals}f}"
            parts.append(f"{s_str}s")
        else:
            parts.append(f"{int(round(s))}s")
            
        return "".join(parts) if parts else "0s"

    return f"{sec_to_str(mean_sec)} $\\pm$ {sec_to_str(std_sec)}" if pd.notna(std_sec) and std_sec != 0 else sec_to_str(mean_sec)


def load_data(json_path="experiment_results.json"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df_bin = pd.DataFrame(data.get("binary_results", []))
    df_reg = pd.DataFrame(data.get("regression_results", []))
    
    if not df_bin.empty:
        df_bin['Block_Split'] = df_bin['Block_Split'].replace({None: "null", np.nan: "null"}).astype(str)
    if not df_reg.empty:
        df_reg['Block_Split'] = df_reg['Block_Split'].replace({None: "null", np.nan: "null"}).astype(str)
        
    return df_bin, df_reg


def generate_combined_latex_table(df, dataset_filter, split_keys, split_titles, main_caption, columns_spec, header_lines, dataset_for_time):
    """Generates combined tables for Binary and Spectral."""
    if df.empty: return f"% No data found for: {main_caption}"
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{main_caption}}}",
        "\\begin{threeparttable}"
    ]
    
    numeric_cols = [col for col, _, _, _ in columns_spec]
    
    for idx, split_val in enumerate(split_keys):
        df_sub = df[(df['Dataset'] == dataset_filter) & (df['Block_Split'] == split_val)].copy()
        
        agg = pd.DataFrame(columns=numeric_cols)
        best_vals = {col: (None, None) for col in numeric_cols}
        
        if not df_sub.empty:
            df_sub['Model'] = df_sub['Model'].str.lower().map(MODEL_MAP)
            df_sub = df_sub.dropna(subset=['Model'])
            agg = df_sub.groupby('Model')[numeric_cols].agg(['mean', 'std'])
            
            for col, higher_is_better, _, do_rank in columns_spec:
                if not do_rank or col not in agg.columns: continue
                # Exclude Null Model from the time ranking pool
                means = agg[col]['mean'].drop(labels=['Null Model'], errors='ignore').dropna()
                if means.empty: continue
                unique_means = sorted(means.unique(), reverse=higher_is_better)
                best = unique_means[0] if len(unique_means) > 0 else None
                second = unique_means[1] if len(unique_means) > 1 else None
                best_vals[col] = (best, second)
                
        lines.append(f"\\vspace{{0.2cm}}")
        lines.append(f"\\centerline{{\\textbf{{{split_titles[idx]}}}}}")
        lines.append(f"\\vspace{{0.1cm}}")
        lines.append("\\resizebox{\\textwidth}{!}{")
        lines.extend(header_lines)
        
        for model in MODELS:
            row = [model]
            for col, higher_is_better, scale, do_rank in columns_spec:
                is_time_col = "time" in col.lower()
                
                if model == "Null Model" and is_time_col:
                    row.append("---")
                elif model in agg.index and col in agg.columns and pd.notna(agg.loc[model, (col, 'mean')]):
                    mean_v = agg.loc[model, (col, 'mean')]
                    std_v = agg.loc[model, (col, 'std')]
                    
                    rank = None
                    if do_rank:
                        b, s = best_vals[col]
                        if b is not None and np.isclose(mean_v, b, atol=1e-10): rank = 1
                        elif s is not None and np.isclose(mean_v, s, atol=1e-10): rank = 2
                    
                    if is_time_col:
                        time_str = format_time_with_std(mean_v, std_v, dataset=dataset_for_time, col_name=col)
                        if rank == 1: row.append(f"\\textbf{{{time_str}}}")
                        elif rank == 2: row.append(f"\\underline{{{time_str}}}")
                        else: row.append(time_str)
                    else:
                        row.append(format_val(mean_v, std_v, scale=scale, rank=rank))
                else:
                    row.append("---")
            lines.append(" & ".join(row) + " \\\\")
            
        lines.extend([
            "\\bottomrule", 
            "\\end{tabular}",
            "}" 
        ])
        
        if idx == 0: lines.append("\\vspace{0.4cm}")
            
    lines.extend([
        "\\begin{tablenotes}",
        "\\small",
        "\\item Best results are in \\textbf{bold}, second best are \\underline{underlined}.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_stacked_regression_table(df, split_keys, split_titles, main_caption):
    """Generates the Regression table stacking eV and log10(f) targets as clean rows without multirow."""
    if df.empty: return f"% No data found for: {main_caption}"
    
    columns_spec = [
        ('raw_mae', False, 1.0, True),
        ('MAE', False, 1.0, True),
        ('MSE', False, 1.0, True),
        ('R2', True, 1.0, True),
        ('train_time_seconds', False, 1.0, True),
        ('time_seconds', False, 1.0, True)
    ]
    numeric_cols = [c[0] for c in columns_spec]

    targets = [
        ("eV", "TMQM_SPECTO_LAMBDA_REGRESSOR"),
        ("$\\log_{10}(f)$", "TMQM_SPECTO_F_REGRESSOR")
    ]
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{main_caption}}}",
        "\\begin{threeparttable}"
    ]
    
    for idx, split_val in enumerate(split_keys):
        lines.append(f"\\vspace{{0.2cm}}")
        lines.append(f"\\centerline{{\\textbf{{{split_titles[idx]}}}}}")
        lines.append(f"\\vspace{{0.1cm}}")
        lines.append("\\resizebox{\\textwidth}{!}{")
        lines.append("\\begin{tabular}{llcccccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{$\\hat{y}$} & \\textbf{Model} & \\textbf{Raw MAE} $\\downarrow$ & \\textbf{MAE} $\\downarrow$ & \\textbf{MSE} $\\downarrow$ & \\textbf{R$^2$} $\\uparrow$ & \\textbf{Train Time} $\\downarrow$ & \\textbf{Test Time} $\\downarrow$ \\\\")
        lines.append("\\midrule")
        
        for t_idx, (t_label, t_dataset) in enumerate(targets):
            df_sub = df[(df['Dataset'] == t_dataset) & (df['Block_Split'] == split_val)].copy()
            agg = pd.DataFrame(columns=numeric_cols)
            best_vals = {col: (None, None) for col in numeric_cols}
            
            if not df_sub.empty:
                df_sub['Model'] = df_sub['Model'].str.lower().map(MODEL_MAP)
                df_sub = df_sub.dropna(subset=['Model'])
                agg = df_sub.groupby('Model')[numeric_cols].agg(['mean', 'std'])
                
                for col, higher_is_better, _, do_rank in columns_spec:
                    if not do_rank or col not in agg.columns: continue
                    means = agg[col]['mean'].drop(labels=['Null Model'], errors='ignore').dropna()
                    if means.empty: continue
                    unique_means = sorted(means.unique(), reverse=higher_is_better)
                    best = unique_means[0] if len(unique_means) > 0 else None
                    second = unique_means[1] if len(unique_means) > 1 else None
                    best_vals[col] = (best, second)

            for m_idx, model in enumerate(MODELS):
                row = []
                # Replaced multirow logic entirely with a simple first-row label placement
                row.append(t_label if m_idx == 0 else "")
                row.append(model)
                
                for col, higher_is_better, scale, do_rank in columns_spec:
                    is_time_col = "time" in col.lower()
                    
                    if model == "Null Model" and is_time_col:
                        row.append("---")
                    elif model in agg.index and col in agg.columns and pd.notna(agg.loc[model, (col, 'mean')]):
                        mean_v = agg.loc[model, (col, 'mean')]
                        std_v = agg.loc[model, (col, 'std')]
                        
                        rank = None
                        if do_rank:
                            b, s = best_vals[col]
                            if b is not None and np.isclose(mean_v, b, atol=1e-10): rank = 1
                            elif s is not None and np.isclose(mean_v, s, atol=1e-10): rank = 2
                        
                        if is_time_col:
                            time_str = format_time_with_std(mean_v, std_v, dataset="tmqmg_regression", col_name=col)
                            if rank == 1: row.append(f"\\textbf{{{time_str}}}")
                            elif rank == 2: row.append(f"\\underline{{{time_str}}}")
                            else: row.append(time_str)
                        else:
                            row.append(format_val(mean_v, std_v, scale=scale, rank=rank))
                    else:
                        row.append("---")
                lines.append(" & ".join(row) + " \\\\")
            
            if t_idx < len(targets) - 1:
                lines.append("\\midrule")
                
        lines.extend([
            "\\bottomrule", 
            "\\end{tabular}",
            "}" 
        ])
        
        if idx == 0: lines.append("\\vspace{0.4cm}")
            
    lines.extend([
        "\\begin{tablenotes}",
        "\\small",
        "\\item Best results are in \\textbf{bold}, second best are \\underline{underlined}.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def save_table(content, filename):
    os.makedirs(TEX_DIR, exist_ok=True)
    filepath = os.path.join(TEX_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {filepath}")

def main():
    print("Loading data...")
    try:
        df_bin, df_reg = load_data('experiment_results.json')
    except FileNotFoundError:
        print("experiment_results.json not found. Place it in the exact directory.")
        return

    # 1. BINARY SPECIFICATIONS (Ranking True for times, Arrow added to header)
    bin_cols = [
        ('F1', True, 1.0, True),
        ('Precision', True, 1.0, True),
        ('Recall', True, 1.0, True),
        ('Accuracy', True, 1.0, True),
        ('train_time_seconds', False, 1.0, True),  
        ('time_seconds', False, 1.0, True) 
    ]
    bin_header = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{F1} $\\uparrow$ & \\textbf{Precision} $\\uparrow$ & \\textbf{Recall} $\\uparrow$ & \\textbf{Accuracy} $\\uparrow$ & \\textbf{Train Time} $\\downarrow$ & \\textbf{Test Time} $\\downarrow$ \\\\",
        "\\midrule"
    ]
    bin_splits = ["none", "test"]

    # 3. SPECTRAL SPECIFICATIONS (Ranking True for times, Arrow added to header)
    spec_cols = [
        ('metric_jsd', False, 1.0, True),
        ('metric_wasserstein', False, 1.0, True),
        ('metric_sid', False, 1.0, True),
        ('metric_stmse', False, 1.0, True),
        ('metric_smse', False, 1e7, True),
        ('metric_srmse', False, 1e4, True),
        ('train_time_seconds', False, 1.0, True),
        ('test_time_seconds', False, 1.0, True)
    ]
    spec_header = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{JSD} $\\downarrow$ & \\textbf{Wasserstein} $\\downarrow$ & \\textbf{SID} $\\downarrow$ & \\textbf{STMSE} $\\downarrow$ & \\textbf{SMSE} ($\\times 10^{-7}$) $\\downarrow$ & \\textbf{SRMSE} ($\\times 10^{-4}$) $\\downarrow$ & \\textbf{Train Time} $\\downarrow$ & \\textbf{Test Time} $\\downarrow$ \\\\",
        "\\midrule"
    ]
    spectral_splits = ["null", "test"]
    split_titles = ["(a) Train \\& Test: Block 3", "(b) Test Only: Block 3"]

    print("\nGenerating and saving 3 combined tables...")

    # 1. Binary Table
    bin_table = generate_combined_latex_table(
        df_bin, "TMQM_SPECTO_BINARY", bin_splits, split_titles, 
        "Binary classification metrics on tmQMg*.", 
        bin_cols, bin_header, dataset_for_time="tmqmg_binary"
    )
    save_table(bin_table, "binary_metrics.tex")

    # 2. Regression Table (Stacked & updated parameters)
    reg_table = generate_stacked_regression_table(
        df_reg, spectral_splits, split_titles, 
        "Regression metrics for 10 eV and $\\log_{10}(f)$ targets on tmQMg*."
    )
    save_table(reg_table, "regression_metrics.tex")

    # 3. Spectral Table
    spec_table = generate_combined_latex_table(
        df_reg, "TMQM_SPECTO_SPECTRAL", spectral_splits, split_titles, 
        "Spectral reconstruction metrics on tmQMg*.", 
        spec_cols, spec_header, dataset_for_time="tmqmg_spectral"
    )
    save_table(spec_table, "spectral_metrics.tex")

    print("\nComplete! Processed all 3 combined structural files into 'TEX_DIR'.")

if __name__ == "__main__":
    main()