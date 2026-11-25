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


def generate_combined_latex_table(df, dataset_filter, split_keys, split_titles, main_caption, columns_spec, header_lines):
    """Generates combined tables for Binary and Spectral without time metrics."""
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
                means = agg[col]['mean'].dropna()
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
                if model in agg.index and col in agg.columns and pd.notna(agg.loc[model, (col, 'mean')]):
                    mean_v = agg.loc[model, (col, 'mean')]
                    std_v = agg.loc[model, (col, 'std')]
                    
                    rank = None
                    if do_rank:
                        b, s = best_vals[col]
                        if b is not None and np.isclose(mean_v, b, atol=1e-10): rank = 1
                        elif s is not None and np.isclose(mean_v, s, atol=1e-10): rank = 2
                    
                    row.append(format_val(mean_v, std_v, scale=scale, rank=rank))
                else:
                    row.append("---")
            lines.append(" & ".join(row) + " \\\\")
            
        lines.extend([
            "\\bottomrule", 
            "\\end{tabular}",
            "}" 
        ])
        
        # FIX: apply space between all subtables, not just after the first one
        if idx < len(split_keys) - 1: 
            lines.append("\\vspace{0.4cm}")
            
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
    """Generates the Regression table stacking eV and log10(f) targets without time metrics."""
    if df.empty: return f"% No data found for: {main_caption}"
    
    columns_spec = [
        ('raw_mae', False, 1.0, True),
        ('MAE', False, 1.0, True),
        ('MSE', False, 1.0, True),
        ('R2', True, 1.0, True)
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
        lines.append("\\begin{tabular}{llcccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{$\\hat{y}$} & \\textbf{Model} & \\textbf{Raw MAE} $\\downarrow$ & \\textbf{MAE} $\\downarrow$ & \\textbf{MSE} $\\downarrow$ & \\textbf{R$^2$} $\\uparrow$ \\\\")
        lines.append("\\midrule")
        
        for t_idx, (t_label, t_dataset) in enumerate(targets):
            df_sub = df[(df['Dataset'] == t_dataset) & (df['Block_Split'] == split_val)].copy()
            agg = pd.DataFrame(columns=numeric_cols)
            best_vals = {col: (None, None) for col in numeric_cols}
            
            if not df_sub.empty:
                df_sub['Model'] = df_sub['Model'].str.lower().map(MODEL_MAP)
                df_sub = df_sub.dropna(subset=['Model'])
                
                # Step 1: Average across States for each Seed
                seed_means = df_sub.groupby(['Model', 'Seed'])[numeric_cols].mean().reset_index()
                
                # Step 2: Calculate Mean and Std across the Seeds
                agg = seed_means.groupby('Model')[numeric_cols].agg(['mean', 'std'])

                for col, higher_is_better, _, do_rank in columns_spec:
                    if not do_rank or col not in agg.columns: continue
                    means = agg[col]['mean'].dropna()
                    if means.empty: continue
                    unique_means = sorted(means.unique(), reverse=higher_is_better)
                    best = unique_means[0] if len(unique_means) > 0 else None
                    second = unique_means[1] if len(unique_means) > 1 else None
                    best_vals[col] = (best, second)
                
            for m_idx, model in enumerate(MODELS):
                row = []
                row.append(t_label if m_idx == 0 else "")
                row.append(model)
                
                for col, higher_is_better, scale, do_rank in columns_spec:
                    if model in agg.index and col in agg.columns and pd.notna(agg.loc[model, (col, 'mean')]):
                        mean_v = agg.loc[model, (col, 'mean')]
                        std_v = agg.loc[model, (col, 'std')]
                        
                        rank = None
                        if do_rank:
                            b, s = best_vals[col]
                            if b is not None and np.isclose(mean_v, b, atol=1e-10): rank = 1
                            elif s is not None and np.isclose(mean_v, s, atol=1e-10): rank = 2
                        
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
        
        # FIX: apply space between all subtables, not just after the first one
        if idx < len(split_keys) - 1: 
            lines.append("\\vspace{0.4cm}")
            
    lines.extend([
        "\\begin{tablenotes}",
        "\\small",
        "\\item Best results are in \\textbf{bold}, second best are \\underline{underlined}.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def save_table(content, target_dir, filename):
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {filepath}")


def process_split_set(df_bin, df_reg, output_subdir, bin_splits, reg_splits, split_titles):
    """Helper to generate and save Binary, Regression, and Spectral tables for a set of splits."""
    
    # 1. Check if the required splits actually exist in the loaded data
    missing_bin = [s for s in bin_splits if s not in df_bin['Block_Split'].unique()] if not df_bin.empty else []
    missing_reg = [s for s in reg_splits if s not in df_reg['Block_Split'].unique()] if not df_reg.empty else []
    
    # If the dataframe isn't empty but is missing required splits, we skip generation
    if (not df_bin.empty and missing_bin) or (not df_reg.empty and missing_reg):
        missing_all = set(missing_bin + missing_reg)
        print(f"\nSkipping '{output_subdir}': missing required splits {list(missing_all)} in the data.")
        return
        
    if df_bin.empty and df_reg.empty:
        print(f"\nSkipping '{output_subdir}': no data available.")
        return

    # 2. Proceed with generation if checks pass
    out_dir = os.path.join(TEX_DIR, output_subdir)
    print(f"\n--- Generating tables for target directory: '{out_dir}' ---")

    # [Binary Table Generation]
    bin_cols = [
        ('F1', True, 1.0, True),
        ('Precision', True, 1.0, True),
        ('Recall', True, 1.0, True),
        ('Accuracy', True, 1.0, True)
    ]
    bin_header = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{F1} $\\uparrow$ & \\textbf{Precision} $\\uparrow$ & \\textbf{Recall} $\\uparrow$ & \\textbf{Accuracy} $\\uparrow$ \\\\",
        "\\midrule"
    ]
    bin_table = generate_combined_latex_table(
        df_bin, "TMQM_SPECTO_BINARY", bin_splits, split_titles, 
        "Binary classification metrics on tmQMg*.", 
        bin_cols, bin_header
    )
    save_table(bin_table, out_dir, "binary_metrics.tex")

    # [Regression Table Generation]
    reg_table = generate_stacked_regression_table(
        df_reg, reg_splits, split_titles, 
        "Regression metrics for 10 eV and $\\log_{10}(f)$ targets on tmQMg*."
    )
    save_table(reg_table, out_dir, "regression_metrics.tex")

    # [Spectral Table Generation]
    spec_cols = [
        ('metric_jsd', False, 1.0, True),
        ('metric_wasserstein', False, 1.0, True),
        ('metric_sid', False, 1.0, True),
        ('metric_stmse', False, 1.0, True),
        ('metric_smse', False, 1e7, True),
        ('metric_srmse', False, 1e4, True)
    ]
    spec_header = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{JSD} $\\downarrow$ & \\textbf{Wasserstein} $\\downarrow$ & \\textbf{SID} $\\downarrow$ & \\textbf{STMSE} $\\downarrow$ & \\textbf{SMSE} ($\\times 10^{-7}$) $\\downarrow$ & \\textbf{SRMSE} ($\\times 10^{-4}$) $\\downarrow$ \\\\",
        "\\midrule"
    ]
    spec_table = generate_combined_latex_table(
        df_reg, "TMQM_SPECTO_SPECTRAL", reg_splits, split_titles, 
        "Spectral reconstruction metrics on tmQMg*.", 
        spec_cols, spec_header
    )
    save_table(spec_table, out_dir, "spectral_metrics.tex")

def main():
    print("Loading data...")
    try:
        df_bin, df_reg = load_data('experiment_results.json')
    except FileNotFoundError:
        print("experiment_results.json not found. Place it in the exact directory.")
        return

    # Define the 3 sets of splits and their sub-titles
    split_titles = [
        "(a) Train \\& Test: Blocks 3, 4, 5",
        "(b) Test: Block 3",
        "(c) Train \\& Test: Block 3"
    ]
    
    bin_splits = ["345", "3test", "none"]
    reg_splits = ["345", "3test", "null"]

    # Generate a single combined set of tables
    process_split_set(
        df_bin, df_reg,
        output_subdir="combined_3_splits",
        bin_splits=bin_splits,
        reg_splits=reg_splits,
        split_titles=split_titles
    )

    print("\nComplete! Processed all tables into the target directory.")

if __name__ == "__main__":
    main()