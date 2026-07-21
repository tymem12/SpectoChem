import os
import json
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
MODELS = ["Null Model", "SchNet", "GCN", "GAT", "GINE"]
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
    
    # Ensure splits are parsed as strings to avoid Python 'None' breaking filters
    if not df_bin.empty:
        df_bin['Block_Split'] = df_bin['Block_Split'].replace({None: "null", np.nan: "null"}).astype(str)
    if not df_reg.empty:
        df_reg['Block_Split'] = df_reg['Block_Split'].replace({None: "null", np.nan: "null"}).astype(str)
        
    return df_bin, df_reg

def generate_latex_table(df, dataset_filter, split_val, title, columns_spec, header_lines):
    """
    columns_spec: list of tuples (column_name, higher_is_better, scale_factor, do_rank)
    """
    if df.empty:
        return f"% No data found to generate: {title}"
        
    df_sub = df[(df['Dataset'] == dataset_filter) & (df['Block_Split'] == split_val)].copy()
    if df_sub.empty:
        return f"% No specific data matched the filter for: {title}"
        
    df_sub['Model'] = df_sub['Model'].str.lower().map(MODEL_MAP)
    df_sub = df_sub.dropna(subset=['Model'])
    
    # Extract only the numeric columns needed for this table to prevent string aggregation errors
    numeric_cols = [col for col, _, _, _ in columns_spec]
    
    # Aggregate only the specified numeric columns
    agg = df_sub.groupby('Model')[numeric_cols].agg(['mean', 'std'])
    
    # Identify 1st and 2nd best values per column
    best_vals = {}
    for col, higher_is_better, scale, do_rank in columns_spec:
        if not do_rank or col not in agg.columns:
            best_vals[col] = (None, None)
            continue
        
        means = agg[col]['mean'].dropna()
        if means.empty:
            best_vals[col] = (None, None)
            continue
        
        unique_means = sorted(means.unique(), reverse=higher_is_better)
        best = unique_means[0] if len(unique_means) > 0 else None
        second = unique_means[1] if len(unique_means) > 1 else None
        best_vals[col] = (best, second)
        
    # Build LaTeX 
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{threeparttable}",
        f"\\caption{{{title}}}",
    ]
    lines.extend(header_lines)
    
    for model in MODELS:
        row = [model]
        if model in agg.index:
            for col, higher_is_better, scale, do_rank in columns_spec:
                if col in agg.columns and pd.notna(agg.loc[model, (col, 'mean')]):
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
        else:
            row.extend(["---"] * len(columns_spec))
            
        lines.append(" & ".join(row) + " \\\\")
        
    lines.extend([
        "\\bottomrule", 
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Best results are in \\textbf{bold}, second best are \\underline{underlined}.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "}",
        "\\end{table}"
    ])
    return "\n".join(lines)

def save_table(content, filename):
    """Saves the LaTeX table string to a file in TEX_DIR."""
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

    # 1. BINARY SPECIFICATIONS
    bin_cols = [
        ('F1', True, 1.0, True),
        ('Precision', True, 1.0, True),
        ('Recall', True, 1.0, True),
        ('Accuracy', True, 1.0, True),
        ('train_time_seconds', False, 1.0, False),  
        ('time_seconds', False, 1.0, False)
    ]
    bin_header = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{F1} $\\uparrow$ & \\textbf{Precision} $\\uparrow$ & \\textbf{Recall} $\\uparrow$ & \\textbf{Accuracy} $\\uparrow$ & \\textbf{Train Time (s)} & \\textbf{Test Time (s)} \\\\",
        "\\midrule"
    ]

    # 2. REGRESSION SPECIFICATIONS
    reg_cols = [
        ('Lambda_raw_mae', False, 1.0, True),
        ('Lambda_test_MAE', False, 1.0, True),
        ('Lambda_test_MSE', False, 1.0, True),
        ('Lambda_test_R2', True, 1.0, True),
        ('F_raw_mae', False, 1.0, True),
        ('F_test_MAE', False, 1.0, True),
        ('F_test_MSE', False, 1.0, True),
        ('F_test_R2', True, 1.0, True),
        ('train_time_seconds', False, 1.0, False),
        ('test_time_seconds', False, 1.0, False)
    ]
    reg_header = [
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        " & \\multicolumn{4}{c}{\\textbf{eV}} & \\multicolumn{4}{c}{\\textbf{$\\log_{10}(f)$}} & \\multicolumn{2}{c}{\\textbf{Time (s)}} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-11}",
        "\\textbf{Model} & \\textbf{Raw MAE} $\\downarrow$ & \\textbf{MAE} $\\downarrow$ & \\textbf{MSE} $\\downarrow$ & \\textbf{R$^2$} $\\uparrow$ & \\textbf{Raw MAE} $\\downarrow$ & \\textbf{MAE} $\\downarrow$ & \\textbf{MSE} $\\downarrow$ & \\textbf{R$^2$} $\\uparrow$ & \\textbf{Train} & \\textbf{Test} \\\\",
        "\\midrule"
    ]

    # 3. SPECTRAL SPECIFICATIONS
    spec_cols = [
        ('metric_jsd', False, 1.0, True),
        ('metric_wasserstein', False, 1.0, True),
        ('metric_sid', False, 1.0, True),
        ('metric_stmse', False, 1.0, True),
        ('metric_smse', False, 1e7, True),
        ('metric_srmse', False, 1e4, True),
        ('train_time_seconds', False, 1.0, False),
        ('test_time_seconds', False, 1.0, False)
    ]
    spec_header = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{JSD} $\\downarrow$ & \\textbf{Wasserstein} $\\downarrow$ & \\textbf{SID} $\\downarrow$ & \\textbf{STMSE} $\\downarrow$ & \\textbf{SMSE} ($\\times 10^{-7}$) $\\downarrow$ & \\textbf{SRMSE} ($\\times 10^{-4}$) $\\downarrow$ & \\textbf{Train Time (s)} & \\textbf{Test Time (s)} \\\\",
        "\\midrule"
    ]

    print("\nGenerating and saving tables...")

    # Binary Tables
    bin_train = generate_latex_table(df_bin, "TMQM_SPECTO_BINARY", "none", "Binary classification metrics on tmQMg* (Train \\& Test: Block 3).", bin_cols, bin_header)
    save_table(bin_train, "binary_train_test_block3.tex")

    bin_test = generate_latex_table(df_bin, "TMQM_SPECTO_BINARY", "test", "Binary classification metrics on tmQMg* (Test Only: Block 3).", bin_cols, bin_header)
    save_table(bin_test, "binary_test_only_block3.tex")

    # Regression Tables
    reg_train = generate_latex_table(df_reg, "TMQM_SPECTO_SPECTRAL", "null", "Regression metrics for 10 eV and $\\log_{10}(f)$ aggregations (Train \\& Test: Block 3).", reg_cols, reg_header)
    save_table(reg_train, "regression_train_test_block3.tex")

    reg_test = generate_latex_table(df_reg, "TMQM_SPECTO_SPECTRAL", "test", "Regression metrics for 10 eV and $\\log_{10}(f)$ aggregations (Test Only: Block 3).", reg_cols, reg_header)
    save_table(reg_test, "regression_test_only_block3.tex")

    # Spectral Tables
    spec_train = generate_latex_table(df_reg, "TMQM_SPECTO_SPECTRAL", "null", "Spectral reconstruction metrics on tmQMg* (Train \\& Test: Block 3).", spec_cols, spec_header)
    save_table(spec_train, "spectral_train_test_block3.tex")

    spec_test = generate_latex_table(df_reg, "TMQM_SPECTO_SPECTRAL", "test", "Spectral reconstruction metrics on tmQMg* (Test Only: Block 3).", spec_cols, spec_header)
    save_table(spec_test, "spectral_test_only_block3.tex")
    
    print("\nAll tables have been successfully generated and saved to the 'TEX_DIR' directory.")

if __name__ == "__main__":
    main()