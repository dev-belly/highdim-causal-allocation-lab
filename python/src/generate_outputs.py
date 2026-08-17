"""
从已有的 results.csv 生成图表与报告。
"""

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiment import (
    plot_coverage_vs_sample_size,
    plot_rmse_heatmap,
    plot_weight_bias,
    generate_report,
)

if __name__ == "__main__":
    output_dir = "output"
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(tables_dir, "results.csv"))

    plot_coverage_vs_sample_size(df, figures_dir)
    plot_rmse_heatmap(df, figures_dir)
    plot_weight_bias(df, figures_dir)

    report_path = generate_report(df, output_dir)
    print(f"Report generated: {report_path}")
