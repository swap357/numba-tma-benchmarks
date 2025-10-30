#!/usr/bin/env python3
"""TMA Metrics Visualization - Hierarchical stacked bar charts"""
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TMA_HIERARCHY = {
    'l1': ['tma_retiring', 'tma_bad_speculation', 'tma_frontend_bound', 'tma_backend_bound'],
    'l2': {
        'retiring': ['tma_light_operations', 'tma_heavy_operations'],
        'bad_speculation': ['tma_branch_mispredicts', 'tma_machine_clears'],
        'frontend': ['tma_fetch_latency', 'tma_fetch_bandwidth'],
        'backend': ['tma_memory_bound', 'tma_core_bound'],
    },
    'l3': {
        'light_operations': ['tma_fp_arith', 'tma_int_operations', 'tma_fused_instructions', 
                             'tma_memory_operations', 'tma_non_fused_branches', 'tma_nop_instructions'],
        'heavy_operations': ['tma_few_uops_instructions', 'tma_microcode_sequencer'],
        'memory_bound': ['tma_l1_bound', 'tma_l2_bound', 'tma_l3_bound', 'tma_dram_bound', 'tma_store_bound'],
        'core_bound': ['tma_divider', 'tma_ports_utilization'],
    }
}

L1_COLORS = {
    'tma_retiring': '#2ecc71',
    'tma_bad_speculation': '#e74c3c', 
    'tma_frontend_bound': '#3498db',
    'tma_backend_bound': '#f39c12',
}

L2_COLORS = {
    'retiring': '#2ecc71', 
    'bad_speculation': '#e74c3c', 
    'frontend': '#3498db', 
    'backend': '#f39c12'
}

L3_COLORS = {
    'light_operations': '#27ae60',  # Green shade for retiring/light
    'heavy_operations': '#229954',  # Darker green for retiring/heavy
    'memory_bound': '#95a5a6',      # Gray for memory bound
    'core_bound': '#7f8c8d',        # Darker gray for core bound
}

def format_time(ns):
    """Format time in appropriate unit (ns, µs, ms, s)"""
    if ns >= 1e9:
        return f"{ns/1e9:.2f}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.2f}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.2f}µs"
    else:
        return f"{ns:.0f}ns"

def format_label(metric_name):
    """Convert tma_metric_name to 'Metric Name'"""
    return metric_name.replace('tma_', '').replace('_', ' ').title()

def add_bar_trace(fig, engine, metric, value, row, color, opacity=1.0, min_text=3, fontsize=10, show_in_legend=False):
    """Helper to add a bar trace with consistent styling"""
    y_label = "Python" if engine == 'py' else "JIT"
    label = format_label(metric)
    
    fig.add_trace(go.Bar(
        name=label,
        x=[value],
        y=[y_label],
        orientation='h',
        marker=dict(color=color, line=dict(color='white', width=0.5), opacity=opacity),
        text=f"{value:.1f}%" if value > min_text else "",
        textposition='inside',
        textfont=dict(size=fontsize),
        hovertemplate=f"<b>{label}</b><br>{y_label}: {value:.1f}%<extra></extra>",
        showlegend=show_in_legend,
        legendgroup=label,
    ), row=row, col=1)

def plot_full_hierarchy(df, function):
    """
    Create comprehensive view with L1, L2, and L3 metrics in stacked subplots
    Python vs JIT comparison at all levels
    """
    func_data = df[df['function'] == function]
    
    if len(func_data) == 0:
        print(f"Error: Function '{function}' not found")
        print(f"Available: {', '.join(df['function'].unique())}")
        sys.exit(1)
    
    # Create subplots: L1, L2, L3
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.25, 0.4, 0.35],
        subplot_titles=(
            '<b>L1: Top-level Categories</b>',
            '<b>L2: Breakdown by Category</b>',
            '<b>L3: Detailed Breakdown (Retiring, Memory & Core)</b>'
        ),
        vertical_spacing=0.12,
        specs=[[{'type': 'bar'}], [{'type': 'bar'}], [{'type': 'bar'}]]
    )
    
    engines = ['py', 'jit']
    
    # L1 METRICS (show in legend only for first engine)
    for i, engine in enumerate(engines):
        row = func_data[func_data['engine'] == engine].iloc[0]
        for metric in TMA_HIERARCHY['l1']:
            if metric in row:
                add_bar_trace(fig, engine, metric, row[metric], row=1, 
                             color=L1_COLORS.get(metric, '#95a5a6'),
                             show_in_legend=(i == 0))
    
    # L2 METRICS
    for engine in engines:
        row = func_data[func_data['engine'] == engine].iloc[0]
        for cat, metrics in TMA_HIERARCHY['l2'].items():
            for metric in metrics:
                if metric in row:
                    add_bar_trace(fig, engine, metric, row[metric], row=2,
                                 color=L2_COLORS.get(cat, '#95a5a6'), opacity=0.8, min_text=2)
    
    # L3 METRICS (only significant ones > 1%)
    for engine in engines:
        row = func_data[func_data['engine'] == engine].iloc[0]
        for cat, metrics in TMA_HIERARCHY['l3'].items():
            for metric in metrics:
                if metric in row and row[metric] > 1.0:
                    add_bar_trace(fig, engine, metric, row[metric], row=3,
                                 color=L3_COLORS.get(cat, '#95a5a6'), opacity=0.7, min_text=2, fontsize=9)
    
    # Get timing info
    py_time_ns = func_data[func_data['engine'] == 'py']['elapsed_ns_mean'].iloc[0]
    jit_time_ns = func_data[func_data['engine'] == 'jit']['elapsed_ns_mean'].iloc[0]
    speedup = py_time_ns / jit_time_ns
    
    fig.update_layout(
        title=f"<b>{function}</b> - Complete TMA Hierarchy<br>" + 
              f"<sub>Python: {format_time(py_time_ns)} | JIT: {format_time(jit_time_ns)} | Speedup: {speedup:.1f}x</sub>",
        barmode='stack',
        height=900,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title="L1 Categories",
        ),
    )
    
    fig.update_xaxes(title_text='Percentage (%)', range=[0, 100])
    fig.update_yaxes(categoryorder='array', categoryarray=['JIT', 'Python'])
    
    return fig

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot.py <csv_file> <function>")
        print("Example: python plot.py results_tma.csv sum1d")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])
    fig = plot_full_hierarchy(df, sys.argv[2])
    fig.show()

if __name__ == "__main__":
    main()

