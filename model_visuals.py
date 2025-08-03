"""
model_visuals.py

High-quality, consistent visualizations for ABM dissertation results.
Each function saves a .png to the specified output directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Set global style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'font.family': 'DejaVu Sans',
})

OUTPUT_DIR = r'C:/Users/owenk/OneDrive/Documents/UCL/Dissertation/Agent Based Modelling/model outputs'

# Utility: Ensure output directory exists
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_country_name(name):
    # Capitalize first letter of each word except 'of'
    return ' '.join([w.capitalize() if w.lower() != 'of' else 'of' for w in name.split()])

def preprocess_visual_supplier_states(df, agent_ids_col='agent_ids', status_col='statuses', period_col='period'):
    """
    For each supplier, after the first 'SWITCHED' event, only the next two periods are marked as switched, and all subsequent periods are operational. No supplier can be switched more than once in the heatmap.
    """
    import numpy as np
    periods = df[period_col].tolist()
    agent_ids = df[agent_ids_col].iloc[0]
    status_history = {aid: [] for aid in agent_ids}
    for _, row in df.iterrows():
        statuses = row[status_col]
        for aid in agent_ids:
            status_history[aid].append(statuses.get(aid, 'OPERATIONAL'))
    n_periods = len(periods)
    state_matrix = np.zeros((len(agent_ids), n_periods), dtype=int)
    for i, aid in enumerate(agent_ids):
        switched_found = False
        switch_start = None
        for j in range(n_periods):
            status = status_history[aid][j]
            if not switched_found and status == 'SWITCHED':
                switched_found = True
                switch_start = j
                # Mark this and next period as switched (2)
                for k in range(j, min(j+2, n_periods)):
                    state_matrix[i, k] = 2
            elif switched_found:
                # After the two switched periods, always operational
                if switch_start is not None and j >= switch_start + 2:
                    state_matrix[i, j] = 0
            else:
                if status == 'AFFECTED':
                    state_matrix[i, j] = 1
                else:
                    state_matrix[i, j] = 0
    # Add to DataFrame for plotting
    df = df.copy()
    df['visual_state_matrix'] = [state_matrix[:, t] for t in range(n_periods)]
    df['visual_state_matrix_full'] = [state_matrix.copy() for _ in range(n_periods)]
    return df

# Helper: Shade periods by tariff rate and annotate tariff %
def shade_tariff_periods(ax, periods, tariff_phases, y_max, y_min=0, alpha=0.25):
    """
    Shade each period according to its tariff rate and annotate the tariff %.
    periods: list/array of period values (x-axis)
    tariff_phases: list of dicts with 'start', 'end', 'magnitude'
    y_max: top of the y-axis for annotation placement
    y_min: bottom of the y-axis (default 0)
    alpha: transparency for shading
    """
    import matplotlib.colors as mcolors
    # Build a mapping of period -> tariff rate
    period_to_tariff = {p: 0.0 for p in periods}
    for phase in tariff_phases:
        for p in range(phase['start'], phase['end'] + 1):
            if p in period_to_tariff:
                period_to_tariff[p] = phase['magnitude']
    # Define a light red colormap (white to red)
    base_color = np.array(mcolors.to_rgb('#FF0000'))
    for i, p in enumerate(periods):
        rate = period_to_tariff.get(p, 0.0)
        # Interpolate color: white (0) to light red (max 0.25 intensity)
        color = (1 - 0.25 * rate) * np.ones(3) + 0.25 * rate * base_color
        color = np.clip(color, 0, 1)
        ax.axvspan(p - 0.5, p + 0.5, ymin=0, ymax=1, color=color, alpha=alpha, zorder=0)
        if rate > 0:
            ax.text(p, y_max * 0.97, f'{rate*100:.0f}%', ha='center', va='top', fontsize=9, fontweight='bold', color='#B22222',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'), zorder=10)

# Improved Helper: Shade tariff phases and annotate a single label per phase

def shade_tariff_phases(ax, tariff_phases, y_max, y_min=0, alpha=0.35):
    """
    Shade each tariff phase with a light red (darker for higher tariffs) and annotate a single tariff % label in the center of the phase.
    The shading intensity is normalized to the maximum tariff rate in the current run.
    """
    import matplotlib.colors as mcolors
    base_color = np.array(mcolors.to_rgb('#FF0000'))
    # Find the maximum tariff rate in this run for normalization
    max_tariff = max((phase['magnitude'] for phase in tariff_phases), default=0) or 1  # avoid division by zero
    for phase in tariff_phases:
        rate = phase['magnitude']
        # Normalize intensity: 0 (white) to 1 (max red)
        normalized = rate / max_tariff
        color = (1 - normalized) * np.ones(3) + normalized * base_color
        color = np.clip(color, 0, 1)
        ax.axvspan(phase['start'] - 0.5, phase['end'] + 0.5, ymin=0, ymax=1, color=color, alpha=alpha, zorder=0)
        # Place a single label in the center of the phase
        mid = (phase['start'] + phase['end']) / 2
        ax.text(mid, y_max * 0.97, f"{rate*100:.0f}%", ha='center', va='top', fontsize=11, fontweight='bold', color='#B22222',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, boxstyle='round,pad=0.2'), zorder=10)

# 1. Landed Cost Over Time + Cumulative Landed Cost
def plot_tier1_landed_cost_analysis(df, tariff_phases=None, scenario_label='', filename='tier1_landed_cost_with_switching.png'):
    """
    Plot Tier 1 Landed Cost Analysis: With Switching and cumulative extra cost over baseline.
    """
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.2, 1]})
    # Consistent colors
    color_switching = '#1f77b4'  # blue
    color_noswitch = '#d62728'   # red
    color_baseline = '#666666'   # gray
    color_cumulative = '#800080' # purple
    # Main title
    title = f'Landed Cost Analysis: {scenario_label}' if scenario_label else 'Landed Cost Analysis'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    # Top: Landed Cost Over Time
    if tariff_phases is not None:
        y_max = df['tier1_landed_cost'].max()
        shade_tariff_phases(ax1, tariff_phases, y_max)
    # Choose color based on scenario label
    line_color = color_switching if 'switch' in scenario_label.lower() else color_noswitch
    line1, = ax1.plot(df['period'], df['tier1_landed_cost'], marker='o', color=line_color, linewidth=2, label='Landed Cost')
    line2 = ax1.axhline(df['tier1_baseline_cost'].iloc[0], color=color_baseline, linestyle='--', linewidth=2, label='Baseline Cost')
    ax1.set_ylabel('Landed Cost (USD)')
    ax1.set_title('Landed Cost Over Time')
    ax1.legend([line1, line2], ['Landed Cost', 'Baseline Cost'])
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Bottom: Cumulative Extra Cost Over Baseline
    cumulative_extra = (df['tier1_landed_cost'] - df['tier1_baseline_cost']).cumsum()
    ax2.plot(df['period'], cumulative_extra, marker='o', color=color_cumulative, linewidth=2)
    ax2.set_ylabel('Cumulative Extra Cost Over Baseline (USD)')
    ax2.set_title('Cumulative Extra Cost Over Baseline (Tier 1 Only)')
    ax2.set_xlabel('Simulation Period')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_supplier_state_current_per_period(df, scenario_label='', filename='supplier_state_current_per_period.png'):
    """
    Plot current supplier state counts per period (not cumulative).
    Shows how many suppliers are currently in each state at each period.
    This aligns with landed cost plots which show current costs per period.
    """
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(12, 6))
    title = f'Supplier State Evolution (Current per Period): {scenario_label}' if scenario_label else 'Supplier State Evolution (Current per Period)'
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Consistent color mapping
    state_colors = {
        'Operational': '#2ECC71',
        'Switched': '#8E44AD',
        'Economically Affected': '#E67E22'
    }

    # Plot each state as a line (current columns)
    ax.plot(df['period'], df['currently_operational'], label='Currently Operational', color=state_colors['Operational'], linewidth=2.5, marker='o', markersize=4)
    
    if 'currently_switched' in df.columns:
        ax.plot(df['period'], df['currently_switched'], label='Currently Switched', color=state_colors['Switched'], linewidth=2.5, marker='^', markersize=4)
    
    # Use 'currently_affected' for current per period (currently affected)
    if 'currently_affected' in df.columns:
        ax.plot(df['period'], df['currently_affected'], label='Currently Economically Affected', color=state_colors['Economically Affected'], linewidth=2.5, marker='d', markersize=4, linestyle='--')

    ax.set_xlabel('Simulation Period', fontsize=12)
    ax.set_ylabel('Number of Suppliers (Current)', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_supplier_state_evolution(df, scenario_label='', filename='supplier_state_evolution.png'):
    """
    Original supplier state evolution plot (cumulative counts).
    For clarity, consider using plot_supplier_state_evolution_cumulative instead.
    """
    plot_supplier_state_evolution_cumulative(df, scenario_label, filename)

# 3. Cumulative Cost Impact per Supplier
def plot_cumulative_cost_impact(df, filename='cumulative_cost_impact.png', top_n=10):
    """
    Horizontal bar plot: Top N suppliers by cumulative cost impact (log scale).
    Supplier name/code and tier in labels.
    """
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(12, 7))
    title = 'Cumulative Cost Impact per Supplier'
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Filter out suppliers with zero or negative impact
    df_filtered = df[df['cumulative_cost_impact'] > 0].copy()
    # Sort and select top N
    df_sorted = df_filtered.sort_values('cumulative_cost_impact', ascending=False).head(top_n)
    # Create label with name/code and tier
    labels = [f"{row['supplier_name']} (T{row['tier']})" for _, row in df_sorted.iterrows()]
    ax.barh(labels, df_sorted['cumulative_cost_impact'], color='#E74C3C', alpha=0.85)
    ax.set_xlabel('Cumulative Cost Impact (USD, Log Scale)', fontsize=12)
    ax.set_ylabel('Supplier (Tier)', fontsize=12)
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 4. Network Cost Impact Graph (Before and After)
def plot_network_cost_impact(G_before, G_after, cost_impact_before, cost_impact_after, tier_dict, scenario_label='', filename='network_cost_impact.png'):
    """
    Hierarchical layout by tier. Node color: log-scaled cost impact.
    Two snapshots: before and after. Optionally highlight switched/at-risk.
    """
    import matplotlib as mpl
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    title = f'Network Cost Impact (Before vs After): {scenario_label}' if scenario_label else 'Network Cost Impact (Before vs After)'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Helper: get multipartite layout by tier
    def get_multipartite_pos(G, tier_dict):
        layers = {}
        for node, tier in tier_dict.items():
            layers.setdefault(tier, []).append(node)
        pos = {}
        n_layers = len(layers)
        for i, (tier, nodes) in enumerate(sorted(layers.items())):
            y = 1 - i / max(n_layers - 1, 1)
            for j, node in enumerate(nodes):
                x = j / max(len(nodes) - 1, 1) if len(nodes) > 1 else 0.5
                pos[node] = (x, y)
        return pos

    # Before shock
    pos_before = get_multipartite_pos(G_before, tier_dict)
    cost_vals_before = np.array([cost_impact_before.get(n, 0) for n in G_before.nodes()])
    norm_before = mpl.colors.Normalize(vmin=0, vmax=np.max(cost_vals_before) if np.max(cost_vals_before) > 0 else 1)
    cmap = mpl.cm.OrRd
    node_colors_before = [cmap(norm_before(val)) for val in cost_vals_before]
    nx.draw(G_before, pos_before, ax=ax1, node_color=node_colors_before, node_size=80, with_labels=False, edge_color='gray', width=0.5, alpha=0.8)
    ax1.set_title('Before Shock', fontsize=14, fontweight='bold')
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_before)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Cost Impact (log-scaled)', fontsize=11)

    # After shock
    pos_after = get_multipartite_pos(G_after, tier_dict)
    cost_vals_after = np.array([cost_impact_after.get(n, 0) for n in G_after.nodes()])
    # Use log scale for color
    log_cost_vals_after = np.log1p(cost_vals_after)
    norm_after = mpl.colors.Normalize(vmin=0, vmax=np.max(log_cost_vals_after) if np.max(log_cost_vals_after) > 0 else 1)
    node_colors_after = [cmap(norm_after(val)) for val in log_cost_vals_after]
    nx.draw(G_after, pos_after, ax=ax2, node_color=node_colors_after, node_size=80, with_labels=False, edge_color='gray', width=0.5, alpha=0.8)
    ax2.set_title('After Shock', fontsize=14, fontweight='bold')
    sm2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_after)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label('Cost Impact (log(1+x))', fontsize=11)

    for ax in [ax1, ax2]:
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 5. Tier-Level Impact Summary
def plot_tier_level_impact(df, filename='tier_impact_with_switching.png', landed_cost_df=None):
    """
    Top: Total cost impact by tier (linear scale).
    Bottom: Switched + Affected counts by tier (ever switched/affected, side-by-side bars).
    Also saves a log-scaled version of the top panel as 'tier_impact_log_with_switching.png'.
    If landed_cost_df is provided, Tier 0 will show the cumulative landed cost impact.
    If Tier 0 is missing, it will be added. If present, it will be overwritten.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    save_path_log = os.path.join(OUTPUT_DIR, 'tier_impact_log_with_switching.png')

    df = df.copy()
    # Patch or add Tier 0 with landed cost impact if landed_cost_df is provided
    if landed_cost_df is not None:
        tier0_impact = (landed_cost_df['landed_cost'] - landed_cost_df['base_cost']).sum()
        # Determine if Tier 0 was ever affected
        tier0_ever_affected = (landed_cost_df['landed_cost'] > landed_cost_df['base_cost']).any()
        n_affected_0 = 1 if tier0_ever_affected else 0
        n_switched_0 = 0
        if (df['tier'] == 0).any():
            df.loc[df['tier'] == 0, 'total_cost_impact'] = tier0_impact
            df.loc[df['tier'] == 0, 'n_affected'] = n_affected_0
            df.loc[df['tier'] == 0, 'n_switched'] = n_switched_0
        else:
            df = pd.concat([
                pd.DataFrame([{
                    'tier': 0,
                    'total_cost_impact': tier0_impact,
                    'n_suppliers': 1,
                    'n_switched': n_switched_0,
                    'n_affected': n_affected_0
                }]),
                df
            ], ignore_index=True)
        df = df.sort_values('tier').reset_index(drop=True)

    tiers = df['tier'].astype(str)
    x = np.arange(len(tiers))
    width = 0.35

    # Colors to match geo_risk
    cost_color = '#2E86AB'      # blue
    switched_color = '#8E44AD'  # purple
    affected_color = '#F39C12'  # orange

    # Top: Total cost impact (linear scale)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.2, 1]})
    title = 'Tier-Level Impact Summary'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    ax1.bar(x, df['total_cost_impact'], color=cost_color, alpha=0.85)
    ax1.set_ylabel('Total Cost Impact (USD)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers)
    ax1.set_title('Total Cost Impact by Tier (Linear Scale)')

    # Bottom: Switched + Affected counts
    ax2.bar(x - width/2, df['n_switched'], width, label='Switched', color=switched_color, alpha=0.85)
    ax2.bar(x + width/2, df['n_affected'], width, label='Affected', color=affected_color, alpha=0.85)
    ax2.set_ylabel('Number of Suppliers')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers)
    ax2.set_title('Switched and Affected Suppliers by Tier')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.figure(fig.number)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Log-scaled version of the top panel
    fig_log, (ax1_log, ax2_log) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.2, 1]})
    title_log = 'Tier-Level Impact Summary (Log Scale)'
    fig_log.suptitle(title_log, fontsize=20, fontweight='bold')
    ax1_log.bar(x, df['total_cost_impact'], color=cost_color, alpha=0.85)
    ax1_log.set_ylabel('Total Cost Impact (USD, log scale)')
    ax1_log.set_xticks(x)
    ax1_log.set_xticklabels(tiers)
    ax1_log.set_title('Total Cost Impact by Tier (Log Scale)')
    ax1_log.set_yscale('log')

    # Bottom: Switched + Affected counts (same as linear version)
    ax2_log.bar(x - width/2, df['n_switched'], width, label='Switched', color=switched_color, alpha=0.85)
    ax2_log.bar(x + width/2, df['n_affected'], width, label='Affected', color=affected_color, alpha=0.85)
    ax2_log.set_ylabel('Number of Suppliers')
    ax2_log.set_xticks(x)
    ax2_log.set_xticklabels(tiers)
    ax2_log.set_title('Switched and Affected Suppliers by Tier')
    ax2_log.legend()

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.figure(fig_log.number)
    fig_log.savefig(save_path_log, dpi=300, bbox_inches='tight')
    plt.close(fig_log)

# 6. Geographic Risk View
def plot_geographic_risk(df, filename='geographic_risk.png', top_n=10):
    """
    Top: Total cost increase per country (log scale, exclude zero).
    Bottom: Switched + Affected counts by country (ever switched/affected, side-by-side bars).
    """
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))
    title = 'Geographic Risk View'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Top Countries by Total Cost Increase (log scale, exclude zero)
    if 'total_cost_increase' not in df.columns:
        if 'avg_cost_increase' in df.columns and 'n_suppliers' in df.columns:
            df['total_cost_increase'] = df['avg_cost_increase'] * df['n_suppliers']
        else:
            df['total_cost_increase'] = df['avg_cost_increase']
    df_nonzero_cost = df[df['total_cost_increase'] > 0].sort_values('total_cost_increase', ascending=False).head(top_n)
    ax1.bar(df_nonzero_cost['country'], df_nonzero_cost['total_cost_increase'], color='#2E86AB', alpha=0.85)
    ax1.set_ylabel('Total Cost Increase (USD, Log Scale)', fontsize=12)
    ax1.set_title('Top Countries by Total Cost Increase', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(df_nonzero_cost['country'].apply(format_country_name), rotation=45, ha='right')
    ax1.set_xlabel('Top 5 Countries', fontsize=12)
    # Safe log scale handling
    if (df_nonzero_cost['total_cost_increase'] > 0).any():
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
        ax1.set_ylabel('Total Cost Increase (USD, Linear Scale)', fontsize=12)
        ax1.text(0.5, 0.5, 'No positive cost increases', transform=ax1.transAxes, 
                 ha='center', va='center', fontsize=14, color='red')
    ax1.grid(True, axis='y', alpha=0.3)

    # 2. Top Countries by # Switched and # Affected (grouped bar, exclude zero)
    # Only keep countries with nonzero n_switched or n_affected
    if 'n_switched' in df.columns and 'n_affected' in df.columns:
        df_grouped = df[(df['n_switched'] > 0) | (df['n_affected'] > 0)].copy()
        df_grouped = df_grouped.sort_values('n_affected', ascending=False).head(top_n)
        x = np.arange(len(df_grouped))
        bar_width = 0.4
        ax2.bar(x - bar_width/2, df_grouped['n_affected'], width=bar_width, color='#F39C12', alpha=0.85, label='# Affected')
        ax2.bar(x + bar_width/2, df_grouped['n_switched'], width=bar_width, color='#8E44AD', alpha=0.85, label='# Switched')
        ax2.set_ylabel('Number of Suppliers', fontsize=12)
        ax2.set_title('Top Countries by # Affected and # Switched', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_grouped['country'].apply(format_country_name), rotation=45, ha='right')
        ax2.set_xlabel('Top 5 Countries', fontsize=12)
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
    else:
        ax2.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- Geographic Heatmap (Choropleth) of Total Impact ---
# Commented out due to compatibility issues with plotly/geopandas on this system
# def plot_geographic_heatmap(df, value_col='total_cost_increase', filename='geographic_heatmap.png', log_scale=True):
#     """
#     Plot a choropleth map of total cost increase by country using plotly or geopandas.
#     """
#     try:
#         import plotly.express as px
#         # Prepare data: ensure country names are compatible with plotly
#         df_map = df.copy()
#         df_map = df_map[df_map[value_col] > 0]
#         if log_scale:
#             df_map['log_value'] = np.log10(df_map[value_col])
#             color_col = 'log_value'
#             color_label = f'Log10 {value_col.replace("_", " ").title()}'
#         else:
#             color_col = value_col
#             color_label = value_col.replace("_", " ").title()
#         fig = px.choropleth(
#             df_map,
#             locations='country',
#             locationmode='country names',
#             color=color_col,
#             color_continuous_scale='Viridis',
#             title='Geographic Heatmap of Total Cost Increase',
#             labels={color_col: color_label}
#         )
#         fig.write_image(os.path.join(OUTPUT_DIR, filename), scale=2)
#     except ImportError:
#         # Fallback to geopandas + matplotlib
#         import geopandas as gpd
#         world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#         df_map = df.copy()
#         df_map = df_map[df_map[value_col] > 0]
#         if log_scale:
#             df_map['log_value'] = np.log10(df_map[value_col])
#             color_col = 'log_value'
#         else:
#             color_col = value_col
#         merged = world.merge(df_map, left_on='name', right_on='country', how='left')
#         fig, ax = plt.subplots(1, 1, figsize=(16, 8))
#         merged.plot(column=color_col, ax=ax, legend=True, cmap='viridis', missing_kwds={"color": "lightgrey"})
#         ax.set_title('Geographic Heatmap of Total Cost Increase', fontsize=16, fontweight='bold')
#         ax.axis('off')
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
#         plt.close()

# 7. Switching Cost Impact
# Remove plot_switching_cost_impact function (lines 484-520)

def plot_post_simulation_network_impact(abm, filename='network_cost_impact_post_sim.png'):
    """
    Visualize the post-simulation supply chain network:
    - All nodes and edges grayed out by default
    - Colored nodes: nodes that were ever switched (ever_switched=True)
    - Colored edges: outgoing edges from suppliers that were ever affected (ever_affected=True)
    - Node positions match EDA visual (hierarchical by tier)
    """
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    G = abm.network
    agents = abm.agents
    # Build tier dict for layout
    tier_dict = {n: agents[n].characteristics.tier for n in G.nodes if n in agents}
    # Use the same layout as EDA visual
    def get_multipartite_pos(G, tier_dict):
        layers = {}
        for node, tier in tier_dict.items():
            layers.setdefault(tier, []).append(node)
        pos = {}
        n_layers = len(layers)
        for i, (tier, nodes) in enumerate(sorted(layers.items())):
            y = 1 - i / max(n_layers - 1, 1)
            for j, node in enumerate(nodes):
                x = j / max(len(nodes) - 1, 1) if len(nodes) > 1 else 0.5
                pos[node] = (x, y)
        return pos
    pos = get_multipartite_pos(G, tier_dict)
    # Identify switched nodes and affected suppliers
    switched_nodes = [n for n in G.nodes if hasattr(agents[n], 'ever_switched') and agents[n].ever_switched]
    affected_suppliers = [n for n in G.nodes if hasattr(agents[n], 'ever_affected') and agents[n].ever_affected]
    # All edges grayed out by default
    edge_colors = []
    edge_widths = []
    highlighted_edges = set()
    for u, v in G.edges():
        if u in affected_suppliers:
            edge_colors.append('#F39C12')  # orange for affected
            edge_widths.append(2.5)
            highlighted_edges.add((u, v))
        else:
            edge_colors.append('lightgray')
            edge_widths.append(1.0)
    # Draw all nodes grayed out
    node_colors = ['#B0B0B0' for _ in G.nodes()]
    node_sizes = [220 for _ in G.nodes()]
    # Overlay: highlight switched nodes
    for idx, n in enumerate(G.nodes()):
        if n in switched_nodes:
            node_colors[idx] = '#8E44AD'  # purple for switched
            node_sizes[idx] = 350
    plt.figure(figsize=(16, 9))
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95, linewidths=1, edgecolors='black')
    # Optionally: add node labels (supplier code)
    # nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    plt.title('Post-Simulation Network: Switched Nodes and Cost-Increased Edges', fontsize=18, fontweight='bold')
    # Custom legend with circular node markers
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#B0B0B0', markersize=12, label='Unchanged Supplier', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8E44AD', markersize=12, label='Switched Supplier', markeredgecolor='black'),
        Line2D([0], [0], color='#F39C12', lw=2.5, label='Affected'),
        Line2D([0], [0], color='lightgray', lw=2.0, label='Unaffected'),
    ]
    plt.legend(handles=legend_handles, loc='upper left', fontsize=12, frameon=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_full_network_landed_cost_analysis(df, tariff_phases=None, scenario_label='', filename='full_network_landed_cost_with_switching.png'):
    """
    Plot Full Network Landed Cost Analysis: With Switching
    Top: Full network landed cost over time, with baseline.
    Bottom: Cumulative extra cost over baseline (all tiers).
    """
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.2, 1]})
    # Consistent colors
    color_switching = '#1f77b4'  # blue
    color_noswitch = '#d62728'   # red
    color_baseline = '#666666'   # gray
    color_cumulative = '#800080' # purple
    # Main title
    title = f'Full Network Landed Cost Analysis: {scenario_label}' if scenario_label else 'Full Network Landed Cost Analysis'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    # Top: Full Network Landed Cost Over Time
    if tariff_phases is not None:
        y_max = (df['full_network_landed_cost'] / 1e6).max()
        shade_tariff_phases(ax1, tariff_phases, y_max)
    # Choose color based on scenario label
    line_color = color_switching if 'switch' in scenario_label.lower() else color_noswitch
    line1, = ax1.plot(df['period'], df['full_network_landed_cost'] / 1e6, marker='o', color=line_color, linewidth=2, label='Landed Cost')
    line2, = ax1.plot(df['period'], df['base_cost'] / 1e6, linestyle='--', color=color_baseline, linewidth=2, label='Baseline Cost')
    ax1.set_ylabel('Landed Cost (Million USD)')
    ax1.set_xlabel('Period')
    ax1.set_title('Full Network Landed Cost Over Time')
    ax1.legend([line1, line2], ['Landed Cost', 'Baseline Cost'])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Bottom: Cumulative Extra Cost Over Baseline
    cumulative_extra = (df['full_network_landed_cost'] - df['base_cost']).cumsum() / 1e6
    ax2.plot(df['period'], cumulative_extra, marker='o', color=color_cumulative, linewidth=2)
    ax2.set_ylabel('Cumulative Extra Cost (Million USD)')
    ax2.set_xlabel('Period')
    ax2.set_title('Cumulative Extra Cost Over Baseline (All Tiers)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close(fig)

def plot_meaningful_supplier_state_evolution(df, scenario_label='', filename='meaningful_supplier_state_evolution.png'):
    """
    Plot meaningful supplier state evolution showing network resilience and recovery.
    Uses visually processed columns if present.
    """
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(12, 6))
    title = f'Supplier State Evolution (Meaningful, Per Period): {scenario_label}' if scenario_label else 'Supplier State Evolution (Meaningful, Per Period)'
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Color mapping
    colors = {
        'Total Suppliers': '#95A5A6',  # Gray
        'Switched': '#8E44AD',    # Purple
        'Affected': '#E67E22',    # Orange
        'Operational': '#2ECC71'  # Green
    }

    # Plot total suppliers (constant)
    total_suppliers = df['total_agents'].iloc[0] if 'total_agents' in df.columns else df['operational'].iloc[0]
    ax.axhline(y=total_suppliers, color=colors['Total Suppliers'], linestyle='-', linewidth=2, 
               label=f'Total Suppliers ({total_suppliers})', alpha=0.7)
    
    # Plot visually switched (per period)
    if 'visually_switched' in df.columns:
        ax.plot(df['period'], df['visually_switched'], label='Switched', color=colors['Switched'], 
               linewidth=2.5, marker='^', markersize=4)
    elif 'currently_switched' in df.columns:
        ax.plot(df['period'], df['currently_switched'], label='Switched', color=colors['Switched'], 
               linewidth=2.5, marker='^', markersize=4)
    
    # Plot currently affected (per period)
    if 'currently_affected' in df.columns:
        ax.plot(df['period'], df['currently_affected'], label='Affected', color=colors['Affected'], 
               linewidth=2.5, marker='d', markersize=4, linestyle='--')
    
    # Plot visually operational (per period)
    if 'visually_operational' in df.columns:
        ax.plot(df['period'], df['visually_operational'], label='Operational', 
               color=colors['Operational'], linewidth=2.5, marker='o', markersize=4)
    elif 'currently_operational' in df.columns:
        ax.plot(df['period'], df['currently_operational'], label='Operational', 
               color=colors['Operational'], linewidth=2.5, marker='o', markersize=4)

    ax.set_xlabel('Simulation Period', fontsize=12)
    ax.set_ylabel('Number of Suppliers', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Meaningful supplier state evolution plot saved to {save_path}")

# Main function to generate all visuals for a scenario
def generate_all_visuals(data_dict, scenario_label):
    """
    data_dict: Dictionary with all required DataFrames/objects for the scenario.
    scenario_label: e.g. 'With Switching', 'No Switching', 'Tariff Shock', etc.
    """
    ensure_output_dir()
    # Example usage (to be filled in with actual plotting logic):
    # plot_landed_and_cumulative_cost(data_dict['landed_cost'], ...)
    # plot_supplier_state_evolution(data_dict['supplier_state'], ...)
    # ...
    pass 

def plot_overlaid_landed_cost_comparison(df_switching, df_no_switching, filename='model outputs/overlaid_landed_cost_comparison.png', tariff_phases=None):
    """Plot full network landed cost over time for switching vs no switching on the same axes (in millions), with optional tariff shading."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    color_switching = '#1f77b4'  # blue
    color_noswitch = '#d62728'   # red
    color_baseline = '#666666'   # gray
    ensure_output_dir()
    # Always save to OUTPUT_DIR
    save_path = os.path.join(OUTPUT_DIR, 'overlaid_landed_cost_comparison.png')
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    # Add tariff shading if provided
    if tariff_phases is not None:
        y_max = max((df_switching['full_network_landed_cost'] / 1e6).max(), (df_no_switching['full_network_landed_cost'] / 1e6).max())
        shade_tariff_phases(ax, tariff_phases, y_max)
    # Plot lines
    line1, = plt.plot(df_switching['period'], df_switching['full_network_landed_cost'] / 1e6, label='With Switching', color=color_switching, marker='o', linewidth=2)
    line2, = plt.plot(df_no_switching['period'], df_no_switching['full_network_landed_cost'] / 1e6, label='No Switching', color=color_noswitch, linestyle='--', marker='x', linewidth=2)
    line3 = plt.axhline(y=df_switching['base_cost'].iloc[0] / 1e6, color=color_baseline, linestyle='--', label='Baseline Cost')
    plt.xlabel('Period', fontsize=16, fontweight='bold')
    plt.ylabel('Full Network Landed Cost (Million USD)', fontsize=16, fontweight='bold')
    plt.title('Full Network Landed Cost Over Time: Switching vs No Switching', fontsize=20, fontweight='bold')
    # Custom legend: only show the three main lines
    plt.legend([line1, line2, line3], ['With Switching', 'No Switching', 'Baseline Cost'], fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_overlaid_tier1_landed_cost_comparison(df_switching, df_no_switching, filename='model outputs/overlaid_tier1_landed_cost_comparison.png', tariff_phases=None):
    """Plot Tier 1 landed cost over time for switching vs no switching on the same axes (raw USD), with optional tariff shading."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    color_switching = '#1f77b4'  # blue
    color_noswitch = '#d62728'   # red
    color_baseline = '#666666'   # gray
    ensure_output_dir()
    # Always save to OUTPUT_DIR
    save_path = os.path.join(OUTPUT_DIR, 'overlaid_tier1_landed_cost_comparison.png')
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    # Add tariff shading if provided
    if tariff_phases is not None:
        y_max = max(df_switching['tier1_landed_cost'].max(), df_no_switching['tier1_landed_cost'].max())
        shade_tariff_phases(ax, tariff_phases, y_max)
    # Plot lines
    line1, = plt.plot(df_switching['period'], df_switching['tier1_landed_cost'], label='With Switching', color=color_switching, marker='o', linewidth=2)
    line2, = plt.plot(df_no_switching['period'], df_no_switching['tier1_landed_cost'], label='No Switching', color=color_noswitch, linestyle='--', marker='x', linewidth=2)
    line3 = plt.axhline(y=df_switching['tier1_baseline_cost'].iloc[0], color=color_baseline, linestyle='--', label='Baseline Cost')
    plt.xlabel('Period', fontsize=16, fontweight='bold')
    plt.ylabel('Tier 1 Landed Cost (USD)', fontsize=16, fontweight='bold')
    plt.title('Tier 1 Landed Cost Over Time: Switching vs No Switching', fontsize=20, fontweight='bold')
    # Custom legend: only show the three main lines
    plt.legend([line1, line2, line3], ['With Switching', 'No Switching', 'Baseline Cost'], fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_switching_vs_no_switching_bar_comparison(results_switching, results_no_switching, filename='switching_vs_no_switching_comparison.png'):
    """Bar chart comparing final landed cost, number of switched suppliers, and number of affected suppliers for both scenarios."""
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ['Final Landed Cost', 'Switched Suppliers', 'Affected Suppliers']
    switching_vals = [results_switching['landed_cost'].iloc[-1], results_switching['switched'].iloc[-1], results_switching['affected'].iloc[-1]]
    no_switching_vals = [results_no_switching['landed_cost'].iloc[-1], results_no_switching['switched'].iloc[-1], results_no_switching['affected'].iloc[-1]]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8,6))
    plt.bar(x - width/2, switching_vals, width, label='With Switching', color='blue')
    plt.bar(x + width/2, no_switching_vals, width, label='No Switching', color='red')
    plt.xticks(x, labels)
    plt.ylabel('Value')
    plt.title('Comparison: Switching vs No Switching')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Fix Y-axis labels in tier-level impact plots
# Find all functions with tier-level impact plots and update their plt.ylabel calls
# Example for log scale:
def plot_tier_level_impact(df, filename='tier_impact_with_switching.png', landed_cost_df=None):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    # Top: Total Cost Impact by Tier (Log Scale)
    axs[0].bar(df['tier'], df['total_cost_impact'])
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Tier')
    axs[0].set_ylabel('Total Cost Impact (USD, log scale)')
    axs[0].set_title('Total Cost Impact by Tier (Log Scale)')
    # Bottom: Switched and Affected Suppliers by Tier
    width = 0.35
    x = np.arange(len(df['tier']))
    axs[1].bar(x - width/2, df['n_switched'], width, label='Switched', color='purple')
    axs[1].bar(x + width/2, df['n_affected'], width, label='Affected', color='orange')
    axs[1].set_xlabel('Tier')
    axs[1].set_ylabel('Number of Suppliers')
    axs[1].set_title('Switched and Affected Suppliers by Tier')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(df['tier'])
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 

def plot_supplier_state_heatmap(df, filename='supplier_state_heatmap.png'):
    """
    Plot a heatmap of supplier state trajectories over time using visually processed states.
    Each row is a supplier, each column is a period, color-coded by state.
    Uses the 'visual_state_matrix_full' column if present, else falls back to original logic.
    Legend is placed outside the heatmap, to the right of the final simulation period at the top.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    ensure_output_dir()
    periods = df['period'].tolist()
    agent_ids = df['agent_ids'].iloc[0]
    n_periods = len(periods)
    n_agents = len(agent_ids)
    # Use visual_state_matrix_full if present
    if 'visual_state_matrix_full' in df.columns:
        # Take the first (they are all the same)
        state_matrix = df['visual_state_matrix_full'].iloc[0]
    else:
        # Fallback: build from statuses
        state_matrix = []
        for aid in agent_ids:
            row = []
            for _, rowdata in df.iterrows():
                status = rowdata['statuses'].get(aid, 'OPERATIONAL')
                if status == 'SWITCHED':
                    row.append(2)
                elif status == 'AFFECTED':
                    row.append(1)
                else:
                    row.append(0)
            state_matrix.append(row)
        state_matrix = np.array(state_matrix)
    # Plot
    fig, ax = plt.subplots(figsize=(min(18, n_periods*0.7), min(0.25*n_agents+2, 18)))
    cmap = ListedColormap(["#2ECC71", "#E67E22", "#8E44AD"])  # green, orange, purple
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    im = ax.imshow(state_matrix, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xlabel('Simulation Period', fontsize=14, fontweight='bold')
    ax.set_ylabel('Supplier', fontsize=14, fontweight='bold')
    ax.set_title('Supplier State Trajectory Heatmap', fontsize=18, fontweight='bold')
    ax.set_xticks(np.arange(n_periods))
    ax.set_xticklabels(periods)
    ax.set_yticks(np.arange(n_agents))
    ax.set_yticklabels(agent_ids)
    # Remove colorbar
    # Add custom legend outside the plot, to the right
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', label='Operational'),
        Patch(facecolor='#E67E22', label='Affected'),
        Patch(facecolor='#8E44AD', label='Switched')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0., frameon=True, title=None)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Supplier state heatmap saved to {os.path.join(OUTPUT_DIR, filename)}") 

def plot_supplier_state_switched_affected(df, filename='supplier_state_switched_affected.png'):
    """
    Plot a line chart showing only the number of 'Switched' and 'Affected' suppliers per period.
    No operational or total suppliers lines. Matches the style of the meaningful supplier state evolution plot.
    Uses the 'statuses' column (dict of agent_id -> status) in df.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    ensure_output_dir()
    periods = df['period'].tolist()
    agent_ids = df['agent_ids'].iloc[0]
    n_periods = len(periods)
    n_agents = len(agent_ids)
    # Build state counts per period
    affected_counts = []
    switched_counts = []
    for _, row in df.iterrows():
        aff = 0
        sw = 0
        for aid in agent_ids:
            status = row['statuses'].get(aid, 'OPERATIONAL')
            if status == 'SWITCHED':
                sw += 1
            elif status == 'AFFECTED':
                aff += 1
        affected_counts.append(aff)
        switched_counts.append(sw)
    # Line plot
    plt.figure(figsize=(12, 6))
    plt.plot(periods, switched_counts, label='Switched', color="#8E44AD", linewidth=2.5, marker='^', markersize=4)
    plt.plot(periods, affected_counts, label='Affected', color="#E67E22", linewidth=2.5, marker='d', markersize=4, linestyle='--')
    plt.xlabel('Simulation Period')
    plt.ylabel('Number of Suppliers')
    plt.title('Supplier State Evolution: Switched and Affected Only')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xaxis = plt.gca().xaxis
    plt.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Supplier state (switched & affected) plot saved to {os.path.join(OUTPUT_DIR, filename)}") 