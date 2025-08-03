import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import PowerNorm

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_excel("Data/processed/integrated_trade_data.xlsx")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Define inflation columns ONCE, right after loading data
inflation_cols = [col for col in df.columns if 'inflation_' in col]

# Capitalize country names in dataframe and mapping table
if 'supplier_country' in df.columns:
    df['supplier_country'] = df['supplier_country'].str.title()
if 'importer_country' in df.columns:
    df['importer_country'] = df['importer_country'].str.title()

# 1. Missing Data Analysis
plt.figure(figsize=(12, 8))
missing_percent = df.isnull().mean().sort_values(ascending=False) * 100
sns.barplot(x=missing_percent.values, y=missing_percent.index)
plt.xlabel("% Missing")
plt.title("Missing Data by Column")
plt.tight_layout()
plt.savefig("plots/missing_data.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Number of Relationships by Tier (Figure 1a)
plt.figure(figsize=(8, 6))
relationship_counts = df['hop_level'].value_counts().sort_index()
sns.barplot(x=relationship_counts.index, y=relationship_counts.values, palette='mako')
plt.title("Number of Relationships by Tier")
plt.xlabel("Tier (Hop Level)")
plt.ylabel("Number of Relationships")
plt.tight_layout()
plt.savefig("plots/tier_relationship_counts.png", dpi=300)
plt.close()

# 3. Number of Unique Suppliers by Tier (Figure 1b)
plt.figure(figsize=(8, 6))
unique_suppliers_per_tier = df.groupby('hop_level')['supplier'].nunique()
sns.barplot(x=unique_suppliers_per_tier.index, y=unique_suppliers_per_tier.values, palette='crest')
plt.title("Number of Unique Suppliers by Tier")
plt.xlabel("Tier (Hop Level)")
plt.ylabel("Number of Unique Suppliers")
plt.tight_layout()
plt.savefig("plots/tier_unique_supplier_counts.png", dpi=300)
plt.close()

# Debug: Print unique supplier names in Tier 2
print("\n--- Unique Supplier Check for Tier 2 ---")
tier2_suppliers = df[df['hop_level'] == 2]['supplier'].unique()
print(f"Number of unique suppliers in Tier 2: {len(tier2_suppliers)}")
print("Sample of Tier 2 suppliers:", tier2_suppliers[:30])  # Print first 30 for inspection

# =============================
# 3. Hierarchical Supply Chain Network (with Knitex as end node)
# =============================

import string

# 1. Assign two-letter codes to suppliers
suppliers = sorted(set(df['supplier']))
def make_codes(n):
    alphabet = string.ascii_uppercase
    codes = []
    for i in range(n):
        codes.append(alphabet[i//26] + alphabet[i%26])
    return codes
supplier_codes = make_codes(len(suppliers))
supplier_code_map = dict(zip(suppliers, supplier_codes))

# 2. Create a mapping table and save as CSV and Markdown
supplier_country_map = df[['supplier', 'supplier_country', 'hop_level']].drop_duplicates(subset=['supplier']).set_index('supplier')
mapping_table = pd.DataFrame({
    'Code': [supplier_code_map[s] for s in suppliers],
    'Supplier': suppliers,
    'Country': [supplier_country_map.loc[s, 'supplier_country'] for s in suppliers],
    'Hop Level': [supplier_country_map.loc[s, 'hop_level'] for s in suppliers]
})
mapping_table.to_csv('plots/supplier_code_table.csv', index=False)
mapping_table.to_markdown('plots/supplier_code_table.md', index=False)

# 3. Prepare the network graph (only suppliers as nodes)
G = nx.DiGraph()
for _, row in df.iterrows():
    if row['supplier'] in supplier_code_map and row['importer'] in supplier_code_map:
        G.add_edge(row['supplier'], row['importer'], value_usd=row['value_usd'])

# Assign a unique color to each country (all countries, not just first 10)
countries = sorted(mapping_table['Country'].unique())
country_palette = sns.color_palette('tab20', n_colors=len(countries))
country_color_map = {country: country_palette[i] for i, country in enumerate(countries)}

# Group nodes by hop level (tier)
level_nodes = {}
for supplier in suppliers:
    level = supplier_country_map.loc[supplier, 'hop_level']
    level_nodes.setdefault(level, []).append(supplier)

# Assign positions: y = -level, x = evenly spaced within level
pos = {}
x_spacing = 2.0
y_spacing = -2.5
for level, nodes in sorted(level_nodes.items()):
    n_nodes = len(nodes)
    for i, node in enumerate(sorted(nodes)):
        pos[node] = (i * x_spacing - (n_nodes-1)*x_spacing/2, level * y_spacing)

# Node colors by country
node_colors = [country_color_map[mapping_table.set_index('Supplier').loc[n, 'Country']] for n in G.nodes()]
node_sizes = [500 for _ in G.nodes()]

# Edge widths by trade value (log scale, range 0.5-12)
trade_vals = [d['value_usd'] for u, v, d in G.edges(data=True)]
if trade_vals:
    min_val, max_val = min(trade_vals), max(trade_vals)
    def scale_width(val):
        log_min = np.log10(min_val+1)
        log_max = np.log10(max_val+1)
        log_val = np.log10(val+1)
        return 0.5 + 11.5 * (log_val - log_min) / (log_max - log_min) if log_max > log_min else 3
    edge_weights = [scale_width(d['value_usd']) for u, v, d in G.edges(data=True)]
else:
    edge_weights = [1 for _ in G.edges()]

# Node labels: two-letter code for suppliers
labels = {n: supplier_code_map[n] for n in G.nodes()}

# === Add Knitex 96 as the end node (final importer) ===
knitex_name = "knitex 96"
knitex_matches = df[df['importer'].str.lower().str.strip() == knitex_name]
if not knitex_matches.empty:
    knitex_country = knitex_matches.iloc[0]['importer_country']
else:
    raise ValueError(f"No importer found matching '{knitex_name}'. Check the spelling and casing in your data.")
knitex_code = "ZZ"  # Use a unique code not in supplier_codes

if knitex_name not in supplier_code_map:
    suppliers.append(knitex_name)
    supplier_code_map[knitex_name] = knitex_code

if knitex_name not in mapping_table['Supplier'].values:
    mapping_table = pd.concat([
        mapping_table,
        pd.DataFrame([{
            'Code': knitex_code,
            'Supplier': knitex_name,
            'Country': knitex_country,
            'Hop Level': 0
        }])
    ], ignore_index=True)

G.add_node(knitex_name)
for _, row in df[df['hop_level'] == 1].iterrows():
    supplier = row['supplier']
    if supplier in supplier_code_map:
        G.add_edge(supplier, knitex_name, value_usd=row['value_usd'])

pos[knitex_name] = (0, 0)
for level, nodes in sorted(level_nodes.items()):
    n_nodes = len(nodes)
    for i, node in enumerate(sorted(nodes)):
        pos[node] = (i * x_spacing - (n_nodes-1)*x_spacing/2, (level+1) * y_spacing)  # +1 to shift down

country_color_map[knitex_country] = (0.2, 0.2, 0.2)  # dark gray for focal node
node_colors = [country_color_map[mapping_table.set_index('Supplier').loc[n, 'Country']] if n != knitex_name else (0.2, 0.2, 0.2) for n in G.nodes()]
node_sizes = [500 if n != knitex_name else 900 for n in G.nodes()]
labels[knitex_name] = knitex_code

legend_elements = [mpatches.Patch(facecolor=country_color_map[c], edgecolor='black', label=c.title()) for c in countries]
legend_elements.append(mpatches.Patch(facecolor=(0.2,0.2,0.2), edgecolor='black', label="Knitex 96 JSC (Focal)"))

plt.figure(figsize=(22, 14))
nx.draw_networkx_edges(G, pos, alpha=0.18, width=edge_weights, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=0.5)
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')
plt.title("Hierarchical Supply Chain Network by Tier and Trade Volume\n(Nodes colored by country, labeled by supplier code, edge width by trade value)", fontsize=18)
plt.axis('off')
plt.legend(handles=legend_elements, title="Country", loc='lower left', fontsize=12, title_fontsize=13, frameon=True)
plt.tight_layout()
plt.savefig("plots/trade_network_hierarchical_suppliercode.png", dpi=400)
plt.close()

# 4. K-Core Decomposition (Country Level)
G_kcore = nx.from_pandas_edgelist(df, 'supplier_country', 'importer_country')
core_numbers = nx.core_number(G_kcore)
unique_cores = sorted(set(core_numbers.values()))
color_list = plt.cm.viridis(np.linspace(0, 1, len(unique_cores)))
color_map = {core: color for core, color in zip(unique_cores, color_list)}
node_colors = [color_map[core_numbers[node]] for node in G_kcore.nodes()]
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G_kcore, seed=42)
nx.draw_networkx_nodes(
    G_kcore, pos,
    node_color=node_colors,
    node_size=600,
    edgecolors='black',
    linewidths=0.7
)
nx.draw_networkx_edges(G_kcore, pos, alpha=0.18, edge_color='gray', width=0.5)
for node, (x, y) in pos.items():
    plt.text(x, y - 0.06, node.title(), fontsize=16, ha='center', va='top',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1'))
legend_handles = [mpatches.Patch(color=color_map[k], label=f'k-core {k}') for k in sorted(color_map)]
plt.legend(handles=legend_handles, title="k-core Level", loc='lower left', fontsize=16, title_fontsize=13, frameon=True)
plt.title("Supply Chain Network k-core Decomposition (Countries as Nodes)", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("plots/supply_chain_kcore.png", dpi=400)
plt.close()

# 5. Top 10 Countries by Betweenness Centrality (Figure 4)
G = nx.from_pandas_edgelist(df, 'supplier_country', 'importer_country')
betweenness = nx.betweenness_centrality(G)
top_central = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
countries, scores = zip(*top_central)
plt.figure(figsize=(10, 6))
sns.barplot(x=list(scores), y=list(countries), palette="viridis")
plt.xlabel("Betweenness Centrality")
plt.title("Top 10 Countries by Betweenness Centrality")
plt.tight_layout()
plt.savefig("plots/top_central_countries.png", dpi=300)
plt.close()

# 6. Histogram of Supplier Profit Margins (Figure 5)
# Get unique suppliers with their profit margins to avoid relationship weighting
unique_suppliers = df[['supplier', 'profit_margin']].drop_duplicates(subset=['supplier'])

plt.figure(figsize=(12, 6))
# Use more bins for better resolution and set explicit bin edges
profit_margins_pct = unique_suppliers['profit_margin'].dropna() * 100
min_margin = profit_margins_pct.min()
max_margin = profit_margins_pct.max()
bins = np.arange(min_margin, max_margin + 1, 0.5)  # 0.5% wide bins from min to max

sns.histplot(profit_margins_pct, bins=bins, kde=True, color='steelblue', alpha=0.7)
plt.axvline(profit_margins_pct.mean(), color='red', linestyle='--', label='Mean')
plt.axvline(profit_margins_pct.median(), color='green', linestyle=':', label='Median')
plt.title("Distribution of Supplier Profit Margins")
plt.xlabel("Profit Margin (%)")
plt.ylabel("Number of Suppliers")
plt.xticks(np.arange(int(min_margin), int(max_margin) + 2, 1))  # Show every 1% from min to max
plt.legend()
plt.tight_layout()
plt.savefig("plots/profit_margin_distribution.png", dpi=300)
plt.close()

# 7. Heatmap of Forecasted Inflation by Country and Year (Figure 6)
# Only keep inflation columns for 2025–2030
years = [2025, 2026, 2027, 2028, 2029, 2030]
inflation_cols = [f'inflation_{year}' for year in years]
heatmap_data = df[['supplier_country'] + inflation_cols].drop_duplicates('supplier_country').set_index('supplier_country')

# Sort countries alphabetically
heatmap_data = heatmap_data.sort_index()

# Convert to percent for display and plotting
heatmap_data_percent = heatmap_data * 100

# Create annotation labels with percent sign
annot_labels = heatmap_data_percent.applymap(lambda x: f"{x:.1f}%")

from matplotlib.colors import PowerNorm

plt.figure(figsize=(16, 8))
ax = sns.heatmap(
    heatmap_data_percent,
    annot=annot_labels,
    fmt="",
    cmap="OrRd",
    norm=PowerNorm(gamma=0.5, vmin=0, vmax=15),
    cbar_kws={'label': 'Inflation Rate (%)'},
    mask=None
)
# Set x-axis labels to just the year
ax.set_xticklabels([str(year) for year in years])

plt.title("Forecasted Inflation (2025–2030) by Country")
plt.xlabel("Year")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("plots/inflation_heatmap_top15.png", dpi=150)
plt.close()

# 8. Boxplot of Log-Scaled Trade Values by Tier (Figure 7)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='hop_level', y='value_usd')
plt.yscale('log')
plt.xlabel("Supply Chain Tier (Hop Level)")
plt.ylabel("Trade Value (USD, Log Scale)")
plt.title("Boxplot of Log-Scaled Trade Values by Tier")
plt.tight_layout()
plt.savefig("plots/trade_value_by_tier.png", dpi=300)
plt.close()

# --- Disabled: Old/Unwanted Plots for Clarity and Quality ---
# The following plots have been commented out as per user request:
# - inflation_projection_heatmap.png
# - tier_distribution.png (old version, not the new unique supplier plots)

# # Disabled: Inflation Projection Heatmap
# plt.figure(figsize=(16, 8))
# inflation_matrix = df.pivot_table(index='supplier_country', values=inflation_cols)
# sns.heatmap(inflation_matrix, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Inflation Rate (%)'})
# plt.title('Forecasted Inflation (2025–2030) by Country')
# plt.xlabel('Year')
# plt.ylabel('Country')
# plt.tight_layout()
# plt.savefig('plots/inflation_projection_heatmap.png', dpi=300)
# plt.close()

# # Disabled: Old Tier Distribution Plot
# plt.figure(figsize=(8, 6))
# tier_counts = df['hop_level'].value_counts().sort_index()
# sns.barplot(x=tier_counts.index, y=tier_counts.values, palette='mako')
# plt.title("Distribution of Supplier Counts by Tier")
# plt.xlabel("Tier (Hop Level)")
# plt.ylabel("Number of Unique Suppliers")
# plt.tight_layout()
# plt.savefig("plots/tier_distribution.png", dpi=300)
# plt.close()