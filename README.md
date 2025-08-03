# Finisterre Supply Chain Agent-Based Modelling (ABM) Project

## Overview

This project implements an **Agent-Based Model (ABM)** to simulate macroeconomic shock propagation (e.g., inflation, tariffs) through Finisterre’s multi-tier sustainable fashion supply chain. The model identifies critical suppliers, assesses supply chain resilience, and analyzes the impact of shocks and switching behavior.

**Key research questions:**
- How do macroeconomic shocks cascade through the supply chain?
- Which suppliers are most critical to supply chain resilience?
- How do switching costs and network structure affect vulnerability?

---

## Features

- **Multi-Tier Supply Chain Network:**  
  Models suppliers, intermediaries, and the final buyer (Knitex 96) as agents in a directed network.
- **Macroeconomic Shock Simulation:**  
  Supports inflation, tariffs, and other shocks, with configurable magnitude, duration, and affected countries.
- **Switching Logic:**  
  Buyers may switch suppliers based on cost increases, with configurable penalties and friction.
- **Shock Propagation:**  
  Shocks propagate through the network, affecting costs and supplier status.
- **Critical Supplier Identification:**  
  Quantifies which suppliers are most critical based on cost impact and network position.
- **Comprehensive Diagnostics:**  
  Tracks supplier status, cost evolution, and at-risk conditions over time.
- **Rich Visualizations:**  
  Includes time series, tier/country impact, and network impact plots with intuitive color coding.

---

## Project Structure

```
Agent Based Modelling/
│
├── Finisterre_ABM_Implementation.py   # Main ABM logic and simulation engine
├── agent_logic.py                     # Agent and supplier behavior logic
├── data_cleaning_and_analysis.py      # Data wrangling and preprocessing scripts
├── dissertation_visualizations.py     # All visualization and plotting functions
├── Data/
│   ├── Finisterre_Trade_Data.xlsx         # Raw trade data
│   └── processed/
│       └── integrated_trade_data.xlsx     # Cleaned, long-format trade data (used in ABM)
├── plots/                            # Output plots and diagnostics
├── sim_test_1.py, sim_test_2.py      # Example simulation scripts
├── README.md                         # Project documentation (this file)
└── requirements.txt                  # Python dependencies
```

---

## Setup and Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   - Ensure your cleaned, integrated trade data is available at `Data/processed/integrated_trade_data.xlsx`.

---

## Running a Simulation

1. **Run a Sample Simulation**
   ```bash
   python sim_test_1.py
   ```
   - This will run a sample simulation and generate output plots in the `plots/` directory.

2. **Visualize Results**
   - Key visualizations include:
     - Landed cost evolution
     - Supplier status over time
     - Tier and country impact
     - **Network impact analysis** (see `sim_test1_results_5_shock_propagation.png`)

---

## Configuration

- **Simulation parameters** (shock magnitude, duration, switching penalty, etc.) can be set in the simulation scripts or passed to the ABM class.
- **Switching logic** and shock propagation are fully configurable for scenario analysis.

---

## Data Flow

1. **Raw trade data** is cleaned and transformed into a long-format table (`integrated_trade_data.xlsx`).
2. **ABM initializes the network** from this data, creating agents and relationships.
3. **Simulation runs** for a specified number of periods, applying shocks and tracking all relevant metrics.
4. **Results and diagnostics** are saved and visualized.

---

## Outputs

- **CSV logs** of simulation results and diagnostics.
- **PNG plots** for all key analyses.
- **Network visualizations** with color-coded impact.

---

## Network Impact Visualization (NEW)

The simulation now includes improved network impact visualizations:

- **Colormap:** Uses the OrRd (Orange-Red) colormap for better visibility of impacted nodes.
- **Zero-Impact Nodes:** All suppliers with zero cumulative cost impact are shown as pure green, making it easy to spot unaffected nodes.
- **Impacted Nodes:** Suppliers with nonzero impact are colored from orange to red, with color intensity reflecting the magnitude of their cumulative cost impact.
- **Dual Plots:** Both cumulative absolute cost impact and log-scaled cumulative normalized cost impact are visualized side-by-side for comprehensive analysis.
- **Interpretation:** This approach makes it easy to distinguish which suppliers absorbed shocks or switching costs, and to what extent, across the entire supply chain network.

See the `sim_test1_results_5_shock_propagation.png` output for an example.

---

## Contributing

Contributions, bug reports, and suggestions are welcome! Please open an issue or submit a pull request.

---

## License

This project is for academic research and educational use. See the LICENSE file for details.