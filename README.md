# Finisterre Supply Chain Agent-Based Modelling (ABM) Project

## Dissertation Project Overview

This repository contains the complete implementation of an **Agent-Based Model (ABM)** for simulating macroeconomic shock propagation through Finisterre's multi-tier sustainable fashion supply chain. This project was developed as part of a UCL dissertation examining supply chain resilience and vulnerability to macroeconomic shocks.

**Research Questions:**
- How do macroeconomic shocks (inflation, tariffs) cascade through multi-tier supply chains?
- Which suppliers are most critical to supply chain resilience?
- How do switching costs and network structure affect vulnerability?
- What are the implications for supply chain risk management?

---

## Key Features

### 1. Multi-Tier Supply Chain Network
- Models suppliers, intermediaries, and final buyer as agents in a directed network
- Supports complex multi-tier relationships with configurable switching costs
- Tracks supplier status, costs, and vulnerability metrics over time

### 2. Macroeconomic Shock Simulation
- **Inflation Shocks:** Country-specific inflation rate changes
- **Tariff Shocks:** Import/export tariff modifications
- **Configurable Parameters:** Magnitude, duration, affected countries
- **Shock Propagation:** Realistic cascading effects through the network

### 3. Switching Logic & Resilience Analysis
- Buyers can switch suppliers based on cost increases
- Configurable switching penalties and friction
- Critical supplier identification based on network position and cost impact
- Vulnerability assessment across different scenarios

### 4. Comprehensive Diagnostics & Visualization
- Time series analysis of landed costs and supplier status
- Network impact visualization with color-coded severity
- Tier and country-level impact analysis
- Critical supplier heatmaps and vulnerability assessments

---

## Repository Structure

```
supply-chain-abm--dissertation/
│
├── Core ABM Implementation
│   ├── Finisterre_ABM_Implementation.py   # Main ABM simulation engine
│   ├── agent_logic.py                     # Agent behavior and supplier logic
│   └── sim_test_1.py                     # Example simulation with comprehensive analysis
│
├── Data Processing & Analysis
│   ├── data_cleaning_and_analysis.py      # Data preprocessing and validation
│   └── supply_chain_eda.py               # Exploratory data analysis
│
├── Visualization & Outputs
│   └── model_visuals.py                  # Comprehensive visualization suite
│
├── Configuration & Setup
│   ├── requirements.txt                   # Python dependencies
│   ├── setup.py                          # Project setup
│   └── README.md                         # This documentation
│
└── Documentation
    ├── LICENSE                           # MIT License for academic use
    └── .gitignore                        # Excludes sensitive data and outputs
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/owenkeyworth/supply-chain-abm--dissertation.git
cd supply-chain-abm--dissertation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import mesa, pandas, numpy, matplotlib, networkx; print('All dependencies installed successfully!')"
```

---

## Running Simulations

### Quick Start - Example Simulation
```bash
python sim_test_1.py
```

This will:
- Load the integrated trade data
- Run a comprehensive simulation with inflation and tariff shocks
- Generate visualizations in the `model outputs/` directory
- Save diagnostic logs and metrics

### Custom Simulation
```python
from Finisterre_ABM_Implementation import SupplyChainABM
import pandas as pd

# Load data
trade_data = pd.read_excel('Data/processed/integrated_trade_data.xlsx')

# Initialize ABM
abm = SupplyChainABM(
    trade_data=trade_data,
    inflation_shock_magnitude=0.15,  # 15% inflation shock
    tariff_shock_magnitude=0.10,     # 10% tariff increase
    switching_penalty=0.05,          # 5% switching cost
    simulation_periods=12            # 12 months
)

# Run simulation
results = abm.run_simulation()

# Generate visualizations
abm.generate_visualizations()
```

---

## Model Parameters & Configuration

### Core Parameters
- **Inflation Shock Magnitude:** Percentage increase in inflation rates
- **Tariff Shock Magnitude:** Percentage increase in import/export tariffs
- **Switching Penalty:** Cost penalty for switching suppliers
- **Simulation Periods:** Number of time steps to simulate

### Network Parameters
- **Supplier Relationships:** Based on actual trade data
- **Tier Structure:** Multi-tier supply chain with intermediaries
- **Cost Structure:** Includes production, transportation, and switching costs

### Shock Configuration
- **Affected Countries:** Configurable list of countries experiencing shocks
- **Shock Duration:** Time period over which shocks are applied
- **Shock Type:** Inflation, tariffs, or combined shocks

---

## Outputs & Analysis

### 1. Time Series Analysis
- Landed cost evolution over time
- Supplier status changes (active/inactive/at-risk)
- Tier-level impact analysis
- Country-level vulnerability assessment

### 2. Network Impact Visualization
- **Color-coded Network Maps:** Green (unaffected) to Red (highly impacted)
- **Cumulative Impact Analysis:** Total cost impact across the network
- **Critical Supplier Identification:** Suppliers with highest network centrality

### 3. Diagnostic Logs
- Detailed simulation logs in CSV format
- Supplier-level diagnostics and metrics
- Network topology analysis

### 4. Key Metrics
- **Cost Impact:** Total cost increase due to shocks
- **Supplier Vulnerability:** Percentage of suppliers at risk
- **Network Resilience:** Ability to absorb shocks without disruption
- **Switching Behavior:** Frequency and patterns of supplier switching

---

## Data Requirements

### Input Data Format
The model requires trade data in the following format:
- **Supplier Code:** Unique identifier for each supplier
- **Country:** Supplier country location
- **Tier:** Supply chain tier (1, 2, 3, etc.)
- **Trade Value:** Monetary value of trade relationships
- **Cost Structure:** Production and transportation costs

### Data Processing
The `data_cleaning_and_analysis.py` script handles:
- Data validation and cleaning
- Missing value imputation
- Network structure validation
- Cost structure estimation

---

## Model Validation

### Validation Components
1. **Data Validation:** Ensures data quality and completeness
2. **Network Validation:** Verifies supply chain structure consistency
3. **Parameter Sensitivity:** Tests model robustness to parameter changes
4. **Scenario Analysis:** Compares different shock scenarios

### Validation Scripts
- `validation_improvements.py`: Comprehensive validation suite
- Diagnostic outputs in `Diagnostic Outputs/` directory

---

## Academic Context

### Research Methodology
- **Agent-Based Modelling:** Individual supplier and buyer behavior
- **Network Analysis:** Supply chain topology and relationships
- **Scenario Analysis:** Multiple shock scenarios and sensitivity testing
- **Empirical Validation:** Based on real trade data from Finisterre

### Key Contributions
1. **Multi-tier Supply Chain Modelling:** Extends ABM to complex supply networks
2. **Macroeconomic Shock Integration:** Realistic shock propagation mechanisms
3. **Switching Behavior Analysis:** Supplier switching under cost pressure
4. **Critical Supplier Identification:** Network-based vulnerability assessment

### Literature Review
See `literature_review_enhancement.md` for comprehensive literature review covering:
- Supply chain resilience literature
- Agent-based modelling in supply chains
- Macroeconomic shock propagation
- Network analysis in supply chain management

---

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Data Loading Issues**
   - Ensure `Data/processed/integrated_trade_data.xlsx` exists
   - Check data format matches expected structure

3. **Memory Issues**
   - Reduce simulation periods for large networks
   - Use smaller datasets for testing

### Getting Help
- Check the diagnostic logs in `Diagnostic Outputs/`
- Review the model assumptions document
- Examine the example simulation in `sim_test_1.py`

---

## Citation & Academic Use

This project was developed as part of a UCL dissertation on supply chain resilience and macroeconomic shock propagation. For academic use, please cite:

```
[Your Name] (2024). "Agent-Based Modelling of Macroeconomic Shock Propagation 
in Multi-Tier Supply Chains: A Case Study of Finisterre's Sustainable 
Fashion Supply Chain." UCL Dissertation.
```

---

## License

This project is for academic research and educational use. The code is provided as-is for dissertation review purposes.

---

## Contact

For questions about this implementation or the underlying research, please contact:
- **Student:** [Your Name]
- **Institution:** University College London
- **Department:** [Your Department]
- **Supervisor:** [Your Supervisor's Name]

---

*This repository contains the complete implementation of an Agent-Based Model for supply chain resilience analysis, developed as part of a UCL dissertation project.*