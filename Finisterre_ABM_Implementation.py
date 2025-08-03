from __future__ import annotations
"""
Finisterre Supply Chain Agent-Based Model (ABM) Implementation
Macroeconomic Shock Propagation Analysis

This ABM simulates macroeconomic shock propagation (inflation, tariffs) through 
Finisterre's multi-tier sustainable fashion supply chain to identify critical 
suppliers and assess supply chain resilience.

Primary Research Question: How do macroeconomic shocks cascade through Finisterre's 
supply chain, and which suppliers are critical to the supply chain?

Based on the detailed project plan with specific focus on:
- Price elasticity in sustainable fashion market
- Low switching costs in fashion industry
- Product Importance Ratio (PIR)
- Shock duration parameters
- Agent switching decision rules
"""

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================
# Configure simulation duration here (12, 18, or 24 periods recommended)
SIMULATION_DURATION = 18  # Change this value to set simulation length

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import random
import warnings
import sys
import collections
from agent_logic import (
    SupplierAgent,
    SupplierCharacteristics,
    SupplierStatus,
    ShockType,
    MacroShock
)
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class TariffPhase:
    """Represents a single phase of tariff application"""
    start_period: int
    end_period: int
    magnitude: float  # tariff rate as percentage (e.g., 0.10 for 10%)
    
    def is_active(self, period: int) -> bool:
        """Check if this tariff phase is active in the given period"""
        return self.start_period <= period <= self.end_period

@dataclass
class TariffSchedule:
    """Defines a schedule of tariff changes over time"""
    phases: List[TariffPhase]
    target_countries: List[str]
    
    def get_tariff_rate(self, period: int) -> float:
        """
        Get the tariff rate for a specific period.
        Returns 0.0 if no tariff phase is active.
        """
        for phase in self.phases:
            if phase.is_active(period):
                return phase.magnitude
        return 0.0
    
    def get_active_phase(self, period: int) -> Optional[TariffPhase]:
        """Get the currently active tariff phase, if any"""
        for phase in self.phases:
            if phase.is_active(period):
                return phase
        return None
    
    def is_any_phase_active(self, period: int) -> bool:
        """Check if any tariff phase is active in the given period"""
        return self.get_tariff_rate(period) > 0.0

class FinisterreSupplyChainABM:
    """Agent-Based Model for Finisterre Supply Chain Macroeconomic Analysis"""
    
    def __init__(self, trade_df: pd.DataFrame, switching_cost_factor: float = 0.2, switching_friction_penalty: float = 0.2, cost_increase_threshold: float = 0.1, disruption_duration: int = 3, product_importance_ratio: float = 0.45, switching_penalty: float = 0.1, config: dict = None):
        """
        Initialize the Finisterre supply chain ABM.
        
        Args:
            trade_df: DataFrame containing trade relationship data
            switching_cost_factor: Factor for switching cost calculation (default 0.2)
            switching_friction_penalty: Temporary cost penalty after switching (default 0.2)
            cost_increase_threshold: Threshold for cost increase to trigger switching (default 0.1)
            disruption_duration: Number of periods for disruption cost calculation (default 3)
            product_importance_ratio: PIR for shock propagation (default 0.45)
        """
        self.agents = {}
        self.network = nx.DiGraph()
        self.active_shocks = []
        self.current_period = 0
        self.switching_cost_factor = switching_cost_factor
        self.switching_friction_penalty = switching_friction_penalty
        self.cost_increase_threshold = cost_increase_threshold
        self.disruption_duration = disruption_duration
        self.pir = product_importance_ratio
        self.switching_penalty = switching_penalty
        
        # Store original pre-shock base costs for accurate baseline reporting
        self.original_base_costs = {}
        self.delayed_shocks = collections.defaultdict(float)  # For temporal lag
        # Config toggles for scenario features
        self.config = config or {}
        self.config['switching_cost_factor'] = switching_cost_factor
        self.config['switching_friction_penalty'] = switching_friction_penalty
        self.config['cost_increase_threshold'] = cost_increase_threshold
        self.config['disruption_duration'] = disruption_duration
        self.config['switching_penalty'] = switching_penalty
        
        # Enhanced configuration defaults for dynamic behavior
        if 'base_switching_probability' not in self.config:
            self.config['base_switching_probability'] = 0.3
        if 'shock_persistence_factor' not in self.config:
            self.config['shock_persistence_factor'] = 0.8
        if 'recovery_rate' not in self.config:
            self.config['recovery_rate'] = 0.02
        if 'shock_accumulation_rate' not in self.config:
            self.config['shock_accumulation_rate'] = 0.1
        if 'max_shock_accumulation' not in self.config:
            self.config['max_shock_accumulation'] = 0.5
        if 'shock_inheritance_factor' not in self.config:
            self.config['shock_inheritance_factor'] = 0.3
        if 'propagation_inheritance_factor' not in self.config:
            self.config['propagation_inheritance_factor'] = 0.7
            
        # Tier-dependent switching costs
        self.tier_switch_costs = {
            1: 0.15,  # Tier 1 (intermediary)
            2: 0.25,  # Tier 2 (manufacturer)
            3: 0.30,  # Tier 3 (components)
            4: 0.40,  # Tier 4 (materials)
            5: 0.50   # Tier 5 (raw materials)
        }
        
        # Initialize network from trade data
        self.initialize_network_from_data(trade_df)
        self.original_supplier_ids = set([aid for aid in self.agents if not str(aid).startswith('Supplier_switch_')])
        
        # Get actual tier range from data
        self.tier_range = self._get_tier_range()
        print(f"Detected tier range: {self.tier_range}")
    
    def initialize_network_from_data(self, trade_df: pd.DataFrame):
        """Initialize network from trade data"""
        print("Initializing network from trade data...")
        # Create supplier agents only from supplier rows
        suppliers = trade_df[['supplier', 'supplier_country', 'hop_level', 'profit_margin']].rename(
            columns={'supplier': 'id', 'supplier_country': 'country', 'hop_level': 'tier'}
        ).drop_duplicates(subset=['id'])
        for _, row in suppliers.iterrows():
            profit_margin = row['profit_margin'] if 'profit_margin' in row and not pd.isna(row['profit_margin']) else None
            characteristics = SupplierCharacteristics(
                id=row['id'],
                country=row['country'],
                tier=row['tier'],
                product_importance_ratio=self.pir,
                profit_margin=profit_margin
            )
            self.agents[row['id']] = SupplierAgent(
                characteristics=characteristics,
                base_cost=1.0,  # Will be updated from trade data
                abm=self
            )
            self.network.add_node(row['id'])
        # Add Knitex 96 as a special agent if not already present
        if 'knitex 96' not in self.agents:
            importer_row = trade_df[trade_df['importer'].str.lower() == 'knitex 96'].iloc[0]
            self.agents['knitex 96'] = SupplierAgent(
                characteristics=SupplierCharacteristics(
                    id='knitex 96',
                    country=importer_row['importer_country'],
                    tier=0,
                    profit_margin=0.214,  # Ensure float, not None
                    product_importance_ratio=0.45
                ),
                base_cost=0.0,
                abm=self
            )
        # Set Knitex 96 profit margin and adjusted margin to 21.4%
        knitex_agent = self.agents.get('knitex 96')
        if knitex_agent:
            knitex_agent.original_margin = 0.214
            knitex_agent.profit_margin = 0.214
            knitex_agent.adjusted_margin = 0.214
        else:
            print("DEBUG: Knitex 96 agent not found in agents dictionary!")
        
        # DEBUG: Print profit margin for knitex 96 after agent creation
        knitex_agent = self.agents.get('knitex 96')
        if knitex_agent:
            print(f"DEBUG: Knitex 96 agent profit margin: {knitex_agent.characteristics.profit_margin}")
        else:
            print("DEBUG: Knitex 96 agent not found in agents dictionary!")
        
        # Add relationships and update costs
        for _, row in trade_df.iterrows():
            supplier_id = row['supplier']
            importer_id = row['importer']
            
            # Add relationship
            self.network.add_edge(supplier_id, importer_id)
            
            # Update supplier's buyers list
            if supplier_id in self.agents:
                self.agents[supplier_id].buyers.add(importer_id)
            
            # Update importer's suppliers list
            if importer_id in self.agents:
                self.agents[importer_id].suppliers.add(supplier_id)
                # print(f"Added supplier {supplier_id} to {importer_id}")
            # print(f'{supplier_id} buyers: {self.agents[supplier_id].buyers}')
            # Update costs and margins if available
            if supplier_id in self.agents:
                supplier = self.agents[supplier_id]
                # Per plan (3.1), base_cost is value_usd, not unit_cost
                if 'value_usd' in row and not pd.isna(row['value_usd']):
                    supplier.base_cost = row['value_usd']
                else:
                    supplier.base_cost = 1.0
                # ✅ Fix: Initialize current_cost explicitly
                supplier.current_cost = supplier.base_cost
                
                # Store the original pre-shock base cost for accurate baseline reporting
                self.original_base_costs[supplier_id] = supplier.base_cost
                
                if 'profit_margin' in row and not pd.isna(row['profit_margin']):
                    supplier.characteristics.profit_margin = row['profit_margin']
        
        print(f"Network initialized with {len(self.agents)} agents and {self.network.number_of_edges()} relationships")
    
    def _get_tier_range(self) -> List[int]:
        """Get the actual tier range from the data"""
        tiers = set()
        for agent in self.agents.values():
            tiers.add(agent.characteristics.tier)
        return sorted(list(tiers))
    
    def apply_macro_shock(self, shock: MacroShock):
        """Adds a shock to the list of active shocks for the simulation."""
        self.active_shocks.append(shock)
        print(f"\nApplied shock: {shock.shock_type.value} of {shock.magnitude:.2%} on {shock.affected_countries} for {shock.duration} periods, starting in period {shock.start_period}.")
    
    def find_alternative_suppliers(self, current_supplier_id: str) -> List[str]:
        """
        Finds alternative suppliers for a given switched supplier.
        
        Args:
            current_supplier_id: The ID of the supplier to be replaced.

        Returns:
            A list of IDs of potential alternative suppliers.
        """
        if current_supplier_id not in self.agents:
            return []

        current_supplier = self.agents[current_supplier_id]
        current_tier = current_supplier.characteristics.tier
        
        alternatives = []
        for agent_id, agent in self.agents.items():
            if (agent.characteristics.tier == current_tier and 
                agent.status == SupplierStatus.OPERATIONAL and
                agent_id != current_supplier_id):
                alternatives.append(agent_id)
        
        return alternatives

    def identify_critical_suppliers(self) -> pd.DataFrame:
        """
        Identify critical suppliers: only those who are at risk or switched.
        Returns a DataFrame with Supplier ID, Tier, Country, Status (Switched/At Risk), # Buyers Switched, Cost Impact, and list of direct importers (buyers).
        """
        records = []
        abs_cost_impacts = []
        base_costs = []
        cumulative_cost_impacts = []
        cumulative_norm_cost_impacts = []
        for agent_id, agent in self.agents.items():
            # Determine status for reporting
            if agent.status == SupplierStatus.SWITCHED:
                status = 'Switched'
            elif agent.affected:
                status = 'Affected'
            else:
                status = 'Operational'
            num_buyers_switched = len(getattr(agent, 'buyers_switched_away', []))
            cost_impact = agent.current_cost - agent.base_cost
            base_cost = agent.base_cost if agent.base_cost else 1.0
            abs_cost_impacts.append(cost_impact)
            base_costs.append(base_cost)
            buyers = list(agent.buyers) if hasattr(agent, 'buyers') else []
            # Cumulative cost impact
            if hasattr(agent, 'period_costs') and hasattr(agent, 'period_base_costs'):
                total_cost_impact = sum(agent.period_costs) - sum(agent.period_base_costs)
                if sum(agent.period_base_costs) != 0:
                    total_norm_cost_impact = total_cost_impact / sum(agent.period_base_costs)
                else:
                    total_norm_cost_impact = 0.0
            else:
                total_cost_impact = 0.0
                total_norm_cost_impact = 0.0
            cumulative_cost_impacts.append(total_cost_impact)
            cumulative_norm_cost_impacts.append(total_norm_cost_impact)
            records.append({
                'Supplier ID': agent_id,
                'Tier': agent.characteristics.tier,
                'Country': agent.characteristics.country,
                'Status': status,
                '# Buyers Switched': num_buyers_switched,
                'Cost Impact': cost_impact,
                'Base Cost': base_cost,
                'Direct Importers': buyers,
                'Cumulative Cost Impact': total_cost_impact,
                'Cumulative Normalized Cost Impact': total_norm_cost_impact
            })
        df = pd.DataFrame(records)
        if not df.empty:
            # Normalized cost impact
            df['Normalized Cost Impact'] = df['Cost Impact'] / df['Base Cost'].replace(0, 1)
            # Debug: Print distribution and quantiles
            print('--- Cost Impact Distribution ---')
            print(df['Cost Impact'].describe())
            print('--- Normalized Cost Impact Distribution ---')
            print(df['Normalized Cost Impact'].describe())
            for metric in ['Cost Impact', 'Normalized Cost Impact']:
                q1 = df[metric].quantile(0.25)
                q3 = df[metric].quantile(0.75)
                print(f'{metric} q1 (25th):', q1, '| q3 (75th):', q3)
                print(df[['Supplier ID', metric]].sort_values(by=metric))
                def classify(val):
                    if val >= q3:
                        return 'High'
                    elif val < q1:
                        return 'Low'
                    else:
                        return 'Medium'
                df[f'{metric} Level'] = df[metric].apply(classify)
            df = df.sort_values(['Status', 'Cost Impact', '# Buyers Switched'], ascending=[True, False, False])
        return df

    def run_comprehensive_simulation(self, periods: int) -> pd.DataFrame:
        """
        Run the comprehensive ABM simulation with time-decaying shocks.
        """
        print(f"\nStarting comprehensive simulation with time-decaying shocks...")
        simulation_data = []
        all_diagnostics = []
        # Initialize per-period cost tracking for each agent
        for agent in self.agents.values():
            agent.period_costs = []
            agent.period_base_costs = []
        for period in range(1, periods + 1):
            self.current_period = period
            # Phase 1: Apply direct shocks to affected suppliers (with time decay)
            for agent in self.agents.values():
                agent.apply_direct_shock(self.active_shocks, period)
            # Phase 2: Update agent states
            for agent in self.agents.values():
                agent.update_period(period)
                # Track per-period costs
                agent.period_costs.append(agent.current_cost)
                agent.period_base_costs.append(agent.base_cost)
                # Collect diagnostics for this period
                if hasattr(agent, 'diagnostics_log') and agent.diagnostics_log:
                    all_diagnostics.append(agent.diagnostics_log[-1])

            # Phase 3: Propagate cost increases through trade links
            for tier in self._get_tier_order():
                for agent in self.agents.values():
                    if agent.characteristics.tier == tier:
                        agent.propagate_cost_increase()

            # Phase 4: Enhanced switching decisions with cooldown and hysteresis
            agent_ids = list(self.agents.keys())
            for agent_id in agent_ids:
                if agent_id in self.agents:  # Check if agent still exists
                    agent = self.agents[agent_id]
                    
                    # Enhanced staggering logic
                    if self.config.get('stagger_switching_by_tier', True):
                        # More sophisticated tier-based staggering
                        tier_stagger = (agent.characteristics.tier + self.current_period) % 3
                        if tier_stagger != 0:
                            continue
                    elif self.config.get('stagger_switching_random', False):
                        # Random staggering with probability based on shock severity
                        shock_severity = agent.applied_shock if hasattr(agent, 'applied_shock') else 0
                        stagger_prob = max(0.3, 1 - shock_severity)  # Higher shock = less staggering
                        if random.random() > stagger_prob:
                            continue
                    
                    agent.decide_and_switch_suppliers()

            # Phase 5: Collect enhanced data for this period
            period_data = self._collect_period_data()
            simulation_data.append(period_data)
            
            # Progress indicator
            if period % 10 == 0:
                print(f"Completed period {period}/{periods}")
        
        # Write diagnostics log to CSV
        diagnostics_df = pd.DataFrame(all_diagnostics)
        diagnostics_df.to_csv('abm_diagnostics_log.csv', index=False)
        print(f"Diagnostics log written to abm_diagnostics_log.csv with {len(diagnostics_df)} records.")
        return pd.DataFrame(simulation_data)

    def get_affected_countries(self) -> List[str]:
        """Get list of countries affected by active shocks."""
        affected_countries = []
        for shock in self.active_shocks:
            affected_countries.extend(shock.affected_countries)
        return list(set(affected_countries))  # Remove duplicates
    
    def is_connected_to_affected_country(self, supplier_id: str) -> bool:
        """Check if a supplier is connected to an affected country through the supply chain."""
        if supplier_id not in self.agents:
            return False
        
        agent = self.agents[supplier_id]
        affected_countries = self.get_affected_countries()
        
        # Direct connection: supplier is in affected country
        if agent.characteristics.country in affected_countries:
            return True
        
        # Indirect connection: supplier has buyers in affected countries
        for buyer_id in agent.buyers:
            if buyer_id in self.agents:
                buyer = self.agents[buyer_id]
                if buyer.characteristics.country in affected_countries:
                    return True
        
        return False

    def _get_recursive_network_cost(self, supplier_id, visited=None):
        """
        Recursively sum current_cost for a supplier and all its upstream suppliers, avoiding cycles.
        """
        if visited is None:
            visited = set()
        if supplier_id in visited:
            return 0.0
        visited.add(supplier_id)
        agent = self.agents[supplier_id]
        total = agent.current_cost
        for upstream_id in agent.suppliers:
            if upstream_id in self.agents:
                total += self._get_recursive_network_cost(upstream_id, visited)
        return total

    def get_full_network_landed_cost(self):
        """
        Aggregate recursively the costs of all Tier 1 suppliers and their full upstream supply chain.
        """
        # Find all Tier 1 suppliers directly connected to the brand (Tier 0)
        tier1_suppliers = [aid for aid, agent in self.agents.items()
                           if agent.characteristics.tier == 1 and 'knitex 96' in agent.buyers]
        total_cost = 0.0
        for supplier_id in tier1_suppliers:
            total_cost += self._get_recursive_network_cost(supplier_id)
        return total_cost

    def _collect_period_data(self) -> Dict:
        """Collects and aggregates data for the current period with time-decaying shocks."""
        # Initialize cumulative tracking variables
        total_suppliers = len(self.agents)
        ever_affected = 0
        ever_switched = 0
        
        total_applied_shock = 0.0
        affected_suppliers = 0
        affected_cost = 0
        for agent in self.agents.values():
            if agent.current_cost > agent.base_cost:
                affected_cost += 1
        
        # Track current states per period (new)
        currently_operational = 0
        currently_switched = 0
        currently_affected = 0
        
        tier_stats = {
            tier: {
                'operational': 0, 'affected': 0, 'switched': 0,
                'total_cost': 0.0, 'total_margin': 0.0, 'total_shock': 0.0,
                'count': 0
            }
            for tier in self.tier_range  # Use actual tier range from data
        }
        
        for agent in self.agents.values():
            # Track cumulative states using "ever" properties
            if hasattr(agent, 'ever_switched') and agent.ever_switched:
                ever_switched += 1
                if agent.characteristics.tier in tier_stats:
                    tier_stats[agent.characteristics.tier]['switched'] += 1
            
            if hasattr(agent, 'ever_affected') and agent.ever_affected:
                ever_affected += 1
                if agent.characteristics.tier in tier_stats:
                    tier_stats[agent.characteristics.tier]['affected'] += 1
            
            # Track tier stats for current states
            if agent.status == SupplierStatus.OPERATIONAL:
                if agent.characteristics.tier in tier_stats:
                    tier_stats[agent.characteristics.tier]['operational'] += 1
            elif agent.status == SupplierStatus.SWITCHED:
                if agent.characteristics.tier in tier_stats:
                    tier_stats[agent.characteristics.tier]['switched'] += 1
            if agent.current_cost > agent.base_cost:
                if agent.characteristics.tier in tier_stats:
                    tier_stats[agent.characteristics.tier]['affected'] += 1
            
            # Track current states per period (new logic)
            if agent.status == SupplierStatus.OPERATIONAL:
                currently_operational += 1
            elif agent.status == SupplierStatus.SWITCHED:
                currently_switched += 1
            
            # Currently affected if current_cost > base_cost
            if agent.current_cost > agent.base_cost:
                currently_affected += 1
                    
            # Track shock metrics
            applied_shock = getattr(agent, 'applied_shock', 0.0)
            total_applied_shock += applied_shock
            if applied_shock > 0:
                affected_suppliers += 1
            
            # Track tier-level KPIs
            if agent.characteristics.tier in tier_stats:
                tier_stats[agent.characteristics.tier]['total_cost'] += agent.current_cost or 0.0
                tier_stats[agent.characteristics.tier]['total_margin'] += getattr(agent, 'adjusted_margin', 0.0)
                tier_stats[agent.characteristics.tier]['total_shock'] += applied_shock
                tier_stats[agent.characteristics.tier]['count'] += 1
                
        # Compute averages
        for tier in tier_stats:
            count = tier_stats[tier]['count']
            if count > 0:
                tier_stats[tier]['avg_cost'] = tier_stats[tier]['total_cost'] / count
                tier_stats[tier]['avg_margin'] = tier_stats[tier]['total_margin'] / count
                tier_stats[tier]['avg_applied_shock'] = tier_stats[tier]['total_shock'] / count
            else:
                tier_stats[tier]['avg_cost'] = 0.0
                tier_stats[tier]['avg_margin'] = 0.0
                tier_stats[tier]['avg_applied_shock'] = 0.0
                
        base_cost, landed_cost = self._get_finisterre_landed_cost()
        effective_shock = (landed_cost - base_cost) / base_cost if base_cost > 0 else 0
        
        # Calculate network-wide averages
        total_agents = len(self.agents)
        avg_applied_shock = total_applied_shock / total_agents if total_agents > 0 else 0
        
        # --- NEW: Track per-agent status for this period ---
        agent_ids = list(self.agents.keys())
        statuses = {aid: agent.status.name if hasattr(agent, 'status') else 'OPERATIONAL' for aid, agent in self.agents.items()}
        
        return {
            'period': self.current_period,
            'operational': total_suppliers,  # All suppliers have been operational at some point
            'affected': ever_affected,
            'switched': ever_switched,
            # New current state columns
            'currently_operational': currently_operational,
            'currently_switched': currently_switched,
            'currently_affected': currently_affected,
            'affected_cost': affected_cost,
            'landed_cost': landed_cost,
            'base_cost': base_cost,
            'full_network_landed_cost': self.get_full_network_landed_cost(),
            'effective_shock': effective_shock,
            'avg_applied_shock': avg_applied_shock,
            'total_applied_shock': total_applied_shock,
            'affected_suppliers': affected_suppliers,
            'total_agents': total_agents,
            'tier_stats': tier_stats,
            'agent_ids': agent_ids,
            'statuses': statuses
        }
    
    def _get_finisterre_landed_cost(self) -> Tuple[float, float]:
        """
        Calculate Finisterre's landed cost using a fixed set of suppliers from period 1.
        Landed cost = sum of current_cost for all suppliers in the original path to Finisterre.
        For SWITCHED suppliers, use penalized cost.
        """
        # Lock in the set of suppliers in the path to Finisterre from period 1
        if not hasattr(self, '_fixed_finisterre_supplier_ids'):
            self._fixed_finisterre_supplier_ids = set(
                agent.id for agent in self.agents.values()
                if self._is_in_path_to_finisterre(agent.id)
            )
        base_cost = 0.0
        current_cost = 0.0
        finisterre_suppliers = [
            self.agents[aid] for aid in self._fixed_finisterre_supplier_ids if aid in self.agents
        ]
        if not finisterre_suppliers:
            print("WARNING: No suppliers in fixed path to Finisterre. Returning baseline cost.")
            fallback_base_cost = sum(self.original_base_costs.values())
            return fallback_base_cost, fallback_base_cost
        # --- DEBUG: Print suppliers included in landed cost calculation ---
        print("[LANDED COST DEBUG] Suppliers in path to Knitex 96:")
        for supplier in finisterre_suppliers:
            print(f"  - {supplier.id} (Tier {supplier.characteristics.tier}, {supplier.characteristics.country})")
        # --- DEBUG: Print cost history for each supplier ---
        for supplier in finisterre_suppliers:
            print(f"[SUPPLIER DEBUG] {supplier.id}: base_cost={supplier.base_cost}, cost_history={getattr(supplier, 'cost_history', [])}")
        for supplier in finisterre_suppliers:
            original_fob_cost = self.original_base_costs.get(supplier.id, supplier.base_cost)
            base_cost += original_fob_cost
            current_cost += supplier.current_cost
        print(f"[LANDED COST DEBUG] base_cost={base_cost}, current_cost={current_cost}, diff={current_cost - base_cost}")
        return base_cost, current_cost

    def _is_in_path_to_finisterre(self, supplier_id: str, visited=None) -> bool:
        """
        Check if a supplier is in the path to Knitex 96 by tracing the supply chain.
        Returns True if the supplier is connected to Knitex 96.
        """
        if visited is None:
            visited = set()
        if supplier_id in visited:
            return False  # Prevent infinite loops in cycles
        visited.add(supplier_id)

        agent = self.agents.get(supplier_id)
        if agent is None:
            return False

        # Base case: is this Knitex 96? (case-insensitive match)
        if supplier_id.lower() == "knitex 96":
            return True

        # Recursively check if any downstream buyer is in the path to Knitex 96
        return any(
            self._is_in_path_to_finisterre(buyer_id, visited)
            for buyer_id in agent.buyers
            if buyer_id in self.agents
        )

    def _get_tier_order(self) -> List[int]:
        """Returns tiers in propagation order (e.g., [4, 3, 2, 1] for 4 tiers). Uses actual tier range from data."""
        return sorted(self.tier_range, reverse=True)  # Return tiers in descending order for propagation

def create_tariff_schedule_example():
    """Create example tariff schedules for demonstration"""
    
    # Example 1: Escalating Trade War
    escalating_war = TariffSchedule(
        phases=[
            TariffPhase(start_period=6, end_period=8, magnitude=0.05),   # Initial 5%
            TariffPhase(start_period=9, end_period=11, magnitude=0.10),  # Escalation 10%
            TariffPhase(start_period=12, end_period=14, magnitude=0.15), # Peak 15%
            TariffPhase(start_period=15, end_period=17, magnitude=0.08), # De-escalation 8%
        ],
        target_countries=["Turkey"]
    )
    
    # Example 2: Cyclical Policy Changes
    cyclical_policy = TariffSchedule(
        phases=[
            TariffPhase(start_period=3, end_period=6, magnitude=0.08),   # Initial 8%
            TariffPhase(start_period=7, end_period=10, magnitude=0.12),  # Increase 12%
            TariffPhase(start_period=11, end_period=14, magnitude=0.06), # Decrease 6%
            TariffPhase(start_period=15, end_period=18, magnitude=0.10), # Final 10%
        ],
        target_countries=["China"]
    )
    
    # Example 3: Sudden Policy Shift
    sudden_shift = TariffSchedule(
        phases=[
            TariffPhase(start_period=1, end_period=9, magnitude=0.0),    # No tariff
            TariffPhase(start_period=10, end_period=15, magnitude=0.20), # Sudden 20%
            TariffPhase(start_period=16, end_period=18, magnitude=0.0),  # Removal
        ],
        target_countries=["India"]
    )
    
    return {
        "escalating_war": escalating_war,
        "cyclical_policy": cyclical_policy,
        "sudden_shift": sudden_shift
    }

def analyze_tariff_effects(abm, results_df, tariff_schedule, country_name):
    """Analyze the effects of dynamic tariff changes"""
    
    print(f"\n=== Tariff Effect Analysis for {country_name} ===")
    
    # Track tariff rates over time
    tariff_rates = []
    for period in range(1, SIMULATION_DURATION + 1):
        rate = tariff_schedule.get_tariff_rate(period)
        tariff_rates.append(rate)
        
        if rate > 0:
            active_phase = tariff_schedule.get_active_phase(period)
            print(f"Period {period}: {rate:.1%} tariff (Phase: {active_phase.start_period}-{active_phase.end_period})")
    
    # Find periods with maximum and minimum tariffs
    max_rate = max(tariff_rates)
    min_rate = min(tariff_rates)
    max_periods = [i+1 for i, rate in enumerate(tariff_rates) if rate == max_rate]
    min_periods = [i+1 for i, rate in enumerate(tariff_rates) if rate == min_rate]
    
    print(f"\nTariff Summary:")
    print(f"  Maximum tariff: {max_rate:.1%} (Periods: {max_periods})")
    print(f"  Minimum tariff: {min_rate:.1%} (Periods: {min_periods})")
    print(f"  Average tariff: {sum(tariff_rates)/len(tariff_rates):.1%}")
    
    # Analyze supplier impacts during different tariff phases
    affected_suppliers = [agent for agent in abm.agents.values() 
                         if agent.characteristics.country == country_name]
    
    if affected_suppliers:
        print(f"\nAffected Suppliers in {country_name}: {len(affected_suppliers)}")
        for supplier in affected_suppliers:
            print(f"  - {supplier.id} (Tier {supplier.characteristics.tier})")
    
    return tariff_rates

def run_macro_shock_scenarios():
    """Run macroeconomic shock scenarios with dynamic tariff scheduling"""
    
    print("=" * 60)
    print("FINISTERRE MACROECONOMIC SHOCK PROPAGATION ABM")
    print("=" * 60)
    print(f"Simulation Duration: {SIMULATION_DURATION} periods")
    print("=" * 60)
    
    # Load data
    try:
        df = pd.read_excel('Data/processed/integrated_trade_data.xlsx')
        print(f"Loaded {len(df)} trade records")
    except FileNotFoundError:
        print("Error: integrated_trade_data.xlsx not found. Please run data processing first.")
        return None, None
    
    # Prompt user for switching parameters
    def get_float_input(prompt, default):
        try:
            val = input(f"{prompt} [default={default}]: ")
            return float(val) if val.strip() != '' else default
        except Exception:
            return default
    def get_int_input(prompt, default):
        try:
            val = input(f"{prompt} [default={default}]: ")
            return int(val) if val.strip() != '' else default
        except Exception:
            return default

    switching_cost_factor = get_float_input("Enter switching_cost_factor (decision threshold, e.g., 0.2)", 0.2)
    switching_friction_penalty = get_float_input("Enter switching_friction_penalty (temporary cost penalty, e.g., 0.2)", 0.2)
    cost_increase_threshold = get_float_input("Enter cost_increase_threshold (e.g., 0.1 for 10%)", 0.1)
    disruption_duration = get_int_input("Enter disruption_duration (periods, e.g., 3)", 3)
    switching_penalty = get_float_input("Enter switching_penalty (e.g., 0.1 for 10%)", 0.1)

    abm = FinisterreSupplyChainABM(
        df,
        switching_cost_factor=switching_cost_factor,
        switching_friction_penalty=switching_friction_penalty,
        cost_increase_threshold=cost_increase_threshold,
        disruption_duration=disruption_duration,
        switching_penalty=switching_penalty
    )
    print(f"Initialized network with {len(abm.agents)} agents")
    print(f"Network has {abm.network.number_of_edges()} supply relationships")
    
    # Scenario 1: Dynamic Turkey Tariff Shock with Trade War Simulation
    print("\n" + "="*50)
    print("SCENARIO 1: Dynamic Turkey Tariff Shock (Trade War Simulation)")
    print("="*50)
    
    # Create a dynamic tariff schedule simulating a trade war
    turkey_tariff_phases = [
        TariffPhase(start_period=6, end_period=8, magnitude=0.05),   # Initial 5% tariff
        TariffPhase(start_period=9, end_period=11, magnitude=0.10),  # Escalation to 10%
        TariffPhase(start_period=12, end_period=14, magnitude=0.15), # Further escalation to 15%
        TariffPhase(start_period=15, end_period=17, magnitude=0.08), # De-escalation to 8%
    ]
    
    turkey_tariff_schedule = TariffSchedule(
        phases=turkey_tariff_phases,
        target_countries=["Turkey"]
    )
    
    # Print tariff schedule for transparency
    print("Turkey Tariff Schedule:")
    for i, phase in enumerate(turkey_tariff_phases):
        print(f"  Phase {i+1}: Periods {phase.start_period}-{phase.end_period} → {phase.magnitude:.1%} tariff")
    
    tariff_shock = MacroShock(
        shock_type=ShockType.TARIFF,
        affected_countries=["Turkey"],
        magnitude=0.10,  # Default magnitude (overridden by schedule)
        duration=SIMULATION_DURATION,  # Use configurable duration
        start_period=6,
        tariff_schedule=turkey_tariff_schedule
    )
    
    abm.apply_macro_shock(tariff_shock)
    results = abm.run_comprehensive_simulation(periods=SIMULATION_DURATION)
    
    # Analyze tariff effects
    analyze_tariff_effects(abm, results, turkey_tariff_schedule, "Turkey")
    
    # Identify critical suppliers
    critical_suppliers = abm.identify_critical_suppliers()
    
    print(f"\nTop 5 Critical Suppliers:")
    for i, (supplier_id, score) in enumerate(critical_suppliers[:5]):
        agent = abm.agents[supplier_id]
        print(f"{i+1}. {supplier_id} (Tier {agent.characteristics.tier}, "
              f"{agent.characteristics.country}) - Score: {score:.3f}")
    
    # Reset for next scenario
    abm = FinisterreSupplyChainABM(df, switching_cost_factor=switching_cost_factor)
    
    # Scenario 2: China Inflation Shock (unchanged)
    print("\n" + "="*50)
    print("SCENARIO 2: China Inflation Shock (+15%)")
    print("="*50)
    
    inflation_shock = MacroShock(
        shock_type=ShockType.INFLATION,
        affected_countries=["China"],
        magnitude=0.15,  # 15% inflation
        duration=36,  # 36 months
        start_period=3
    )
    
    abm.apply_macro_shock(inflation_shock)
    results2 = abm.run_comprehensive_simulation(periods=SIMULATION_DURATION)
    
    # Scenario 3: Multi-Country Dynamic Tariff Scenario
    print("\n" + "="*50)
    print("SCENARIO 3: Multi-Country Dynamic Tariff Scenario")
    print("="*50)
    
    # Reset ABM for third scenario
    abm = FinisterreSupplyChainABM(df, switching_cost_factor=switching_cost_factor)
    
    # Create a complex tariff schedule affecting multiple countries
    china_tariff_phases = [
        TariffPhase(start_period=3, end_period=6, magnitude=0.08),   # Initial 8% tariff
        TariffPhase(start_period=7, end_period=10, magnitude=0.12),  # Escalation to 12%
        TariffPhase(start_period=11, end_period=14, magnitude=0.06), # Reduction to 6%
        TariffPhase(start_period=15, end_period=18, magnitude=0.10), # Final 10% tariff
    ]
    
    china_tariff_schedule = TariffSchedule(
        phases=china_tariff_phases,
        target_countries=["China"]
    )
    
    print("China Tariff Schedule:")
    for i, phase in enumerate(china_tariff_phases):
        print(f"  Phase {i+1}: Periods {phase.start_period}-{phase.end_period} → {phase.magnitude:.1%} tariff")
    
    china_tariff_shock = MacroShock(
        shock_type=ShockType.TARIFF,
        affected_countries=["China"],
        magnitude=0.10,  # Default magnitude (overridden by schedule)
        duration=SIMULATION_DURATION,
        start_period=3,
        tariff_schedule=china_tariff_schedule
    )
    
    abm.apply_macro_shock(china_tariff_shock)
    results3 = abm.run_comprehensive_simulation(periods=SIMULATION_DURATION)
    
    # Analyze tariff effects for China scenario
    analyze_tariff_effects(abm, results3, china_tariff_schedule, "China")
    
    print("\n" + "="*60)
    print("MACRO SHOCK SIMULATION COMPLETED")
    print("="*60)
    print("\nKey Insights:")
    print("- Dynamic tariff schedules simulate realistic trade war scenarios")
    print("- Tariff rates change over time based on policy phases")
    print("- Turkey tariff shocks affect Tier-2 suppliers directly")
    print("- China inflation shocks cascade up from Tier-3/4/5 suppliers")
    print("- Product Importance Ratio determines shock transmission")
    print("- Switching decisions based on cost vs. disruption trade-off")
    
    return abm, results

if __name__ == "__main__":
    abm, results = run_macro_shock_scenarios()
    # After simulation, before visualizations, add debug for sappi papier holding gmbh
    if 'sappi papier holding gmbh' in abm.agents:
        agent = abm.agents['sappi papier holding gmbh']
        print('sappi papier holding gmbh period_costs:', agent.period_costs)
        print('sappi papier holding gmbh period_base_costs:', agent.period_base_costs)
        print('sappi papier holding gmbh base_cost:', agent.base_cost)
        if hasattr(agent, 'period_status'):
            print('sappi papier holding gmbh status history:', agent.period_status)
        else:
            print('sappi papier holding gmbh status history: Not tracked')