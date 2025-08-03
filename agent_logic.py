from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Union
import random # Added for random.random()

class SupplierStatus(Enum):
    """Status of a supplier in the network"""
    OPERATIONAL = "operational"
    SWITCHED = "switched"
    AFFECTED = "affected"
    IN_TRANSITION = "in_transition"

class ShockType(Enum):
    """Types of shocks that can be applied"""
    NONE = "none"
    INFLATION = "inflation"
    TARIFF = "tariff"

# ShockDuration enum removed - duration is now set as user parameter

class SupplierTier(Enum):
    """Supplier tiers in the network"""
    TIER1_INTERMEDIARY = 1    # Direct suppliers to Finisterre
    TIER2_MANUFACTURER = 2    # Manufacturers
    TIER3_COMPONENTS = 3      # Specialized component suppliers
    TIER4_MATERIALS = 4       # Material suppliers
    # Note: TIER5_RAW removed as data only contains 4 tiers

@dataclass
class SupplierCharacteristics:
    """Core characteristics of a supplier"""
    id: str
    country: str
    tier: int
    product_importance_ratio: float
    profit_margin: float

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

@dataclass
class MacroShock:
    """Macroeconomic shock definition"""
    shock_type: ShockType
    affected_countries: List[str]
    magnitude: float  # percentage increase (e.g., 0.10 for 10%)
    duration: int  # Duration in periods (set by user)
    start_period: int
    # New field for tariff schedules
    tariff_schedule: Optional[TariffSchedule] = None
    
    def get_effective_magnitude(self, period: int) -> float:
        """
        Get the effective magnitude for a specific period.
        For tariff shocks with schedules, this returns the current tariff rate.
        For other shocks, this returns the original magnitude.
        """
        if self.shock_type == ShockType.TARIFF and self.tariff_schedule is not None:
            return self.tariff_schedule.get_tariff_rate(period)
        return self.magnitude

class SupplierAgent:
    """
    Agent representing a supplier in the supply chain network.
    Enhanced for more dynamic behavior:
    - Shock persistence and accumulation over time
    - Gradual recovery mechanisms
    - More sophisticated switching decisions
    - Shock memory and hysteresis effects
    """
    def __init__(self, characteristics: SupplierCharacteristics, base_cost: float, abm=None):
        """Initialize a supplier agent"""
        self.id = characteristics.id
        self.characteristics = characteristics
        self.status = SupplierStatus.OPERATIONAL
        self.abm = abm
        
        # Enhanced cost tracking with persistence
        self.base_cost = base_cost
        self.current_cost = base_cost
        self.incoming_shock = 0.0  # Shock received from downstream
        self.applied_shock = 0.0   # Shock applied to costs
        self.accumulated_shock = 0.0  # Total shock accumulated over time
        self.shock_history = []  # Track shock over time for hysteresis
        
        # Recovery and persistence parameters
        self.recovery_rate = 0.02  # 2% recovery per period when no new shocks
        self.shock_persistence = 0.8  # 80% of shock persists to next period
        self.max_shock_accumulation = 0.5  # Maximum 50% accumulated shock
        self.recovery_threshold = 0.05  # Start recovery when shock < 5%
        
        # Network relationships
        self.buyers: Set[str] = set()
        self.suppliers: Set[str] = set()
        
        # Enhanced state tracking
        self.periods_under_shock = 0
        self.periods_in_recovery = 0
        self.original_margin = characteristics.profit_margin
        self.adjusted_margin = characteristics.profit_margin
        self.affected = False
        self.switched_supplier = False
        self.switches_made = 0  # Track number of switches made by this buyer
        self.max_switches = 2   # Maximum allowed switches per buyer per simulation
        self.switching_cooldown = 0  # Periods to wait before switching again
        self.recovery_period = 0  # Periods remaining in recovery/ramp-up
        # Per-buyer friction penalty tracking
        self.switching_friction_penalty = {}  # buyer_id -> periods left
        self.switching_friction_penalty_factor = {}  # buyer_id -> penalty factor
        self.diagnostics_log = []  # Per-agent diagnostics log
        
        # Status tracking
        self.has_been_switched = False  # Flag: has any buyer switched away from this supplier
        self.buyers_switched_away = set()  # Track which buyers have switched away
        
        # Enhanced switching logic
        self.switching_probability = 0.3  # Base probability of switching
        self.switching_memory = []  # Track switching decisions over time
        
        # Switch logging for post-analysis
        self.switch_log = []
        
        # Log base cost at initialization
        if abm is not None and hasattr(abm, 'base_cost_log'):
            abm.base_cost_log.append({'supplier_id': self.id, 'base_cost': self.base_cost})
        
        self.switching_penalty = abm.switching_penalty if abm and hasattr(abm, 'switching_penalty') else 0.1
        self.switch_period = None  # Track the period when supplier is switched
        self.ever_affected = False
        self.ever_switched = False
        self.cost_history = []
        
    def add_buyer(self, buyer_id: str):
        """Add downstream buyer relationship"""
        self.buyers.add(buyer_id)
    
    def add_supplier(self, supplier_id: str):
        """Add upstream supplier relationship"""
        self.suppliers.add(supplier_id)
    
    def receive_propagated_shock(self, shock_magnitude: float):
        """Receives a shock from a downstream buyer and adds it to incoming shock."""
        if self.status == SupplierStatus.OPERATIONAL:
            self.incoming_shock += shock_magnitude

    def apply_direct_shock(self, shocks, period):
        """
        Apply direct macro shocks to this agent if affected.
        For INFLATION: exponential decay (shock = magnitude * exp(-lambda * time_elapsed)).
        For TARIFF: dynamic scheduling with time-varying rates or no decay (shock = magnitude for duration).
        Lambda is configurable via abm.config['shock_decay_lambda'], default 0.115.
        """
        import math
        current_direct_shock = 0.0
        lambda_decay = 0.03  # Set decay rate to 0.03 for inflation shocks
        if self.abm is not None and hasattr(self.abm, 'config'):
            lambda_decay = self.abm.config.get('shock_decay_lambda', 0.03)
        
        for shock in shocks:
            if self.characteristics.country in shock.affected_countries and period >= shock.start_period:
                duration_value = shock.duration
                shock_end_period = shock.start_period + duration_value
                
                if period <= shock_end_period:
                    time_elapsed = period - shock.start_period
                    
                    if hasattr(shock, 'shock_type') and getattr(shock, 'shock_type', None) is not None:
                        if str(shock.shock_type).upper().endswith('INFLATION'):
                            # Exponential decay for inflation only
                            current_direct_shock = shock.magnitude * math.exp(-lambda_decay * time_elapsed)
                            if period == shock.start_period:
                                print(f"[DIRECT SHOCK] {self.id} | Country: {self.characteristics.country} | Tier: {self.characteristics.tier} | Initial Shock: {shock.magnitude:.4f} | Exponential Decay Lambda: {lambda_decay:.4f}")
                            elif current_direct_shock > 0:
                                print(f"[SHOCK DECAY] {self.id} | Period: {period} | Lambda: {lambda_decay:.4f} | Applied Shock: {current_direct_shock:.4f}")
                        
                        elif str(shock.shock_type).upper().endswith('TARIFF'):
                            # No decay for tariff shocks
                            if hasattr(shock, 'tariff_schedule') and shock.tariff_schedule is not None:
                                tariff_rate = shock.tariff_schedule.get_tariff_rate(period)
                                current_direct_shock = tariff_rate
                                
                                active_phase = shock.tariff_schedule.get_active_phase(period)
                                if active_phase:
                                    print(f"[TARIFF SCHEDULE] {self.id} | Period: {period} | Phase: {active_phase.start_period}-{active_phase.end_period} | Rate: {tariff_rate:.4f}")
                                elif tariff_rate == 0.0:
                                    print(f"[TARIFF SCHEDULE] {self.id} | Period: {period} | No active phase | Rate: 0.0")
                            else:
                                current_direct_shock = shock.magnitude
                                if period == shock.start_period:
                                    print(f"[DIRECT SHOCK] {self.id} | Country: {self.characteristics.country} | Tier: {self.characteristics.tier} | Tariff Shock: {shock.magnitude:.4f} (No Decay)")
                                else:
                                    print(f"[TARIFF SHOCK] {self.id} | Period: {period} | Applied Shock: {current_direct_shock:.4f}")
                        else:
                            # Default to previous linear decay if unknown type
                            decay_factor = max(0, 1 - time_elapsed / duration_value)
                            current_direct_shock = shock.magnitude * decay_factor
                    else:
                        # Fallback: treat as linear decay
                        decay_factor = max(0, 1 - time_elapsed / duration_value)
                        current_direct_shock = shock.magnitude * decay_factor
                break
        
        self.applied_shock = current_direct_shock
        if self.applied_shock > 0:
            self.current_cost = self.base_cost * (1 + self.applied_shock)
            print(f"[COST UPDATE] {self.id} | Base Cost: {self.base_cost:.2f} | Applied Shock: {self.applied_shock:.4f} | New Cost: {self.current_cost:.2f}")
        # Track ever_switched (status-based)
        self.ever_switched = self.ever_switched or (self.status == SupplierStatus.SWITCHED)

    def update_period(self, current_period: int):
        """
        Update agent state for the current period with time-decaying shocks and recursive shock propagation.
        """
        # Special case: Knitex 96 (focal company) should never be marked as switched
        if self.id.lower() == 'knitex 96':
            if self.current_cost > self.base_cost:
                self.status = SupplierStatus.AFFECTED
            else:
                self.status = SupplierStatus.OPERATIONAL
            # Still track ever_affected and cost_history
            self.ever_affected = self.ever_affected or (self.current_cost > self.base_cost)
            self.cost_history.append(self.current_cost)
            return
        # --- RECURSIVE SHOCK PROPAGATION ---
        # Aggregate input cost increases from all suppliers (if any)
        propagated_shock = 0.0
        if hasattr(self, 'input_cost_increases') and self.input_cost_increases:
            total_input_increase = sum(self.input_cost_increases.values())
            propagated_shock = total_input_increase / max(self.base_cost, 1e-6)
            self.input_cost_increases = {}
        effective_shock = self.applied_shock + propagated_shock
        # Diagnostic: Print cost and shock details for first 5 periods
        if current_period <= 5:
            print(f"[DEBUG COST] {self.id} | Period: {current_period} | base_cost: {self.base_cost:.4f} | applied_shock: {self.applied_shock:.4f} | propagated_shock: {propagated_shock:.4f} | effective_shock: {effective_shock:.4f} | current_cost: {self.current_cost:.4f}")
        if self.id.lower() == 'sappi papier holding gmbh':
            print(f"[DEBUG] Period: {current_period} | switch_period: {self.switch_period} | Status: {self.status} | applied_shock: {self.applied_shock} | propagated_shock: {propagated_shock}")
        if self.status == SupplierStatus.SWITCHED:
            if self.switch_period is None:
                self.switch_period = current_period
            if current_period == self.switch_period or current_period == self.switch_period + 1:
                self.current_cost = self.base_cost * (1 + self.applied_shock + propagated_shock + self.switching_penalty)
                if self.id.lower() == 'sappi papier holding gmbh':
                    print(f"[DEBUG] Applying switching penalty. current_cost: {self.current_cost}")
            else:
                self.applied_shock = 0.0
                propagated_shock = 0.0
                self.current_cost = self.base_cost
                if self.id.lower() == 'sappi papier holding gmbh':
                    print(f"[DEBUG] Penalty window ended. current_cost reset to base: {self.current_cost}")
        else:
            self.switch_period = None
            self.current_cost = self.base_cost * (1 + effective_shock) + sum(
                self.base_cost * self.switching_friction_penalty_factor.get(buyer_id, 0.0)
                for buyer_id in self.switching_friction_penalty
                if self.switching_friction_penalty[buyer_id] > 0
            )
        # Always track ever_affected and cost_history, regardless of status or early return
        self.ever_affected = self.ever_affected or (self.current_cost > self.base_cost)
        self.cost_history.append(self.current_cost)
        reported_margin = self.original_margin
        self.adjusted_margin = reported_margin * (1 - effective_shock)
        print(f"[MARGIN DIAGNOSTIC] {self.id} | Period: {current_period} | Original Margin: {reported_margin:.4f} | Adjusted Margin: {self.adjusted_margin:.4f} | Effective Shock: {effective_shock:.4f}")
        # New affected logic: affected if current_cost > base_cost (no threshold)
        self.affected = self.current_cost > self.base_cost
        if self.affected:
            print(f"[AFFECTED] {self.id} | Country: {self.characteristics.country} | Tier: {self.characteristics.tier} | Current Cost: {self.current_cost:.2f} | Base Cost: {self.base_cost:.2f} | Applied Shock: {self.applied_shock:.4f} | Propagated Shock: {propagated_shock:.4f}")
        # Update switching cooldown
        if self.switching_cooldown > 0:
            self.switching_cooldown -= 1
        
        # Check if switching penalties have expired and clear buyers_switched_away
        if self.buyers_switched_away:
            # Check if all switching penalties have expired
            all_penalties_expired = True
            for buyer_id in list(self.buyers_switched_away):
                if buyer_id in self.switching_friction_penalty and self.switching_friction_penalty[buyer_id] > 0:
                    all_penalties_expired = False
                    break
            
            # If all penalties expired, clear the switched away status
            if all_penalties_expired:
                self.buyers_switched_away.clear()
                print(f"[STATUS] {self.id}: All penalties expired, clearing switched away status")
        
        # NEW LOGIC: Mark as switched if ANY buyer has switched away from this supplier
        if any(buyer_id in self.buyers_switched_away for buyer_id in self.buyers):
            self.status = SupplierStatus.SWITCHED
            print(f"[STATUS] {self.id}: SWITCHED (buyer switched away)")
            return
        
        # New: assign AFFECTED status if affected and not switched
        if self.affected:
            self.status = SupplierStatus.AFFECTED
            print(f"[STATUS] {self.id}: AFFECTED (cost above baseline)")
            return
        
        self.status = SupplierStatus.OPERATIONAL
        print(f"[STATUS] {self.id}: OPERATIONAL (supplying {len(self.buyers) - len(self.buyers_switched_away)} buyers)")
        self.diagnostics_log.append({
            'period': current_period,
            'supplier_id': self.id,
            'reported_margin': self.original_margin,
            'applied_shock': effective_shock,
            'adjusted_margin': self.adjusted_margin,
            'status': self.status.value,
            'affected': self.affected,
            'switch_count': self.switches_made,
            'in_recovery_periods_remaining': self.recovery_period,
            'current_cost': self.current_cost
        })
        # Track ever_affected (cost-based, not status-based)
        # self.ever_affected = self.ever_affected or (self.current_cost > self.base_cost) # This line is now handled above
        # self.cost_history.append(self.current_cost) # This line is now handled above

    def decide_and_switch_suppliers(self):
        abm = self.abm
        
        # === NO SWITCHING CHECK ===
        # Check if switching is disabled via config flag
        if hasattr(abm, 'config') and abm.config.get('no_switching', False):
            return  # Exit immediately - no switching allowed
        # === END NO SWITCHING CHECK ===
        
        switching_threshold = abm.config.get('cost_increase_threshold', 0.1)  # Not used, but kept for flexibility
        persistence_required = 2
        disruption_duration = abm.config.get('disruption_duration', 3)
        switching_cost_factor = abm.switching_cost_factor
        friction_penalty = abm.switching_friction_penalty
        penalty_duration = 2
        if self.switching_cooldown > 0 or self.switches_made >= self.max_switches:
            return
        # --- DEBUG: Print switching evaluation for Knitex 96 ---
        if self.id.lower() == 'knitex 96':
            print(f"[DEBUG SWITCH] Knitex 96 evaluating switching. Current suppliers: {list(self.suppliers)}")
        if not hasattr(self, 'periods_above_penalty'):
            self.periods_above_penalty = {}
        for supplier_id in list(self.suppliers):
            supplier = abm.agents[supplier_id]
            # Calculate percentage change in price
            if supplier.base_cost == 0:
                price_change_pct = 0.0
            else:
                price_change_pct = (supplier.current_cost - supplier.base_cost) / supplier.base_cost
            disruption_cost = price_change_pct * disruption_duration
            if supplier_id not in self.periods_above_penalty:
                self.periods_above_penalty[supplier_id] = 0
            if disruption_cost > switching_cost_factor:
                self.periods_above_penalty[supplier_id] += 1
            else:
                self.periods_above_penalty[supplier_id] = 0
            # --- DEBUG: Print switching threshold evaluation ---
            if self.id.lower() == 'knitex 96':
                print(f"[DEBUG SWITCH] Supplier: {supplier_id}, Disruption cost: {disruption_cost:.4f}, Switching cost factor: {switching_cost_factor}, Periods above penalty: {self.periods_above_penalty[supplier_id]}")
            if self.periods_above_penalty[supplier_id] >= persistence_required:
                # Always apply switching penalty and mark as switched, do not look for alternatives
                supplier.switching_friction_penalty[self.id] = penalty_duration
                supplier.switching_friction_penalty_factor[self.id] = friction_penalty
                supplier.buyers_switched_away.add(self.id)
                supplier.has_been_switched = True
                self.switching_cooldown = 6
                self.switches_made += 1
                if hasattr(abm, 'switch_log'):
                    abm.switch_log.append({
                        'buyer_id': self.id,
                        'from_supplier': supplier_id,
                        'to_supplier': None,
                        'period': abm.current_period,
                        'switching_cost_factor': switching_cost_factor,
                        'switching_friction_penalty': friction_penalty,
                        'from_base_cost': supplier.base_cost,
                        'to_base_cost': None
                    })
                print(f"Switch event: {self.id} switched away from {supplier_id} at period {abm.current_period} (penalty applied, no alternative supplier)")
                self.periods_above_penalty[supplier_id] = 0
                break

    def propagate_cost_increase(self):
        """
        Propagate cost increases through actual buyer-supplier trade links.
        Buyers of a shocked supplier will feel a higher input cost.
        No inheritance factors - just direct cost pass-through.
        """
        if self.applied_shock > 0:
            print(f"[PROPAGATE] {self.id} (Tier {self.characteristics.tier}, {self.characteristics.country}) has shock: {self.applied_shock:.4f}")
            
            # Propagate to all buyers through trade links
            for buyer_id in self.buyers:
                if buyer_id in self.abm.agents:
                    buyer = self.abm.agents[buyer_id]
                    
                    # Calculate cost increase based on trade relationship
                    # This represents the buyer's increased input cost from this supplier
                    cost_increase = self.current_cost - self.base_cost
                    
                    # Log the propagation path
                    print(f"[PROPAGATION PATH] {self.id} -> {buyer_id} | Cost Increase: {cost_increase:.2f} | Shock: {self.applied_shock:.4f}")
                    
                    # The buyer will experience this cost increase in their next period
                    # This affects their switching decisions and cost calculations
                    if hasattr(buyer, 'input_cost_increases'):
                        buyer.input_cost_increases[self.id] = cost_increase
                    else:
                        buyer.input_cost_increases = {self.id: cost_increase}

    def get_buyer_cost(self, buyer_id):
        # Used for landed cost calculation per buyer
        penalty = 0.0
        if buyer_id in self.switching_friction_penalty and self.switching_friction_penalty[buyer_id] > 0:
            penalty = self.base_cost * self.switching_friction_penalty_factor.get(buyer_id, 0.0)
            self.switching_friction_penalty[buyer_id] -= 1
            if self.switching_friction_penalty[buyer_id] == 0:
                del self.switching_friction_penalty[buyer_id]
                del self.switching_friction_penalty_factor[buyer_id]
        return self.base_cost * (1 + self.applied_shock) + penalty

    def _print_shock_diagnostics(self, shock_impact: float):
        """Print detailed diagnostic information about shock application"""
        print(f"\nShock applied to {self.id} (Tier {self.characteristics.tier}):")
        print(f"  Country: {self.characteristics.country}")
        print(f"  Base cost: {self.base_cost:.2f}")
        print(f"  Original margin: {self.original_margin:.2%}")
        print(f"  Shock impact: {shock_impact:.2%}")
        print(f"  PIR: {self.characteristics.product_importance_ratio:.2f}")
        print(f"  Applied shock: {self.applied_shock:.2%}")
        print(f"  Accumulated shock: {self.accumulated_shock:.2%}")
        print(f"  New cost: {self.current_cost:.2f}")
        print(f"  Adjusted margin: {self.adjusted_margin:.2%}")
        print(f"  Affected: {self.affected}")
        print(f"  Status: {self.status.value}")
        print(f"  Periods under shock: {self.periods_under_shock}")
        print(f"  Periods in recovery: {self.periods_in_recovery}") 