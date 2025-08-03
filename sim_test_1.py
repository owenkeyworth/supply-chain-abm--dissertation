import pandas as pd
import numpy as np
from Finisterre_ABM_Implementation import (
    FinisterreSupplyChainABM, 
    MacroShock, 
    ShockType, 
    SupplierStatus,
    TariffPhase,
    TariffSchedule,
    SIMULATION_DURATION
)
# from dissertation_visualizations import create_dissertation_visualizations

# --- Add imports for new visuals ---
from model_visuals import (
    plot_supplier_state_evolution,
    plot_cumulative_cost_impact,
    plot_geographic_risk,
    plot_tier1_landed_cost_analysis,
    plot_tier_level_impact,
    plot_full_network_landed_cost_analysis,
    plot_meaningful_supplier_state_evolution,
    plot_post_simulation_network_impact,  # <-- add this import
    plot_overlaid_landed_cost_comparison,
    plot_overlaid_tier1_landed_cost_comparison,
    preprocess_visual_supplier_states,
    plot_supplier_state_heatmap,  # <-- add this import
    plot_supplier_state_switched_affected  # <-- use this instead of stacked area
)
import os
from extract_abm_metrics import extract_metrics

def get_available_countries(df):
    """Get list of available countries from the dataset."""
    countries = sorted(df['supplier_country'].unique())
    return [country.title() for country in countries if pd.notna(country)]

def get_inflation_rate(df, country, year=None, custom_rate=None):
    """Get inflation rate for a country and year, or use custom rate."""
    if custom_rate is not None:
        return custom_rate
    
    country_lower = country.lower()
    year_col = f'inflation_{year}'
    
    if year_col not in df.columns:
        raise ValueError(f"Inflation data for year {year} not available. Available years: {[col.split('_')[1] for col in df.columns if col.startswith('inflation_')]}")
    
    inflation_rate = df[df['supplier_country'].str.lower() == country_lower][year_col].mean()
    
    if pd.isna(inflation_rate):
        raise ValueError(f"No inflation data found for {country} in year {year}")
    
    return inflation_rate

def main():
    tariff_schedule = None  # Ensure this is always defined, even if not set later
    # 1. Load integrated data
    print("Loading trade data...")
    df = pd.read_excel('Data/processed/integrated_trade_data.xlsx')
    
    # Get available countries
    available_countries = get_available_countries(df)
    
    # 2. User parameter selection
    print("\n=== SIMULATION PARAMETERS ===")
    
    # Country selection
    print(f"\nAvailable countries ({len(available_countries)}):")
    for i, country in enumerate(available_countries, 1):
        print(f"{i:2d}. {country}")
    
    while True:
        try:
            country_choice = int(input(f"\nSelect country (1-{len(available_countries)}): "))
            if 1 <= country_choice <= len(available_countries):
                selected_country = available_countries[country_choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Shock type selection
    print("\nShock Type:")
    print("1. Inflation")
    print("2. Tariff")
    
    while True:
        try:
            shock_type_choice = int(input("Select shock type (1-2): "))
            if shock_type_choice in [1, 2]:
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Shock magnitude based on type
    if shock_type_choice == 1:  # Inflation
        print("\nInflation Options:")
        print("1. Use forecast data (2025-2030)")
        print("2. Custom inflation rate")
        
        while True:
            try:
                inflation_choice = int(input("Select option (1-2): "))
                if inflation_choice in [1, 2]:
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        if inflation_choice == 1:  # Forecast data
            available_years = [2025, 2026, 2027, 2028, 2029, 2030]
            print(f"\nAvailable years: {available_years}")
            
            while True:
                try:
                    year = int(input("Select year: "))
                    if year in available_years:
                        break
                    else:
                        print("Invalid year. Please try again.")
                except ValueError:
                    print("Please enter a valid year.")
            
            # Get inflation rate from data
            inflation_rate = get_inflation_rate(df, selected_country, year)
            print(f"{selected_country} {year} inflation rate: {inflation_rate:.2%}")
            shock_magnitude = inflation_rate
            shock_type = ShockType.INFLATION
            
        else:  # Custom rate
            while True:
                try:
                    custom_rate = float(input("Enter custom inflation rate as decimal (e.g., 0.15 for 15%): "))
                    if 0 <= custom_rate <= 2:  # Allow up to 200% inflation
                        break
                    else:
                        print("Please enter a rate between 0 and 2 (0% to 200%).")
                except ValueError:
                    print("Please enter a valid number.")
            
            shock_magnitude = custom_rate
            shock_type = ShockType.INFLATION
            
    else:  # Tariff
        print("\nTariff Options:")
        print("1. Fixed tariff rate (single rate for entire duration)")
        print("2. Dynamic tariff schedule (multiple phases with different rates)")
        
        while True:
            try:
                tariff_choice = int(input("Select option (1-2): "))
                if tariff_choice in [1, 2]:
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        if tariff_choice == 1:  # Fixed tariff
            while True:
                try:
                    tariff_rate = float(input("Enter tariff rate as decimal (e.g., 0.55 for 55%): "))
                    if 0 <= tariff_rate <= 1:  # Allow up to 100% tariff
                        break
                    else:
                        print("Please enter a rate between 0 and 1 (0% to 100%).")
                except ValueError:
                    print("Please enter a valid number.")
            shock_magnitude = tariff_rate
            shock_type = ShockType.TARIFF
            tariff_schedule = None
        else:  # Dynamic tariff schedule
            print(f"\nCreating dynamic tariff schedule")
            print("You can define multiple tariff phases. Each phase has:")
            print("- Start period (when this phase begins)")
            print("- End period (when this phase ends)") 
            print("- Tariff rate (e.g., 0.10 for 10%)")
            print("\nExample: A trade war that escalates then de-escalates:")
            print("  Phase 1: Periods 1-5 → 10% tariff (initial)")
            print("  Phase 2: Periods 6-10 → 20% tariff (escalation)")
            print("  Phase 3: Periods 11-15 → 5% tariff (de-escalation)")
            print("\nEnter phases (enter start_period=0 to finish):")
            phases = []
            last_end = 0
            while True:
                try:
                    recommended_start = last_end + 1
                    if last_end > 0:
                        print(f"Next recommended start period: {recommended_start}")
                    start_period = int(input(f"Phase {len(phases)+1} - Start period ({recommended_start} or higher): "))
                    if start_period == 0:
                        break
                    if start_period < 1:
                        print("Start period must be 1 or higher")
                        continue
                    # Check for overlap
                    overlap = False
                    for prev in phases:
                        if start_period <= prev.end_period:
                            overlap = True
                            break
                    if overlap:
                        print(f"WARNING: This start period overlaps with a previous phase (which ends at {prev.end_period}). Overlapping phases will override earlier ones for overlapping periods.")
                        confirm = input("Are you sure you want to overlap? (y/n): ").lower().strip()
                        if confirm not in ['y', 'yes']:
                            continue
                    end_period = int(input(f"Phase {len(phases)+1} - End period ({start_period} or higher): "))
                    if end_period < start_period:
                        print(f"End period must be {start_period} or higher")
                        continue
                    magnitude = float(input(f"Phase {len(phases)+1} - Tariff rate (e.g., 0.10 for 10%): "))
                    if magnitude < 0 or magnitude > 1:
                        print("Tariff rate must be between 0 and 1 (0% to 100%)")
                        continue
                    phases.append(TariffPhase(start_period, end_period, magnitude))
                    print(f"Added phase: Periods {start_period}-{end_period} → {magnitude:.1%} tariff")
                    last_end = max(last_end, end_period)
                    # Ask if they want to continue
                    continue_choice = input(f"Add another phase? (y/n): ").lower().strip()
                    if continue_choice not in ['y', 'yes']:
                        break
                except ValueError:
                    print("Please enter valid numbers.")
            if not phases:
                print("No phases defined. Using fixed 10% tariff.")
                shock_magnitude = 0.10
                tariff_schedule = None
            else:
                # Show summary before confirming
                print(f"\nTariff Schedule Summary:")
                for i, phase in enumerate(phases):
                    print(f"  Phase {i+1}: Periods {phase.start_period}-{phase.end_period} → {phase.magnitude:.1%} tariff")
                tariff_schedule = TariffSchedule(phases=phases, target_countries=[selected_country.lower()])
                shock_magnitude = max(phase.magnitude for phase in phases)  # Use max for display
            shock_type = ShockType.TARIFF
    
    # Other parameters
    scf = float(input("\nEnter the Switching Cost Factor (e.g., 0.2): "))
    
    # Prompt for Product Importance Ratio (PIR) and simulation length

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

    pir = get_float_input("Enter the Product Importance Ratio (PIR)", 0.45)
    sim_length = get_int_input("Enter the simulation duration in periods", 18)

    # Determine the shock magnitude and create the MacroShock
    if shock_type_choice == 1:  # Inflation
        if inflation_choice == 1:  # Forecast data
            shock_magnitude = inflation_rate
        else:  # Custom rate
            shock_magnitude = custom_rate
        macro_shock = MacroShock(
            shock_type=ShockType.INFLATION,
            affected_countries=[selected_country.lower()],
            magnitude=shock_magnitude,
            duration=sim_length,
            start_period=1
        )
    else:  # Tariff
        macro_shock = MacroShock(
            shock_type=ShockType.TARIFF,
            affected_countries=[selected_country.lower()],
            magnitude=shock_magnitude,
            duration=sim_length,
            start_period=1,
            tariff_schedule=tariff_schedule if 'tariff_schedule' in locals() else None
        )

    # --- WITH SWITCHING ---
    abm = FinisterreSupplyChainABM(
        trade_df=df,
        switching_cost_factor=scf,
        product_importance_ratio=pir
    )
    abm.apply_macro_shock(macro_shock)
    results_with_switching = abm.run_comprehensive_simulation(periods=sim_length)

    # --- NO SWITCHING ---
    abm_no_switch = FinisterreSupplyChainABM(
        trade_df=df,
        switching_cost_factor=9999,  # Effectively disables switching
        product_importance_ratio=pir,
        config={'no_switching': True}
    )
    abm_no_switch.apply_macro_shock(macro_shock)
    results_no_switching = abm_no_switch.run_comprehensive_simulation(periods=sim_length)

    # --- Add total_cost_by_tier and tier1_landed_cost columns ---
    def get_costs_by_tier(abm_instance, period):
        costs = {}
        for agent in abm_instance.agents.values():
            t = agent.characteristics.tier
            if t not in costs:
                costs[t] = 0.0
            costs[t] += agent.period_costs[period-1]
        return costs

    # For results_with_switching
    results_with_switching['total_cost_by_tier'] = [
        get_costs_by_tier(abm, p) for p in results_with_switching['period']
    ]
    results_with_switching['tier1_landed_cost'] = [
        costs.get(1, 0) for costs in results_with_switching['total_cost_by_tier']
    ]
    # Add tier1_baseline_cost column
    tier1_supplier_ids = [s for s, agent in abm.agents.items() if agent.characteristics.tier == 1]
    tier1_baseline_cost = sum([abm.agents[s].base_cost for s in tier1_supplier_ids])
    results_with_switching['tier1_baseline_cost'] = tier1_baseline_cost

    # For results_no_switching
    results_no_switching['total_cost_by_tier'] = [
        get_costs_by_tier(abm_no_switch, p) for p in results_no_switching['period']
    ]
    results_no_switching['tier1_landed_cost'] = [
        costs.get(1, 0) for costs in results_no_switching['total_cost_by_tier']
    ]
    # Add tier1_baseline_cost column
    tier1_supplier_ids_no_switch = [s for s, agent in abm_no_switch.agents.items() if agent.characteristics.tier == 1]
    tier1_baseline_cost_no_switch = sum([abm_no_switch.agents[s].base_cost for s in tier1_supplier_ids_no_switch])
    results_no_switching['tier1_baseline_cost'] = tier1_baseline_cost_no_switch

    # Use results_with_switching and results_no_switching for metric extraction and plotting
    # Example: get_costs_by_tier(abm, p) for p in results_with_switching['period']
    # Example: get_costs_by_tier(abm_no_switch, p) for p in results_no_switching['period']

    # After running the simulation:
    # Preprocess results for visual supplier states (for both scenarios)
    results_with_switching = preprocess_visual_supplier_states(results_with_switching)
    results_no_switching = preprocess_visual_supplier_states(results_no_switching)

    # 7. Visualizations
    # from dissertation_visualizations import plot_landed_and_cumulative_cost_analysis, extract_tariff_schedule
    # tariff_schedule = extract_tariff_schedule(abm)
    # plot_landed_and_cumulative_cost_analysis(
    #     results_with_switching,
    #     save_path="landed_cost_with_switching.png",
    #     tariff_schedule=tariff_schedule,
    #     title="Landed Cost Analysis (With Switching)"
    # )
    # plot_landed_and_cumulative_cost_analysis(
    #     results_no_switching,
    #     save_path="landed_cost_no_switching.png",
    #     tariff_schedule=tariff_schedule,
    #     title="Landed Cost Analysis (No Switching)"
    # )
    # print("\nSaved: landed_cost_with_switching.png and landed_cost_no_switching.png")
    
    # 8. Print results
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Final period results:")
    print(f"  Operational suppliers: {results_with_switching['operational'].iloc[-1]}")
    print(f"  Affected suppliers: {results_with_switching['affected'].iloc[-1]}")
    print(f"  Switched suppliers: {results_with_switching['switched'].iloc[-1]}")
    print(f"  Base cost: {results_with_switching['base_cost'].iloc[-1]:.2f}")
    print(f"  Landed cost: {results_with_switching['landed_cost'].iloc[-1]:.2f}")
    print(f"  Effective shock: {results_with_switching['effective_shock'].iloc[-1]:.2%}")
    
    # 9. Identify critical suppliers
    critical_suppliers = abm.identify_critical_suppliers()
    print("\n=== CRITICAL SUPPLIERS ===")
    print("Top 5 Critical Suppliers:")
    print(critical_suppliers.head(5).to_string(index=False))
    
    # 10. Create visualizations
    print("\n=== Creating Dissertation Visualizations ===")
    # create_dissertation_visualizations(results_with_switching, abm, save_prefix="sim_test1_results")

    # --- NEW: Create visuals using model_visuals.py ---
    # Extract tariff phases for shading
    tariff_phases = None
    if tariff_schedule is not None:
        tariff_phases = [
            {'start': phase.start_period, 'end': phase.end_period, 'magnitude': phase.magnitude}
            for phase in tariff_schedule.phases
        ]

    # 1. Tier 1 Landed Cost (with/without switching)
    plot_tier1_landed_cost_analysis(
        results_with_switching, tariff_phases=tariff_phases, scenario_label='With Switching', filename='tier1_cost_switch.png'
    )

    # 1b. Full Network Landed Cost (recursive, all tiers)
    plot_full_network_landed_cost_analysis(
        results_with_switching, tariff_phases=tariff_phases, scenario_label='With Switching', filename='network_cost_switch.png'
    )

    # 2. Supplier State Evolution (Cumulative)
    plot_meaningful_supplier_state_evolution(
        results_with_switching, scenario_label='With Switching', filename='supplier_states_switch.png'
    )

    # 3. Network Cost Impact Graph (Before vs After)
    # Placeholder: You may need to adjust how you get these objects from your ABM
    # G_before = abm.network.copy()  # or however you get the pre-shock network
    # G_after = abm.network
    # cost_impact_before = {n: 0 for n in G_before.nodes()}
    # cost_impact_after = {n: getattr(abm.agents[n], 'cumulative_cost_impact', 0) for n in G_after.nodes()}
    # tier_dict = {n: abm.agents[n].characteristics.tier for n in abm.agents}
    # plot_network_cost_impact(
    #     G_before, G_after, cost_impact_before, cost_impact_after, tier_dict,
    #     scenario_label='With Switching', filename='network_cost_impact_with_switching.png'
    # )

    # --- Cost Impact Extraction: Use diagnostics_log for true cumulative impact ---
    # Build supplier_impact_df directly from agent objects after simulation
    supplier_impact_df = pd.DataFrame([
        {
            'supplier_name': supplier_id,
            'tier': agent.characteristics.tier,
            'country': agent.characteristics.country,
            'cumulative_cost_impact': sum(agent.cost_history) - agent.base_cost * len(agent.cost_history),
            'ever_switched': agent.ever_switched,
            'ever_affected': agent.ever_affected,
            'status': agent.status
        }
        for supplier_id, agent in abm.agents.items()
        if hasattr(agent, 'cost_history')
    ])

    # Debug printout: inspect agent attributes
    print("\n=== AGENT DEBUG INFO ===")
    for supplier_id, agent in abm.agents.items():
        print(f"{supplier_id}: base_cost={agent.base_cost}, cost_history={agent.cost_history}, ever_affected={agent.ever_affected}, ever_switched={agent.ever_switched}, status={agent.status}")

    print("\n=== SUPPLIER IMPACT DF (first 20 rows) ===")
    print(supplier_impact_df.head(20))
    print(supplier_impact_df[['supplier_name', 'cumulative_cost_impact', 'ever_affected', 'ever_switched', 'status']])

    # Tier aggregation
    tier_impact_df = (
        supplier_impact_df
        .groupby('tier')
        .agg(
            total_cost_impact=('cumulative_cost_impact', 'sum'),
            n_suppliers=('supplier_name', 'count'),
            n_switched=('ever_switched', 'sum'),
            n_affected=('ever_affected', 'sum')
        )
        .reset_index()
    )

    # Country aggregation
    geo_risk_df = (
        supplier_impact_df
        .groupby('country')
        .agg(
            total_cost_increase=('cumulative_cost_impact', 'sum'),
            n_suppliers=('supplier_name', 'count'),
            n_switched=('ever_switched', 'sum'),
            n_affected=('ever_affected', 'sum')
        )
        .reset_index()
    )

    # Debug: print the DataFrame and dtypes
    print(tier_impact_df)
    print(tier_impact_df.dtypes)

    # Pass counts, not percentages, and remove scenario_label from title
    plot_tier_level_impact(
        tier_impact_df, filename='tier_impact_switch.png', landed_cost_df=results_with_switching[['landed_cost', 'base_cost']]
    )
    plot_tier_level_impact(
        tier_impact_df, filename='tier_impact_log_switch.png', landed_cost_df=results_with_switching[['landed_cost', 'base_cost']]
    )
    plot_cumulative_cost_impact(
        supplier_impact_df, filename='supplier_impact_switch.png', top_n=10
    )
    plot_geographic_risk(
        geo_risk_df, filename='geo_risk_switch.png'
    )

    # 5. Cost Impact per Supplier
    # plot_switching_cost_impact(
    #     switching_cost_df, scenario_label='With Switching', filename='switching_cost_distribution_with_switching.png'
    # )

    # 7. Geographic Risk
    # Build geo_risk_df from real simulation results
    # geo_risk_df = (
    #     supplier_impact_df
    #     .groupby('country')
    #     .agg(
    #         avg_cost_increase=('cumulative_cost_impact', 'mean'),
    #         n_suppliers=('supplier_name', 'count'),
    #         n_switched=('supplier_name', lambda names: sum(abm.agents[name].ever_switched for name in names)),
    #         n_affected=('supplier_name', lambda names: sum(abm.agents[name].ever_affected for name in names)),
    #         n_in_transition=('status', lambda x: (x == SupplierStatus.IN_TRANSITION).sum()),
    #         total_cost_increase=('cumulative_cost_impact', 'sum')
    #     )
    #     .reset_index()
    # )

    # plot_geographic_risk(
    #     geo_risk_df, filename='geographic_risk_with_switching.png'
    # )
    # New: Geographic Heatmap
    # from model_visuals import plot_geographic_heatmap
    # plot_geographic_heatmap(
    #     geo_risk_df, value_col='total_cost_increase', filename='geographic_heatmap_with_switching.png', log_scale=True
    # )  # Commented out due to compatibility issues

    print("\nNew model_visuals.py plots saved to model outputs directory.")

    # --- NO SWITCHING VISUALIZATIONS ---
    print("\n=== Creating No-Switching Visualizations ===")
    
    # Build supplier_impact_df for no-switching scenario
    supplier_impact_df_no_switch = pd.DataFrame([
        {
            'supplier_name': supplier_id,
            'tier': agent.characteristics.tier,
            'country': agent.characteristics.country,
            'cumulative_cost_impact': sum(agent.cost_history) - agent.base_cost * len(agent.cost_history),
            'ever_switched': agent.ever_switched,
            'ever_affected': agent.ever_affected,
            'status': agent.status
        }
        for supplier_id, agent in abm_no_switch.agents.items()
        if hasattr(agent, 'cost_history')
    ])

    # Tier aggregation for no-switching
    tier_impact_df_no_switch = (
        supplier_impact_df_no_switch
        .groupby('tier')
        .agg(
            total_cost_impact=('cumulative_cost_impact', 'sum'),
            n_suppliers=('supplier_name', 'count'),
            n_switched=('ever_switched', 'sum'),
            n_affected=('ever_affected', 'sum')
        )
        .reset_index()
    )

    # Country aggregation for no-switching
    geo_risk_df_no_switch = (
        supplier_impact_df_no_switch
        .groupby('country')
        .agg(
            total_cost_increase=('cumulative_cost_impact', 'sum'),
            n_suppliers=('supplier_name', 'count'),
            n_switched=('ever_switched', 'sum'),
            n_affected=('ever_affected', 'sum')
        )
        .reset_index()
    )

    # 1. Tier 1 Landed Cost (no switching)
    plot_tier1_landed_cost_analysis(
        results_no_switching, tariff_phases=tariff_phases, scenario_label='No Switching', filename='tier1_cost_noswitch.png'
    )

    # 1b. Full Network Landed Cost (no switching)
    plot_full_network_landed_cost_analysis(
        results_no_switching, tariff_phases=tariff_phases, scenario_label='No Switching', filename='network_cost_noswitch.png'
    )

    # 2. Supplier State Evolution (Cumulative) - no switching
    plot_meaningful_supplier_state_evolution(
        results_no_switching, scenario_label='No Switching', filename='supplier_states_noswitch.png'
    )

    # 3. Tier Impact (no switching)
    plot_tier_level_impact(
        tier_impact_df_no_switch, filename='tier_impact_log_switch.png', landed_cost_df=results_no_switching[['landed_cost', 'base_cost']]
    )

    # 4. Cost Impact per Supplier (no switching)
    plot_cumulative_cost_impact(
        supplier_impact_df_no_switch, filename='supplier_impact_noswitch.png', top_n=10
    )

    # 5. Geographic Risk (no switching)
    plot_geographic_risk(
        geo_risk_df_no_switch, filename='geo_risk_noswitch.png'
    )

    # 6. Post-simulation network impact (no switching)
    plot_post_simulation_network_impact(abm_no_switch, filename='network_postsim_noswitch.png')

    print("\nNo-switching visualizations saved to model outputs directory.")

    # After preprocessing results for visual supplier states (for both scenarios)
    plot_supplier_state_heatmap(results_with_switching, filename='supplier_heatmap_switch.png')
    plot_supplier_state_switched_affected(results_with_switching, filename='supplier_switched_affected_switch.png')
    plot_supplier_state_heatmap(results_no_switching, filename='supplier_heatmap_noswitch.png')
    plot_supplier_state_switched_affected(results_no_switching, filename='supplier_switched_affected_noswitch.png')

    # === Combined Landed Cost Visuals: Switching vs No Switching ===
    plot_overlaid_landed_cost_comparison(
        results_with_switching, results_no_switching, filename='overlaid_landed_cost_comparison.png', tariff_phases=tariff_phases
    )
    plot_overlaid_tier1_landed_cost_comparison(
        results_with_switching, results_no_switching, filename='overlaid_tier1_landed_cost_comparison.png', tariff_phases=tariff_phases
    )
    print("\nCombined landed cost comparison plots saved: overlaid_landed_cost_comparison.png and overlaid_tier1_landed_cost_comparison.png")

    # 11. Save data
    results_with_switching.to_csv('sim_test1_timeseries.csv', index=False)
    results_no_switching.to_csv('sim_test1_timeseries_no_switching.csv', index=False)
    print(f"\nSimulation completed. Results saved to sim_test1_timeseries.csv and sim_test1_timeseries_no_switching.csv")
    print(f"12 dissertation visualizations saved (6 with switching, 6 without switching)")

    # --- NEW: Post-simulation network impact visual ---
    plot_post_simulation_network_impact(abm, filename='network_postsim_switch.png')
    print("Post-simulation network impact visual saved as network_postsim_switch.png")

    # Diagnostic printout: sample agent states after simulation
    print("\nSample Agent States After Simulation:\n")
    for i, agent in enumerate(abm.agents.values()):
        if i > 10:
            break
        print(f"Agent {agent.id} | Country: {agent.characteristics.country} | Tier: {agent.characteristics.tier}")
        print(f"  Base Cost: {agent.base_cost}")
        print(f"  Cost History (len={len(agent.cost_history)}): {agent.cost_history}")
        print(f"  Ever Affected: {agent.ever_affected}")
    
    # --- COMPARISON SUMMARY ---
    print("\n" + "="*60)
    print("COMPARISON: With Switching vs No Switching")
    print("="*60)
    
    # Final period comparison
    with_switch_final = results_with_switching.iloc[-1]
    no_switch_final = results_no_switching.iloc[-1]
    
    print(f"Final Period Results:")
    print(f"  Landed Cost:")
    print(f"    With Switching: ${with_switch_final['landed_cost']:,.2f}")
    print(f"    No Switching:   ${no_switch_final['landed_cost']:,.2f}")
    print(f"    Difference:     ${no_switch_final['landed_cost'] - with_switch_final['landed_cost']:,.2f}")
    
    print(f"  Switched Suppliers:")
    print(f"    With Switching: {with_switch_final['switched']}")
    print(f"    No Switching:   {no_switch_final['switched']}")
    
    print(f"  Affected Suppliers:")
    print(f"    With Switching: {with_switch_final['affected']}")
    print(f"    No Switching:   {no_switch_final['affected']}")
    
    print(f"  Effective Shock:")
    print(f"    With Switching: {with_switch_final['effective_shock']:.2%}")
    print(f"    No Switching:   {no_switch_final['effective_shock']:.2%}")
    
    # Calculate cost savings from switching
    cost_savings = no_switch_final['landed_cost'] - with_switch_final['landed_cost']
    cost_savings_pct = (cost_savings / with_switch_final['landed_cost']) * 100
    print(f"\nCost Savings from Switching: ${cost_savings:,.2f} ({cost_savings_pct:.1f}%)")
    
    print("="*60)

    # --- SENSITIVITY ANALYSIS METRIC EXTRACTION ---
    scenario_label = f"{selected_country} {year if 'year' in locals() else ''} {macro_shock.shock_type.name.title()}"
    extract_metrics(
        results_with_switching,
        scenario=scenario_label,
        switching='yes',
        inflation=shock_magnitude,
        scf=scf,
        pir=pir
    )
    extract_metrics(
        results_no_switching,
        scenario=scenario_label,
        switching='no',
        inflation=shock_magnitude,
        scf=scf,
        pir=pir
    )

if __name__ == "__main__":
    main() 