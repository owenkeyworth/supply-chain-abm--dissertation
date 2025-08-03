"""
# Data Consolidation, Cleaning, Imputation, Filtering, and Analysis Pipeline

This script demonstrates an **iterative, decision-driven approach** to supply chain trade data preparation and analysis. Each section is clearly marked and explained, showing the rationale for every major step and threshold.
"""

import pandas as pd
import numpy as np
import logging
import re
import os
from pathlib import Path
from typing import Tuple

# ========== 0. DATA CONSOLIDATION FROM MULTIPLE SHEETS ==========

def consolidate_trade_data(input_file, sheets, output_file=None):
    """
    ## Step 1: Data Consolidation
    Consolidates data from multiple sheets into a single DataFrame.
    """
    print("\n=== Consolidating Trade Data ===")
    
    # Column mappings for standardization
    value_columns = {
        'Total value in USD': 'value_usd',
        'Total Value USD': 'value_usd',
        'Total Value USD.1': 'value_usd'
    }
    weight_columns = {
        'Total weight in Kg': 'weight_kg',
        'Weight KG': 'weight_kg',
        'weight_kg': 'weight_kg'
    }
    hs_code_columns = {
        'HS Code': 'hs_code',
        'HS Code Hop 1': 'hs_code',
        'HS Code Hop 2': 'hs_code'
    }
    quantity_columns = ['Quantity', 'Quantity.1']
    unit_columns = ['Unit', 'Unit.1']
    description_priority = {
        1: ['HS Codes Hop 1 Description', 'HS Codes Hop 2 Description', 'HS Code Description'],
        2: ['HS Codes Hop 2 Description', 'HS Codes Hop 1 Description', 'HS Code Description'],
        3: ['HS Codes Hop 3 Description', 'HS Codes Hop 2 Description', 'HS Codes Hop 1 Description'],
        4: ['HS Codes Hop 2 Description', 'HS Codes Hop 1 Description', 'HS Code Description']
    }
    
    all_data = []
    
    for idx, sheet in enumerate(sheets, 1):
        print(f"Reading sheet: {sheet}")
        df = pd.read_excel(input_file, sheet_name=sheet)
        df['hop_level'] = idx
        df['start_date'] = pd.NA
        df['end_date'] = pd.NA
        
        if 'Date' in df.columns:  # Hop 2 case
            df['start_date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df['end_date'] = df['start_date']
        else:  # Hop 1, 3, 4 case
            if 'Start Date' in df.columns:
                df['start_date'] = pd.to_datetime(df['Start Date'], dayfirst=True, errors='coerce')
            if 'End Date' in df.columns:
                df['end_date'] = pd.to_datetime(df['End Date'], dayfirst=True, errors='coerce')
            df['start_date'] = df['start_date'].fillna(df['end_date'])
            df['end_date'] = df['end_date'].fillna(df['start_date'])
        
        # Format dates as DD/MM/YYYY without time
        if not df['start_date'].isna().all():
            df['start_date'] = df['start_date'].dt.strftime('%d/%m/%Y')
        if not df['end_date'].isna().all():
            df['end_date'] = df['end_date'].dt.strftime('%d/%m/%Y')
        
        # Rest of the column processing
        for old_col, new_col in value_columns.items():
            if old_col in df.columns:
                if new_col not in df.columns:
                    df[new_col] = df[old_col]
                else:
                    df[new_col] = df[new_col].fillna(df[old_col])
        
        for old_col, new_col in weight_columns.items():
            if old_col in df.columns:
                if new_col not in df.columns:
                    df[new_col] = df[old_col]
                else:
                    df[new_col] = df[new_col].fillna(df[old_col])
        
        for old_col, new_col in hs_code_columns.items():
            if old_col in df.columns:
                if new_col not in df.columns:
                    df[new_col] = df[old_col]
                else:
                    df[new_col] = df[new_col].fillna(df[old_col])
        
        df['quantity'] = pd.NA
        for col in quantity_columns:
            if col in df.columns:
                if df['quantity'].isna().all():
                    df['quantity'] = df[col]
                else:
                    df['quantity'] = df['quantity'].fillna(df[col])
        
        df['unit'] = pd.NA
        for col in unit_columns:
            if col in df.columns:
                if df['unit'].isna().all():
                    df['unit'] = df[col]
                else:
                    df['unit'] = df['unit'].fillna(df[col])
        
        df['product_description'] = pd.NA
        desc_cols = description_priority.get(idx, [])
        for col in desc_cols:
            if col in df.columns:
                if df['product_description'].isna().all():
                    df['product_description'] = df[col]
                else:
                    df['product_description'] = df['product_description'].fillna(df[col])
        
        essential_columns = [
            'hs_code',
            'product_description',
            'importer',
            'importer_country',
            'supplier',
            'supplier_country',
            'value_usd',
            'quantity',
            'unit',
            'weight_kg',
            'start_date',
            'end_date',
            'hop_level'
        ]
        
        column_mapping = {
            'Importer': 'importer',
            'Importer Country': 'importer_country',
            'Supplier': 'supplier',
            'Supplier Country': 'supplier_country'
        }
        df = df.rename(columns=column_mapping)
        
        for col in essential_columns:
            if col not in df.columns:
                df[col] = pd.NA
        
        df = df[essential_columns]
        
        if 'hs_code' in df.columns:
            df['hs_code'] = df['hs_code'].astype(str).str.replace('.0', '', regex=False)
        
        all_data.append(df)
    
    consolidated_df = pd.concat(all_data, ignore_index=True)
    print(f"Consolidated data shape: {consolidated_df.shape}")
    
    if output_file:
        consolidated_df.to_excel(output_file, index=False)
        print(f"Consolidated data saved to: {output_file}")
    
    return consolidated_df

# ========== 1. DATA LOADING & CONSOLIDATION ==========

def load_data(input_file):
    """
    ## Step 1: Data Loading
    Loads the raw or consolidated trade data from Excel or CSV.
    """
    print("\n=== Loading Data ===")
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    return df

# ========== 2. INITIAL DATA CLEANING ==========

def print_missing_stats(df, stage_name=""):
    """
    Prints missing value statistics for each column at a given stage.
    """
    print(f"\n=== Missing Values Report {stage_name} ===")
    print(f"Total rows: {len(df)}")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    stats = pd.DataFrame({
        'Missing Values': missing,
        'Missing %': missing_pct
    })
    print(stats[stats['Missing Values'] > 0].round(2))

def standardize_company_name(text):
    """
    Standardizes company names: removes non-alphanumeric chars, converts to lowercase.
    """
    if pd.isna(text):
        return text
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    return cleaned.lower().strip()

def standardize_text(text):
    """
    Standardizes text fields: strips spaces, handles NaN, consistent capitalization.
    """
    if pd.isna(text):
        return text
    return str(text).strip().title()

def clean_trade_data(df):
    """
    ## Step 2: Initial Data Cleaning
    - Removes duplicates
    - Drops rows with missing supplier/importer/country
    - Standardizes company names and text fields
    - Moves weight_kg to quantity (if quantity is null), sets unit to 'kg', and drops weight_kg
    - Reports on logical consistency and missing values
    """
    print("\n=== Initial Data Cleaning ===")
    print_missing_stats(df, "INITIAL STATE")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop rows with missing essentials
    df = df.dropna(subset=['supplier', 'importer', 'supplier_country', 'importer_country'])
    
    # Standardize company names
    df['supplier'] = df['supplier'].apply(standardize_company_name)
    df['importer'] = df['importer'].apply(standardize_company_name)
    
    # Standardize other text fields
    text_columns = ['supplier_country', 'importer_country', 'unit']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(standardize_text)
    
    # Move weight_kg to quantity if quantity is null, set unit to 'kg'
    if 'weight_kg' in df.columns:
        mask = df['weight_kg'].notnull() & (df['quantity'].isnull())
        df.loc[mask, 'quantity'] = df.loc[mask, 'weight_kg']
        df.loc[mask, 'unit'] = 'kg'
        # Drop the weight_kg column
        df = df.drop(columns=['weight_kg'])
    
    # Logical consistency checks
    print("\n=== Logical Consistency Checks ===")
    if 'value_usd' in df.columns:
        print(f"Rows with non-positive values: {(df['value_usd'] <= 0).sum()}")
    if 'quantity' in df.columns:
        print(f"Rows with non-positive quantities: {(df['quantity'] <= 0).sum()}")
    
    # Format dates in British format without time
    if 'start_date' in df.columns and 'end_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        
        # Format dates as DD/MM/YYYY
        df['start_date'] = df['start_date'].dt.strftime('%d/%m/%Y')
        df['end_date'] = df['end_date'].dt.strftime('%d/%m/%Y')
        
        # Check for invalid date ranges (after converting back to datetime for comparison)
        start_dates = pd.to_datetime(df['start_date'], format='%d/%m/%Y', errors='coerce')
        end_dates = pd.to_datetime(df['end_date'], format='%d/%m/%Y', errors='coerce')
        invalid_dates = (start_dates > end_dates)
        print(f"Rows with invalid date ranges: {invalid_dates.sum()}")
    
    print_missing_stats(df, "AFTER CLEANING")
    return df

# ========== 3. VERIFICATION OF CLEANING ==========

def verify_cleaning(original, cleaned):
    """
    ## Step 3: Verification of Cleaning
    Compares before/after cleaning for row counts, standardization, value ranges, duplicates, and missing values.
    """
    print("\n=== Cleaning Verification ===")
    print(f"Original rows: {len(original)} | Cleaned rows: {len(cleaned)} | Rows removed: {len(original) - len(cleaned)}")
    for col in ['supplier_country', 'importer_country', 'unit']:
        if col in cleaned.columns:
            print(f"Unique {col} values: {sorted(cleaned[col].dropna().unique())}")
    print("Sample of supplier names:")
    print(cleaned['supplier'].head(10))
    print("\nValue USD range:")
    if 'value_usd' in cleaned.columns:
        print(f"Min: {cleaned['value_usd'].min()} | Max: {cleaned['value_usd'].max()}")
    if 'quantity' in cleaned.columns:
        print(f"Min: {cleaned['quantity'].min()} | Max: {cleaned['quantity'].max()}")
    print(f"Remaining duplicates: {cleaned.duplicated().sum()}")
    print_missing_stats(cleaned, "CLEANED DATA")

# ========== 4. MISSING VALUE ANALYSIS ==========

def analyze_missing_values(df):
    """
    ## Step 4: Missing Value Analysis
    Summarizes missingness by column, supplier, product, etc.
    """
    print("\n=== Missing Value Analysis ===")
    print_missing_stats(df, "MISSING VALUE ANALYSIS")
    # By supplier
    print("\nMissing by supplier:")
    print(df.groupby('supplier')['value_usd'].apply(lambda x: x.isna().sum()).sort_values(ascending=False).head())
    # By HS code
    if 'hs_code' in df.columns:
        print("\nMissing by HS code:")
        print(df.groupby('hs_code')['value_usd'].apply(lambda x: x.isna().sum()).sort_values(ascending=False).head())

# ========== 5. DATA IMPUTATION ==========

def impute_value_usd(df):
    """
    Impute missing value_usd using medians by HS code and supplier country.
    """
    print("\n=== Imputing value_usd ===")
    df_imp = df.copy()
    medians_detailed = df_imp.groupby(['hs_code', 'supplier_country'])['value_usd'].median()
    medians_hs = df_imp.groupby('hs_code')['value_usd'].median()
    median_all = df_imp['value_usd'].median()
    for idx in df_imp[df_imp['value_usd'].isna()].index:
        hs = df_imp.loc[idx, 'hs_code']
        country = df_imp.loc[idx, 'supplier_country']
        try:
            value = medians_detailed[hs, country]
        except:
            try:
                value = medians_hs[hs]
            except:
                value = median_all
        df_imp.loc[idx, 'value_usd'] = value
    print(f"value_usd imputation complete. {df_imp['value_usd'].isna().sum()} missing values remain.")
    return df_imp

def standardize_units(df):
    """
    Standardize units and impute missing units based on HS code patterns.
    """
    print("\n=== Standardizing and imputing units ===")
    df_imp = df.copy()
    unit_map = {}
    for hs in df_imp['hs_code'].unique():
        units = df_imp[df_imp['hs_code'] == hs]['unit'].dropna()
        if not units.empty:
            unit_map[hs] = units.mode().iloc[0] if not units.mode().empty else None
    most_common_unit = df_imp['unit'].mode().iloc[0] if not df_imp['unit'].mode().empty else 'Kilogram'
    for idx in df_imp[df_imp['unit'].isna()].index:
        hs = df_imp.loc[idx, 'hs_code']
        df_imp.loc[idx, 'unit'] = unit_map.get(hs, most_common_unit)
    print(f"unit imputation complete. {df_imp['unit'].isna().sum()} missing values remain.")
    return df_imp

def impute_quantity(df):
    """
    Impute missing quantity using value_usd and average unit value (by hs_code and unit, then hs_code, then global),
    then fallback to medians by hs_code and supplier country, then global median. Track imputation source.
    """
    print("\n=== Imputing quantity ===")
    df_imp = df.copy()
    df_imp['unit_value_source'] = None

    # Calculate unit value where both value_usd and quantity are present and quantity > 0
    valid_mask = (df_imp['value_usd'].notna()) & (df_imp['quantity'].notna()) & (df_imp['quantity'] > 0)
    df_imp['unit_value'] = None
    df_imp.loc[valid_mask, 'unit_value'] = df_imp.loc[valid_mask, 'value_usd'] / df_imp.loc[valid_mask, 'quantity']

    # Get average unit value by hs_code and unit, then by hs_code, then global
    avg_unit_value_hs_unit = df_imp[valid_mask].groupby(['hs_code', 'unit'])['unit_value'].median()
    avg_unit_value_hs = df_imp[valid_mask].groupby('hs_code')['unit_value'].median()
    global_avg_unit_value = df_imp.loc[valid_mask, 'unit_value'].median()

    # Impute using value_usd and average unit value
    for idx in df_imp[df_imp['quantity'].isna() & df_imp['value_usd'].notna()].index:
        hs = df_imp.loc[idx, 'hs_code']
        unit = df_imp.loc[idx, 'unit']
        value_usd = df_imp.loc[idx, 'value_usd']
        # Try hs_code + unit
        try:
            unit_value = avg_unit_value_hs_unit[hs, unit]
            source = 'product and unit level'
        except:
            try:
                unit_value = avg_unit_value_hs[hs]
                source = 'product level'
            except:
                unit_value = global_avg_unit_value
                source = 'global level'
        if unit_value and unit_value > 0:
            df_imp.loc[idx, 'quantity'] = value_usd / unit_value
            df_imp.loc[idx, 'unit_value_source'] = source

    # Fallback: median-based imputation for any still-missing quantities
    still_missing = df_imp['quantity'].isna()
    if still_missing.any():
        medians_detailed = df_imp.groupby(['hs_code', 'supplier_country'])['quantity'].median()
        medians_hs = df_imp.groupby('hs_code')['quantity'].median()
        median_all = df_imp['quantity'].median()
        for idx in df_imp[still_missing].index:
            hs = df_imp.loc[idx, 'hs_code']
            country = df_imp.loc[idx, 'supplier_country']
            try:
                value = medians_detailed[hs, country]
                source = 'product and supplier country level'
            except:
                try:
                    value = medians_hs[hs]
                    source = 'product level'
                except:
                    value = median_all
                    source = 'global level'
            df_imp.loc[idx, 'quantity'] = value
            df_imp.loc[idx, 'unit_value_source'] = source

    # If still null, set source to Expert Heuristics
    df_imp.loc[df_imp['quantity'].isna(), 'unit_value_source'] = 'Expert Heuristics'

    print(f"quantity imputation complete. {df_imp['quantity'].isna().sum()} missing values remain.")
    # Drop the temporary unit_value column
    if 'unit_value' in df_imp.columns:
        df_imp = df_imp.drop('unit_value', axis=1)

    # Add unit_cost column (value_usd / quantity)
    df_imp['unit_cost'] = df_imp['value_usd'] / df_imp['quantity']

    # Winsorize unit_cost: cap at 1st and 99th percentiles, set near-zero/zero to NaN
    unit_cost_valid = (df_imp['unit_cost'].notna()) & (df_imp['unit_cost'] > 0)
    if unit_cost_valid.any():
        lower = df_imp.loc[unit_cost_valid, 'unit_cost'].quantile(0.01)
        upper = df_imp.loc[unit_cost_valid, 'unit_cost'].quantile(0.99)
        df_imp.loc[df_imp['unit_cost'] < lower, 'unit_cost'] = lower
        df_imp.loc[df_imp['unit_cost'] > upper, 'unit_cost'] = upper
        df_imp.loc[df_imp['unit_cost'] <= 1e-6, 'unit_cost'] = float('nan')

    return df_imp

# ========== 6. FILTERING ==========

def filter_small_trades(df, min_threshold=100):
    """
    ## Step 6: Filtering
    Removes trades below a minimum value threshold.
    """
    print(f"\n=== Filtering trades below ${min_threshold} ===")
    original_total = len(df)
    original_value = df['value_usd'].sum()
    df_filtered = df[df['value_usd'] >= min_threshold]
    filtered_total = len(df_filtered)
    filtered_value = df_filtered['value_usd'].sum()
    print(f"Original records: {original_total:,} | Filtered records: {filtered_total:,} | Records removed: {original_total - filtered_total:,}")
    print(f"Original total value: ${original_value:,.2f} | Filtered total value: ${filtered_value:,.2f}")
    print(f"Value removed: ${original_value - filtered_value:,.2f} ({((original_value - filtered_value)/original_value)*100:.4f}%)")
    return df_filtered

# ========== 7. FINAL ANALYSIS ==========

def analyze_range_stats(df):
    """
    ## Step 7: Final Analysis
    Range stats, IQR, outlier analysis, small trade impact.
    """
    def get_range_stats(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return pd.Series({
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'Q1': Q1,
            'median': data.median(),
            'Q3': Q3,
            'IQR': IQR,
            'count': len(data)
        })
    print("\nOverall Range Statistics:")
    print("-----------------------")
    overall_stats = get_range_stats(df['value_usd'])
    for stat, value in overall_stats.items():
        if stat in ['min', 'max', 'range', 'Q1', 'median', 'Q3', 'IQR']:
            print(f"{stat:>8}: ${value:,.2f}")
        else:
            print(f"{stat:>8}: {value:,}")
    print("\nRange Statistics by Hop:")
    print("----------------------")
    for hop in sorted(df['hop_level'].unique()):
        print(f"\nHop {hop}:")
        hop_data = df[df['hop_level'] == hop]['value_usd']
        hop_stats = get_range_stats(hop_data)
        for stat, value in hop_stats.items():
            if stat in ['min', 'max', 'range', 'Q1', 'median', 'Q3', 'IQR']:
                print(f"{stat:>8}: ${value:,.2f}")
            else:
                print(f"{stat:>8}: {value:,}")
    print("\nData Distribution Analysis:")
    print("-------------------------")
    for hop in sorted(df['hop_level'].unique()):
        hop_data = df[df['hop_level'] == hop]['value_usd']
        Q1 = hop_data.quantile(0.25)
        Q3 = hop_data.quantile(0.75)
        IQR = Q3 - Q1
        within_iqr = hop_data[(hop_data >= Q1) & (hop_data <= Q3)]
        below_iqr = hop_data[hop_data < Q1]
        above_iqr = hop_data[hop_data > Q3]
        print(f"\nHop {hop} Distribution:")
        print(f"Below Q1 (${Q1:,.2f}): {len(below_iqr)} transactions ({len(below_iqr)/len(hop_data)*100:.1f}%)")
        print(f"Within IQR: {len(within_iqr)} transactions ({len(within_iqr)/len(hop_data)*100:.1f}%)")
        print(f"Above Q3 (${Q3:,.2f}): {len(above_iqr)} transactions ({len(above_iqr)/len(hop_data)*100:.1f}%)")

# ========== 8. NETWORK CHANGE ANALYSIS ==========

def analyze_network_changes(df_original, df_filtered):
    """
    ## Step 8: Network Change Analysis
    Compares the network before and after filtering/imputation.
    """
    print("\n=== Network Changes Analysis ===")
    original_edges = set(zip(df_original['supplier'], df_original['importer']))
    filtered_edges = set(zip(df_filtered['supplier'], df_filtered['importer']))
    lost_edges = original_edges - filtered_edges
    print(f"Original number of unique trade relationships: {len(original_edges)}")
    print(f"Filtered number of unique trade relationships: {len(filtered_edges)}")
    print(f"Number of lost trade relationships: {len(lost_edges)}")
    if lost_edges:
        print("\nLost Trade Relationships:")
        for supplier, importer in lost_edges:
            print(f"- {supplier} → {importer}")
    # Suppliers without buyers
    current_suppliers = set(df_filtered['supplier'])
    current_importers = set(df_filtered['importer'])
    suppliers_without_buyers = current_suppliers - current_importers
    if suppliers_without_buyers:
        print("\nSuppliers without any incoming trades (terminal nodes):")
        for supplier in suppliers_without_buyers:
            print(f"- {supplier}")
    # Importers without suppliers
    importers_without_suppliers = current_importers - current_suppliers
    if importers_without_suppliers:
        print("\nImporters without any outgoing trades (source nodes):")
        for importer in importers_without_suppliers:
            print(f"- {importer}")

# ========== 8. LOAD AND INTEGRATE ADDITIONAL DATA ==========

def load_inflation_forecasts(file_path: str) -> pd.DataFrame:
    """
    Load IMF inflation forecast data from the PCPIPCH sheet.
    """
    print("\n=== Loading Inflation Forecasts ===")
    
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name='PCPIPCH')
    print("Initial columns:", df.columns.tolist())
    
    # Convert numeric year columns to strings for consistency
    year_columns = [col for col in df.columns if isinstance(col, (int, float)) and 2020 <= int(col) <= 2030]
    column_mapping = {col: str(int(col)) for col in year_columns}
    df = df.rename(columns=column_mapping)
    
    # Rename the country column
    df = df.rename(columns={df.columns[0]: 'country'})
    
    # Keep only relevant columns
    columns_to_keep = ['country'] + [str(year) for year in range(2020, 2031)]
    df = df[['country'] + [col for col in columns_to_keep[1:] if col in df.columns]]
    
    print("Columns after processing:", df.columns.tolist())
    
    # Melt the dataframe to convert years to rows
    df_melted = pd.melt(
        df,
        id_vars=['country'],
        var_name='year',
        value_name='inflation_forecast'
    )
    
    # Convert 'no data' to NaN
    df_melted['inflation_forecast'] = pd.to_numeric(df_melted['inflation_forecast'], errors='coerce')
    
    print(f"Loaded inflation forecasts for {len(df_melted['country'].unique())} countries")
    print(f"Years available: {sorted(df_melted['year'].unique().tolist())}")
    
    df.columns = df.columns.map(str)
    
    return df_melted

def load_industry_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load supplier-industry mapping and industry profit margins from the new, clean file.
    Uses only the columns: supplier, supplier_country, industry, profit_margin.
    Returns the same DataFrame twice for compatibility with the integration logic.
    """
    print("\n=== Loading Industry Data (NEW FORMAT) ===")
    df = pd.read_excel(file_path, sheet_name='Industry Mapping and Margins')
    # Only keep the relevant columns
    df = df[['supplier', 'supplier_country', 'industry', 'profit_margin']].drop_duplicates()
    print(f"Loaded {len(df)} unique supplier-country-industry-profit_margin rows.")
    print(f"Sample:\n{df.head()}")
    return df, df  # Return as both supplier_industry and industry_margins for compatibility

def robust_standardize(name):
    if pd.isna(name):
        return name
    import re
    # Convert to string, encode to ASCII, decode, strip whitespace, lowercase
    standardized = str(name).encode('ascii', 'ignore').decode().strip().lower()
    # Replace non-breaking spaces and multiple spaces
    standardized = standardized.replace('\xa0', ' ').replace('  ', ' ')
    # Remove punctuation (except spaces)
    standardized = re.sub(r'[^\w\s]', '', standardized)
    # Replace multiple spaces with single space
    standardized = re.sub(r'\s+', ' ', standardized).strip()
    return standardized

def integrate_all_data(trade_file, margins_file, inflation_file):
    """
    Integrate clean trade data, margins/industry data, and inflation data into a single DataFrame.
    Ensures one row per trade, no duplicate supplier columns, and all inflation years included.
    """
    # 1. Load data
    trade = pd.read_excel(trade_file)
    margins = pd.read_excel(margins_file)
    inflation = pd.read_excel(inflation_file)
    inflation.columns = inflation.columns.map(str)  # Ensure all columns are strings

    # 2. Standardize and rename columns
    trade = trade.rename(columns={
        'uct_descri': 'product_description',
        'orter_cou': 'importer_country',
        'plier_cour': 'supplier_country',
    })
    margins = margins.rename(columns={
        'product': 'product_description',
    })
    inflation = inflation.rename(columns={
        'Country': 'country',
    })

    # 3. Robust standardize country and supplier names
    for col in ['supplier', 'supplier_country', 'importer', 'importer_country']:
        if col in trade.columns:
            trade[col] = trade[col].apply(robust_standardize)
        if col in margins.columns:
            margins[col] = margins[col].apply(robust_standardize)
    if 'country' in inflation.columns:
        inflation['country'] = inflation['country'].apply(robust_standardize)

    # 4. Deduplicate margins and inflation data
    margins = margins.drop_duplicates(subset=['supplier', 'supplier_country'])
    inflation = inflation.drop_duplicates(subset=['country'])

    # Add knitex 96 profit margin row if not present
    if not ((margins['supplier'].str.strip().str.lower() == 'knitex 96') & (margins['supplier_country'].str.strip().str.lower() == 'bulgaria')).any():
        knitex_row = {
            'supplier': robust_standardize('knitex 96'),
            'supplier_country': robust_standardize('Bulgaria'),
            'industry': 'Textiles - Knitted & Crocheted Fabric',  # or whatever is appropriate
            'profit_margin': 21.4
        }
        margins = pd.concat([margins, pd.DataFrame([knitex_row])], ignore_index=True)

    # 5. Clean inflation data: replace 'no data' with np.nan, convert to float, rename columns
    inflation = inflation.replace('no data', np.nan)
    for year in range(2020, 2031):
        col = str(year)
        if col in inflation.columns:
            inflation[col] = pd.to_numeric(inflation[col], errors='coerce')
            inflation = inflation.rename(columns={col: f'inflation_{year}'})

    # 6. Merge margins/industry onto trade data
    trade = trade.merge(
        margins[['supplier', 'supplier_country', 'industry', 'profit_margin']],
        on=['supplier', 'supplier_country'],
        how='left'
    )

    # 7. Merge all inflation columns at once
    inflation_cols = [col for col in inflation.columns if col.startswith('inflation_')]

    # Debug prints for USA
    print('--- USA in trade ---')
    print(trade[trade['supplier_country'] == 'united states of america'].shape[0])
    print('--- USA in inflation ---')
    print(inflation[inflation['country'] == 'united states of america'])
    print('--- Inflation columns ---')
    print([col for col in inflation.columns if col.startswith('inflation_')])

    trade = trade.merge(
        inflation[['country'] + inflation_cols],
        left_on='supplier_country',
        right_on='country',
        how='left'
    ).drop('country', axis=1)

    # 8. Final check: ensure row count matches original trade data
    print(f"Rows in original trade data: {pd.read_excel(trade_file).shape[0]}")
    print(f"Rows after full merge: {trade.shape[0]}")
    print(f"Inflation columns present: {[col for col in trade.columns if col.startswith('inflation_')]}")

    # Standardize 'unit' column: replace 'Kilogram' with 'kg'
    if 'unit' in trade.columns:
        trade['unit'] = trade['unit'].replace({'Kilogram': 'kg', 'kilogram': 'kg'})

    return trade

def convert_percent_columns_to_decimal(df):
    """
    Convert all inflation columns and profit_margin to decimals (divide by 100).
    This should be called after all files are merged and cleaned, just before saving/exporting.
    """
    # Convert profit_margin
    if 'profit_margin' in df.columns:
        df['profit_margin'] = df['profit_margin'] / 100.0
    # Convert all inflation columns
    for col in df.columns:
        if col.startswith('inflation_'):
            df[col] = df[col] / 100.0
    return df

# ========== MAIN PIPELINE ==========

if __name__ == "__main__":
    # --- File paths and sheet names (edit as needed) ---
    data_dir = Path(r"C:\Users\owenk\OneDrive\Documents\UCL\Dissertation\Agent Based Modelling\Data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    input_file = data_dir / "Finisterre_Trade_Data.xlsx"
    consolidated_file = processed_dir / "Finisterre_Trade_Data_Consolidated.xlsx"
    cleaned_file = processed_dir / "Finisterre_Trade_Data_Cleaned.xlsx"
    imputed_file = processed_dir / "Finisterre_Trade_Data_Imputed.xlsx"
    filtered_file = processed_dir / "Finisterre_Trade_Data_Filtered.xlsx"
    
    inflation_file = data_dir / "IFM Inflation Forecasts.xlsx"
    industry_file = data_dir / "Industry Mapping and Margins.xlsx"
    
    sheets = ['Hop1 cleaned', 'Hop2 cleaned', 'Hop3 cleaned', 'Hop4 cleaned']
    
    # --- Process trade data ---
    df_consolidated = consolidate_trade_data(input_file, sheets, output_file=consolidated_file)
    df_cleaned = clean_trade_data(df_consolidated)
    df_cleaned.to_excel(cleaned_file, index=False)
    
    verify_cleaning(df_consolidated, df_cleaned)
    analyze_missing_values(df_cleaned)
    
    df_imputed = impute_value_usd(df_cleaned)
    df_imputed = standardize_units(df_imputed)
    df_imputed = impute_quantity(df_imputed)
    df_imputed.to_excel(imputed_file, index=False)
    
    df_filtered = filter_small_trades(df_imputed, min_threshold=100)
    df_filtered.to_excel(filtered_file, index=False)
    
    analyze_range_stats(df_filtered)
    analyze_network_changes(df_imputed, df_filtered)
    
    # --- Load and integrate additional data ---
    inflation_data = load_inflation_forecasts(inflation_file)
    supplier_industry, industry_margins = load_industry_data(industry_file)
    
    # --- Create final integrated dataset ---
    integrated_data = integrate_all_data(
        filtered_file,
        industry_file,
        inflation_file
    )
    
    # --- Save processed data ---
    output_file = processed_dir / "integrated_trade_data.xlsx"
    integrated_data = convert_percent_columns_to_decimal(integrated_data)
    integrated_data.to_excel(output_file, index=False)
    print(f"\nFinal integrated dataset saved to: {output_file}")
    print(f"Final shape: {integrated_data.shape}")
