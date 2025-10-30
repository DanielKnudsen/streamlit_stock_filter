#!/usr/bin/env python3
"""
Usage examples for ConfigMappings in app.py

This shows how to replace brittle string manipulation with robust config-driven mappings.
"""

from config_mappings import load_mappings_from_config
import pandas as pd

def example_usage():
    """Examples of how to use ConfigMappings in your app.py"""
    
    # Load mappings (do this once in app.py after loading config)
    mappings = load_mappings_from_config("rank-config.yaml")
    
    print("=== ConfigMappings Usage Examples ===\n")
    
    # Example 1: Replace string manipulation for ratio-to-value mapping
    print("1. Ratio-to-Value Mapping:")
    ratio_rank_column = "ROE_trend_ratioRank"
    
    # OLD WAY (brittle):
    # ratio_value_old = ratio_rank_column.replace('_ratioRank', '_ratioValue')
    
    # NEW WAY (robust):
    ratio_value_new = mappings.get_corresponding_value_column(ratio_rank_column)
    
    print(f"   Input: {ratio_rank_column}")
    print(f"   Output: {ratio_value_new}")
    print(f"   ✅ No string manipulation needed!\n")
    
    # Example 2: Extract base ratio name safely
    print("2. Base Ratio Extraction:")
    complex_column = "PE_tal_ttm_ratioRank"
    
    # OLD WAY (brittle):
    # base_ratio_old = complex_column.replace('_ttm_ratioRank', '')
    
    # NEW WAY (robust):
    base_ratio_new = mappings.extract_ratio_base(complex_column)
    period_type = mappings.extract_period_type(complex_column)
    
    print(f"   Input: {complex_column}")
    print(f"   Base ratio: {base_ratio_new}")
    print(f"   Period type: {period_type}")
    print(f"   ✅ Works with any period/value type combination!\n")
    
    # Example 3: Get all ratios for a category
    print("3. Category-to-Ratios Mapping:")
    category = "Kvalitet"
    ratios = mappings.get_ratios_for_category(category)
    
    print(f"   Category: {category}")
    print(f"   Contains ratios: {ratios}")
    
    # Generate all column names for this category
    for period in ['trend', 'latest', 'ttm']:
        for ratio in ratios:
            rank_col = f"{ratio}_{period}_ratioRank"
            value_col = mappings.get_corresponding_value_column(rank_col)
            print(f"   - {rank_col} ↔ {value_col}")
    print()
    
    # Example 4: Display names and metadata
    print("4. Display Names and Metadata:")
    for ratio in ratios:
        display_name = mappings.get_display_name(ratio)
        higher_better = mappings.is_higher_better(ratio)
        print(f"   {ratio}: '{display_name}' (higher_is_better: {higher_better})")
    print()
    
    # Example 5: Period descriptions for UI
    print("5. Period Descriptions for UI:")
    for period in ['trend', 'latest', 'ttm']:
        description = mappings.get_period_description(period)
        print(f"   {period}: '{description}'")
    print()
    
    # Example 6: Replace complex category column generation
    print("6. Category Column Generation:")
    category_base = "Kvalitet"
    
    # Generate all category column variants
    for period in ['trend', 'latest', 'ttm']:
        cat_rank_col = f"{category_base}_{period}_catRank"
        print(f"   Category rank: {cat_rank_col}")
        
        # Get all ratios for this category and period
        for ratio in mappings.get_ratios_for_category(category_base):
            ratio_rank_col = f"{ratio}_{period}_ratioRank"
            ratio_value_col = mappings.get_corresponding_value_column(ratio_rank_col)
            higher_better = mappings.is_higher_better(ratio)
            
            print(f"     → {ratio_rank_col} (higher_better: {higher_better})")
            print(f"     → {ratio_value_col}")
    
    print("\n=== Ready to use in app.py! ===")

def migration_example():
    """Show how to migrate existing app.py code"""
    
    mappings = load_mappings_from_config("rank-config.yaml")
    
    print("\n=== Migration Examples ===\n")
    
    print("Before (brittle string manipulation):")
    print("```python")
    print("# OLD CODE - FRAGILE")
    print("r_data = f\"{r.replace('_trend_ratioRank', '_trend_ratioValue')}\"")
    print("base_ratio = ratio.replace('_trend_ratioRank', '')")
    print("category_name = col.replace('catRank', 'ratioRank')")
    print("```\n")
    
    print("After (robust config-driven):")
    print("```python")
    print("# NEW CODE - ROBUST")
    print("r_data = mappings.get_corresponding_value_column(r)")
    print("base_ratio = mappings.extract_ratio_base(ratio)")
    print("period_type = mappings.extract_period_type(col)")
    print("category_ratios = mappings.get_ratios_for_category(category_base)")
    print("higher_better = mappings.is_higher_better(base_ratio)")
    print("display_name = mappings.get_display_name(base_ratio)")
    print("```\n")
    
    print("Benefits:")
    print("✅ No more string manipulation bugs")
    print("✅ Config-driven - change config, everything updates")
    print("✅ Type safety and clear method names")
    print("✅ Built-in validation and error checking")
    print("✅ Easy to extend and maintain")

if __name__ == "__main__":
    example_usage()
    migration_example()