#!/usr/bin/env python3
"""
Test script for ConfigMappings class
Run this to verify the mappings work correctly with your rank-config.yaml
"""

from config_mappings import load_mappings_from_config

def test_config_mappings():
    """Test the ConfigMappings class with your config"""
    
    print("=== Testing ConfigMappings ===")
    
    # Load mappings from your config
    try:
        mappings = load_mappings_from_config("rank-config.yaml")
        print("✅ Successfully loaded ConfigMappings from rank-config.yaml")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Test basic functionality
    print("\n--- Debug Info ---")
    debug_info = mappings.debug_mappings()
    print(f"Ratio bases: {len(debug_info['ratio_bases'])} found")
    print(f"Category bases: {len(debug_info['category_bases'])} found")
    print(f"Period types: {debug_info['period_types']}")
    
    # Test ratio to value mapping
    print("\n--- Ratio-to-Value Mapping Test ---")
    test_rank_columns = ['ROE_long_trend_ratioRank', 'PE_tal_ttm_current_ratioRank', 'ROIC_ttm_momentum_ratioRank']
    
    for rank_col in test_rank_columns:
        value_col = mappings.get_corresponding_value_column(rank_col)
        if value_col:
            print(f"✅ {rank_col} → {value_col}")
        else:
            print(f"❌ No mapping found for {rank_col}")
    
    # Test ratio base extraction
    print("\n--- Ratio Base Extraction Test ---")
    test_columns = ['ROE_long_trend_ratioRank', 'Kvalitet_ttm_current_catRank', 'PE_tal_ttm_momentum_ratioValue']
    
    for col in test_columns:
        base = mappings.extract_ratio_base(col)
        period = mappings.extract_period_type(col)
        print(f"✅ {col} → base: '{base}', period: '{period}'")
    
    # Test category mappings
    print("\n--- Category Mappings Test ---")
    for category_base in list(mappings.category_bases)[:3]:  # Test first 3 categories
        ratios = mappings.get_ratios_for_category(category_base)
        print(f"✅ Category '{category_base}' contains ratios: {ratios}")
    
    # Test higher_is_better lookup
    print("\n--- Higher-is-Better Test ---")
    test_ratios = ['ROE', 'PE_tal', 'ROIC']
    for ratio in test_ratios:
        if ratio in mappings.ratio_bases:
            higher_better = mappings.is_higher_better(ratio)
            print(f"✅ {ratio}: higher_is_better = {higher_better}")
    
    # Test display names
    print("\n--- Display Names Test ---")
    for ratio in test_ratios:
        if ratio in mappings.ratio_bases:
            display_name = mappings.get_display_name(ratio)
            print(f"✅ {ratio} → '{display_name}'")
    
    # Test period descriptions
    print("\n--- Period Descriptions Test ---")
    for period in mappings.period_types:
        description = mappings.get_period_description(period)
        print(f"✅ {period} → '{description}'")
    
    # Validate config consistency
    print("\n--- Config Validation ---")
    issues = mappings.validate_config_consistency()
    if issues:
        print("❌ Config validation issues found:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"   - {issue}")
    else:
        print("✅ Config validation passed - no issues found")
    
    # Test the new convenience method: get_underlying_ratios_for_category_rank()
    print("\n--- New Method: get_underlying_ratios_for_category_rank() ---")
    print("This method takes a category rank column and returns all underlying ratios\n")
    
    # Test with first available category
    if mappings.category_bases:
        first_category = mappings.category_bases[0]
        test_category_rank = f"{first_category}_long_trend_catRank"
        
        try:
            result = mappings.get_underlying_ratios_for_category_rank(test_category_rank)
            print(f"✅ Input: '{test_category_rank}'")
            print(f"   Category: {result['category']}")
            print(f"   Period: {result['period']}")
            print(f"   Period Description: {result['period_description']}")
            print(f"   Underlying Ratios: {result['ratio_bases']}")
            print(f"   Value Columns ({len(result['ratio_value_columns'])}): {result['ratio_value_columns']}")
            print(f"   Rank Columns ({len(result['ratio_rank_columns'])}): {result['ratio_rank_columns']}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test with another period
        print("\n   Testing with ttm_current period:")
        test_category_rank_ttm = f"{first_category}_ttm_current_catRank"
        try:
            result = mappings.get_underlying_ratios_for_category_rank(test_category_rank_ttm)
            print(f"✅ Input: '{test_category_rank_ttm}'")
            print(f"   Underlying Ratios: {result['ratio_bases']}")
            print(f"   Sample Value Column: {result['ratio_value_columns'][0] if result['ratio_value_columns'] else 'N/A'}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n=== ConfigMappings Test Complete ===")

if __name__ == "__main__":
    test_config_mappings()