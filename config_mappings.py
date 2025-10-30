from typing import Dict, List, Optional, Set

class ConfigMappings:
    """
    Config-driven mappings for Swedish stock analysis periods and ratios.
    
    Supports three period types:
    - long_trend: 4-year annual trend (slope over 4 years)
    - ttm_current: Latest TTM values (current absolute performance)  
    - ttm_momentum: TTM momentum (consecutive TTM slope, e.g., Q3-2025 vs Q2-2025)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.ratio_definitions = config.get('ratio_definitions', {})
        self.category_ratios = config.get('category_ratios', {})
        self.kategorier = config.get('kategorier', {})
        self.display_names = config.get('display_names', {})
        self.cluster_config = config.get('cluster', {})
        
        # Period types extracted from cluster config
        self.period_types = list(self.cluster_config.keys())
        
        # Build comprehensive mappings
        self._build_mappings()
    
    def _build_mappings(self):
        """Build all necessary mappings from config structure"""
        
        # 1. Extract ratio base names from ratio_definitions
        self.ratio_bases = list(self.ratio_definitions.keys())
        
        # 2. Extract category base names from kategorier
        self.category_bases = list(self.kategorier.keys())
        
        # 3. Value types for ratios
        self.value_types = ['ratioValue', 'ratioRank']
        self.category_types = ['catRank', 'ratioRank']
        
        # 4. Build category to ratios mapping
        self._build_category_mappings()
        
        # 5. Build column name mappings
        self._build_column_mappings()
        
        # 6. Build cluster mappings
        self._build_cluster_mappings()
    
    def _build_category_mappings(self):
        """Build mappings between categories and their constituent ratios"""
        self.category_to_ratios = {}
        self.ratio_to_categories = {}
        
        # From kategorier section
        for category_base, ratios in self.kategorier.items():
            self.category_to_ratios[category_base] = ratios
            
            # Reverse mapping: ratio -> categories
            for ratio in ratios:
                if ratio not in self.ratio_to_categories:
                    self.ratio_to_categories[ratio] = []
                self.ratio_to_categories[ratio].append(category_base)
    
    def _build_column_mappings(self):
        """Build comprehensive column name mappings"""
        self.rank_to_value = {}
        self.value_to_rank = {}
        
        # For each ratio and period combination (new naming only)
        for ratio_base in self.ratio_bases:
            for period in self.period_types:
                rank_col = f"{ratio_base}_{period}_ratioRank"
                value_col = f"{ratio_base}_{period}_ratioValue"
                
                self.rank_to_value[rank_col] = value_col
                self.value_to_rank[value_col] = rank_col
    
    def _build_cluster_mappings(self):
        """Build cluster rank mappings"""
        # Map: long_trend -> long_trend_clusterRank, ttm_current -> ttm_current_clusterRank, etc.
        self.cluster_mappings = {period: f"{period}_clusterRank" for period in self.period_types}
        # Build list of all cluster columns with correct naming
        self.cluster_columns = list(self.cluster_mappings.values())
    
    # =====================================================================
    # PUBLIC METHODS FOR GETTING MAPPINGS
    # =====================================================================
    
    def get_corresponding_value_column(self, rank_column: str) -> Optional[str]:
        """
        Get the value column for a given rank column.
        
        Examples:
            'ROE_latest_ratioRank' -> 'ROE_latest_ratioValue'
            'ROE_ttm_current_ratioRank' -> 'ROE_ttm_current_ratioValue'
        """
        return self.rank_to_value.get(rank_column)
    
    def get_corresponding_rank_column(self, value_column: str) -> Optional[str]:
        """
        Get the rank column for a given value column.
        
        Examples:
            'ROE_latest_ratioValue' -> 'ROE_latest_ratioRank'
            'ROE_ttm_current_ratioValue' -> 'ROE_ttm_current_ratioRank'
        """
        return self.value_to_rank.get(value_column)
    
    def get_ratios_for_category(self, category_base: str) -> List[str]:
        """Get all ratios belonging to a category"""
        return self.category_to_ratios.get(category_base, [])
    
    def get_categories_for_ratio(self, ratio_base: str) -> List[str]:
        """Get all categories that include this ratio"""
        return self.ratio_to_categories.get(ratio_base, [])
    
    def get_category_columns(self, category_base: str, period_type: Optional[str] = None, 
                           category_type: str = 'catRank') -> Dict[str, str]:
        """
        Get all column variants for a category.
        
        Args:
            category_base: Base category name (e.g., 'Kvalitet')
            period_type: Specific period ('long_trend', 'ttm_current', 'ttm_momentum') or None for all
            category_type: 'catRank' or 'ratioRank'
            
        Returns:
            Dict mapping period -> full column name
        """
        result = {}
        periods = [period_type] if period_type else self.period_types
        
        for period in periods:
            col_name = f"{category_base}_{period}_{category_type}"
            result[period] = col_name
        
        return result
    
    def get_ratio_metadata(self, ratio_base: str) -> Dict:
        """Get metadata for a ratio from ratio_definitions"""
        return self.ratio_definitions.get(ratio_base, {})
    
    def is_higher_better(self, ratio_base: str) -> bool:
        """Check if higher values are better for this ratio"""
        metadata = self.get_ratio_metadata(ratio_base)
        return metadata.get('higher_is_better', True)
    
    def get_display_name(self, column_or_ratio: str) -> str:
        """Get Swedish display name for a column or ratio"""
        # Try direct lookup first
        if column_or_ratio in self.display_names:
            return self.display_names[column_or_ratio]
        
        # Try extracting base ratio name
        ratio_base = self.extract_ratio_base(column_or_ratio)
        if ratio_base in self.display_names:
            return self.display_names[ratio_base]
        
        # Fallback to the original name
        return column_or_ratio
    
    def extract_ratio_base(self, column_name: str) -> str:
        """
        Extract base ratio name from column.
        
        Examples:
            'ROE_long_trend_ratioRank' -> 'ROE'
            'Kvalitet_ttm_current_catRank' -> 'Kvalitet'
            'PE_tal_ttm_momentum_ratioValue' -> 'PE_tal'
        """
        # Try all known suffixes using current period types
        all_suffixes = []
        
        for period in self.period_types:
            for value_type in self.value_types + self.category_types:
                all_suffixes.append(f"_{period}_{value_type}")
        
        # Sort by length (longest first) to match most specific suffix
        all_suffixes.sort(key=len, reverse=True)
        
        for suffix in all_suffixes:
            if column_name.endswith(suffix):
                return column_name.replace(suffix, '')
        
        return column_name
    
    def extract_period_type(self, column_name: str) -> Optional[str]:
        """
        Extract period type from column name.
        
        Examples:
            'ROE_long_trend_ratioRank' -> 'long_trend'
            'ROE_ttm_current_ratioValue' -> 'ttm_current'
        """
        # Sort by length (longest first) to match most specific period first
        periods_sorted = sorted(self.period_types, key=len, reverse=True)
        
        for period in periods_sorted:
            if f"_{period}_" in column_name:
                return period
        
        return None
    
    def get_all_ratio_columns(self, ratio_base: str, include_historical: bool = False) -> Dict[str, List[str]]:
        """
        Get all column variants for a ratio.
        
        Returns:
            Dict with keys: 'values', 'ranks', 'historical' (if requested)
        """
        result = {'values': [], 'ranks': [], 'historical': []}
        
        # Current period columns (new naming only)
        for period in self.period_types:
            value_col = f"{ratio_base}_{period}_ratioValue"
            rank_col = f"{ratio_base}_{period}_ratioRank"
            
            result['values'].append(value_col)
            result['ranks'].append(rank_col)
        
        # Historical columns (if requested)
        if include_historical:
            # These would need to be detected from actual dataframe columns
            # For now, return common patterns
            historical_patterns = [
                f"{ratio_base}_year_",
                f"{ratio_base}_quarter_"
            ]
            result['historical'] = historical_patterns
        
        return result
    
    def validate_column_exists(self, column_name: str, available_columns: Set[str]) -> bool:
        """Check if a column exists in the available columns set"""
        return column_name in available_columns
    
    def get_missing_columns(self, required_columns: List[str], available_columns: Set[str]) -> List[str]:
        """Get list of required columns that are missing from available columns"""
        return [col for col in required_columns if col not in available_columns]
    
    def get_category_rank_column(self, category: str, period_type: str) -> str:
        """
        Get the category rank column name for a specific category and period.
        
        Args:
            category: Category base name (e.g., 'Kvalitet', 'Hälsa', 'Lönsamhet')
            period_type: The period ('long_trend', 'ttm_current', 'ttm_momentum')
        
        Returns:
            Full category rank column name in format: '{category}_{period}_catRank'
            
        Examples:
            >>> mappings.get_category_rank_column('Kvalitet', 'long_trend')
            'Kvalitet_long_trend_catRank'
            
            >>> mappings.get_category_rank_column('Hälsa', 'ttm_current')
            'Hälsa_ttm_current_catRank'
        """
        return f"{category}_{period_type}_catRank"
    
    def get_cluster_col_name(self, period_type: str) -> str:
        """
        Get the display column name for a cluster period from the config.
        
        Args:
            period_type: The period ('long_trend', 'ttm_current', 'ttm_momentum')
        
        Returns:
            The col_name value from cluster config for this period
            
        Examples:
            >>> mappings.get_cluster_col_name('long_trend')
            'Trend4år'
            
            >>> mappings.get_cluster_col_name('ttm_current')
            'TTM12M'
        """
        return self.cluster_config.get(period_type, {}).get('col_name', period_type)
    
    def get_period_description(self, period_type: str) -> str:
        """Get Swedish description of period type"""
        descriptions = {
            'long_trend': '4-års trend från årsrapporter',
            'ttm_current': 'Senaste TTM-värden', 
            'ttm_momentum': 'TTM-momentum (konsekutiva kvartal)'
        }
        return descriptions.get(period_type, period_type)
    
    def get_category_rank_columns_for_period(self, period_type: str) -> List[str]:
        """
        Get all category rank columns for a specific period.
        
        Args:
            period_type: The period ('long_trend', 'ttm_current', 'ttm_momentum')
        
        Returns:
            List of column names in format: '{category}_{period}_catRank'
            
        Examples:
            >>> mappings.get_category_rank_columns_for_period('long_trend')
            ['Kvalitet_long_trend_catRank', 'Hälsa_long_trend_catRank', ...]
        """
        return [f"{category}_{period_type}_catRank" for category in self.category_bases]
    
    def get_underlying_ratios_for_category_rank(self, category_rank_column: str) -> Dict[str, List[str]]:
        """
        Convenient method: Given a category rank column name, get all underlying ratio columns.
        
        This is the main method for navigating from a category rank to its constituent ratio values.
        
        Args:
            category_rank_column: Full category rank column name 
                                 (e.g., 'Finansiell_Hälsa_long_trend_catRank')
        
        Returns:
            Dict with keys:
                - 'category': The category base name (e.g., 'Finansiell_Hälsa')
                - 'period': The period type (e.g., 'long_trend', 'ttm_current')
                - 'ratio_bases': List of ratio base names (e.g., ['SoliditetGrad', 'Likviditet'])
                - 'ratio_value_columns': Full column names for ratio values
                - 'ratio_rank_columns': Full column names for ratio ranks
                - 'period_description': Swedish description of the period
        
        Examples:
            >>> mappings.get_underlying_ratios_for_category_rank('Finansiell_Hälsa_long_trend_catRank')
            {
                'category': 'Finansiell_Hälsa',
                'period': 'long_trend',
                'ratio_bases': ['SoliditetGrad', 'Likviditet', ...],
                'ratio_value_columns': ['SoliditetGrad_long_trend_ratioValue', ...],
                'ratio_rank_columns': ['SoliditetGrad_long_trend_ratioRank', ...],
                'period_description': '4-års trend från årsrapporter'
            }
        """
        # Extract category base and period from column name
        category_base = self.extract_ratio_base(category_rank_column)
        period_type = self.extract_period_type(category_rank_column)
        
        if not category_base or not period_type:
            raise ValueError(
                f"Could not extract category and period from '{category_rank_column}'. "
                f"Expected format: '{{category}}_{{period}}_catRank'"
            )
        
        # Get ratios for this category
        ratio_bases = self.get_ratios_for_category(category_base)
        if not ratio_bases:
            raise ValueError(
                f"Category '{category_base}' not found. Available categories: {self.category_bases}"
            )
        
        # Build full column names
        ratio_value_columns = [
            f"{ratio}_{period_type}_ratioValue" 
            for ratio in ratio_bases
        ]
        
        ratio_rank_columns = [
            f"{ratio}_{period_type}_ratioRank" 
            for ratio in ratio_bases
        ]
        
        return {
            'category': category_base,
            'period': period_type,
            'ratio_bases': ratio_bases,
            'ratio_value_columns': ratio_value_columns,
            'ratio_rank_columns': ratio_rank_columns,
            'period_description': self.get_period_description(period_type)
        }
    
    def get_cluster_rank_columns(self) -> List[str]:
        """
        Get all cluster rank columns across all periods.
        
        Returns:
            List of column names in format: '{period}_clusterRank'
            Examples: ['long_trend_clusterRank', 'ttm_current_clusterRank', 'ttm_momentum_clusterRank']
        """
        return self.cluster_columns
    
    def get_category_rank_columns(self) -> List[str]:
        """
        Get all category rank columns across all periods and categories.
        
        Returns:
            List of column names in format: '{category}_{period}_catRank'
            Examples: ['Kvalitet_long_trend_catRank', 'Hälsa_ttm_current_catRank', ...]
        """
        result = []
        for period in self.period_types:
            result.extend(self.get_category_rank_columns_for_period(period))
        return result
    
    def get_rank_score_columns(self) -> List[str]:
        """
        Get all rank score columns (both cluster rank and category rank columns).
        
        This is a convenience method that combines cluster ranks and category ranks.
        
        Returns:
            List of all rank score columns
        """
        return self.get_cluster_rank_columns() + self.get_category_rank_columns()
    
    # =====================================================================
    # DEBUGGING AND VALIDATION METHODS
    # =====================================================================
    
    def debug_mappings(self) -> Dict:
        """Return debug information about all mappings"""
        return {
            'ratio_bases': self.ratio_bases,
            'category_bases': self.category_bases,
            'period_types': self.period_types,
            'total_rank_to_value_mappings': len(self.rank_to_value),
            'sample_mappings': {
                'rank_to_value': dict(list(self.rank_to_value.items())[:3])
            }
        }
    
    def validate_config_consistency(self) -> List[str]:
        """Validate that config is internally consistent"""
        issues = []
        
        # Check that all ratios in kategorier exist in ratio_definitions
        for category, ratios in self.kategorier.items():
            for ratio in ratios:
                if ratio not in self.ratio_definitions:
                    issues.append(f"Ratio '{ratio}' in category '{category}' not found in ratio_definitions")
        
        # Check that all ratios in category_ratios have corresponding base ratios
        for category_rank, ratios in self.category_ratios.items():
            for ratio_rank in ratios.keys():
                ratio_base = self.extract_ratio_base(ratio_rank)
                if ratio_base not in self.ratio_definitions:
                    issues.append(f"Base ratio '{ratio_base}' from '{ratio_rank}' not found in ratio_definitions")
        
        return issues


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def load_mappings_from_config(config_path: str = "rank-config.yaml") -> ConfigMappings:
    """Load ConfigMappings from YAML config file"""
    from io_utils import load_yaml
    
    config = load_yaml(config_path)
    if not config:
        raise ValueError(f"Could not load config from {config_path}")
    
    return ConfigMappings(config)
