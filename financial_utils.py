import numpy as np
import pandas as pd
from typing import List, Optional, Union

def calculate_growth_trend_slope(values: Union[List[float], np.ndarray]) -> Optional[float]:
    """
    Calculate growth as the normalized slope of a linear trend fitted to the values.
    Normalization (slope / avg_value) provides relative growth for comparability.
    
    - Handles negatives/zeros/NaNs robustly.
    - Suitable for currency (revenue, profits) and ratios (ROE, P/E).
    - For annual (4 reports) or quarterly (2 reports) sequences.
    
    Formula: Fit y = mx + b, then return m / avg(y) for relative growth.
    - Positive: Growing/upward trend.
    - Negative: Declining/downward trend.
    - Near-zero: Stable/no clear trend.
    
    Args:
        values (Union[List[float], np.ndarray]): Sequence of values (e.g., [val1, val2, val3, val4]).
    
    Returns:
        Optional[float]: Normalized slope (relative growth), or None if insufficient data.
    
    Examples:
        - Revenue [100, 120]: ~0.182 (18.2% growth)
        - Profit [0, 10]: 2.0 (200% growth approximation)
        - ROE [-0.05, 0.02]: ~0.033 (improving trend)
    """
    if values is None or len(values) < 2:
        return None
    
    # Prepare data
    x = np.arange(1, len(values) + 1)
    y = np.array(values)
    
    # Filter out NaN/inf values
    valid_mask = ~np.isnan(y) & ~np.isinf(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(y_valid) < 2:
        return None
    
    try:
        # Fit linear trend
        slope, _ = np.polyfit(x_valid, y_valid, 1)
        
        # Normalize by average value for relative growth
        avg_value = np.mean(y_valid)
        if avg_value != 0:
            relative_slope = slope / avg_value
            return relative_slope
        else:
            # Fallback: If average is zero, return absolute slope or infinity
            return slope if slope != 0 else float('inf')
    except (np.linalg.LinAlgError, ValueError):
        return 0.0  # No trend