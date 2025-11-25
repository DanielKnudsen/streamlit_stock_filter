# URL Query Parameter Filtering for Stock Analysis App

## Overview

The Streamlit stock analysis app supports URL query parameter filtering, allowing users to share specific filter combinations via links. This enables blog posts and articles to link directly to filtered stock results, improving user experience and content discoverability.

## How It Works

When a user clicks a link with filter parameters, the app automatically:
1. Parses the URL query parameters on page load
2. Applies the specified filters to sliders, pills, and ticker inputs
3. Displays the filtered results immediately

## Parameter Format

Parameters are appended to the app URL using standard query string format:

```
https://your-app.streamlit.app?parameter1=value1&parameter2=value2&parameter3=value3
```

**Important Note:** Sector names are in Swedish. Valid sectors include:
- `Dagligvaror` (Consumer Goods)
- `Hälsovård` (Healthcare) 
- `Industri` (Industrials)
- `Sällanköpsvaror` (Discretionary Goods)

### Parameter Types

#### Slider Parameters
- **Format:** `slider_{ratio_name}={min_value},{max_value}`
- **Example:** `slider_ROE_ttm=10,25` (filters ROE between 10% and 25%)
- **Multiple values:** Comma-separated for range sliders

#### Pill/Filter Parameters
- **Format:** `pills_{category}={value1},{value2},{value3}`
- **Example:** `pills_Sektor=Hälsovård,Industri` (filters to Healthcare and Industrials sectors)
- **Multiple values:** Comma-separated for multi-select pills

#### Ticker Input
- **Format:** `ticker_input={ticker_symbol}`
- **Example:** `ticker_input=ERIC.ST` (pre-fills ticker search)

## Available Parameters

### Financial Ratio Sliders
All TTM (Trailing Twelve Months) ratios support filtering:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `slider_ROE_ttm` | Return on Equity | `slider_ROE_ttm=15,30` |
| `slider_ROA_ttm` | Return on Assets | `slider_ROA_ttm=5,15` |
| `slider_Gross_Margin_ttm` | Gross Margin | `slider_Gross_Margin_ttm=20,40` |
| `slider_Operating_Margin_ttm` | Operating Margin | `slider_Operating_Margin_ttm=10,25` |
| `slider_Net_Margin_ttm` | Net Margin | `slider_Net_Margin_ttm=5,20` |
| `slider_EPS_ttm` | Earnings Per Share | `slider_EPS_ttm=1,10` |
| `slider_PE_Ratio_ttm` | Price/Earnings Ratio | `slider_PE_Ratio_ttm=10,25` |
| `slider_PB_Ratio_ttm` | Price/Book Ratio | `slider_PB_Ratio_ttm=1,3` |
| `slider_EV_EBITDA_ttm` | EV/EBITDA Ratio | `slider_EV_EBITDA_ttm=8,15` |
| `slider_Debt_Equity_ttm` | Debt/Equity Ratio | `slider_Debt_Equity_ttm=0,1` |
| `slider_Current_Ratio_ttm` | Current Ratio | `slider_Current_Ratio_ttm=1,3` |
| `slider_Quick_Ratio_ttm` | Quick Ratio | `slider_Quick_Ratio_ttm=0.5,2` |
| `slider_ROIC_ttm` | Return on Invested Capital | `slider_ROIC_ttm=10,25` |
| `slider_ROCE_ttm` | Return on Capital Employed | `slider_ROCE_ttm=12,30` |

### Growth Metrics
| Parameter | Description | Example |
|-----------|-------------|---------|
| `slider_Revenue_Growth_3Y` | 3-Year Revenue Growth | `slider_Revenue_Growth_3Y=10,25` |
| `slider_EPS_Growth_3Y` | 3-Year EPS Growth | `slider_EPS_Growth_3Y=15,30` |
| `slider_Book_Value_Growth_3Y` | 3-Year Book Value Growth | `slider_Book_Value_Growth_3Y=5,20` |

### Market Data
| Parameter | Description | Example |
|-----------|-------------|---------|
| `slider_Market_Cap` | Market Capitalization (MSEK) | `slider_Market_Cap=1000,10000` |
| `slider_Price` | Stock Price (SEK) | `slider_Price=50,500` |
| `slider_Volume_Avg_3M` | Average 3-Month Volume | `slider_Volume_Avg_3M=10000,100000` |

### Categorical Filters (Pills)
| Parameter | Description | Values |
|-----------|-------------|---------|
| `pills_Sektor` | Stock Sectors (Swedish) | `Dagligvaror`, `Hälsovård`, `Industri`, `Sällanköpsvaror` |
| `pills_Lista` | Stock Exchange Lists | `Large Cap`, `Mid Cap`, `Small Cap` |
| `pills_country` | Country | `Sweden` (primarily) |
| `pills_currency` | Currency | `SEK` |

## Examples

### High-Quality Growth Stocks
Filter for companies with strong profitability and growth:
```
?slider_ROE_ttm=15,40&slider_ROA_ttm=8,25&slider_Revenue_Growth_3Y=15,50&pills_Sektor=Hälsovård,Industri
```

### Value Investing Screen
Look for undervalued companies with solid fundamentals:
```
?slider_PE_Ratio_ttm=8,18&slider_PB_Ratio_ttm=0.8,2.5&slider_Debt_Equity_ttm=0,0.5&slider_Market_Cap=500,5000
```

### Defensive Stocks
Companies with strong balance sheets and stable margins:
```
?slider_Current_Ratio_ttm=1.5,4&slider_Debt_Equity_ttm=0,0.3&slider_Net_Margin_ttm=8,25&pills_Sektor=Dagligvaror,Sällanköpsvaror
```

### Technology Sector Deep Dive
Focus on a specific sector with ticker pre-filled:
```
?pills_Sektor=Industri&slider_Market_Cap=1000,50000&ticker_input=ERIC.ST
```

## Generating Shareable URLs

### Manual URL Creation
1. Apply your desired filters in the app
2. Click the "🔗 Generera delbar länk" button
3. Copy the generated URL from the code block
4. Share the URL in blog posts, articles, or social media

### URL Structure
```
https://your-app.streamlit.app?{filter_parameters}
```

Replace `https://your-app.streamlit.app` with your actual deployed app URL.

## Technical Implementation

### URL Parsing Logic
The app parses URL parameters on initialization:

```python
# Parse URL parameters
query_params = st.query_params
for key, value in query_params.items():
    if key.startswith(('slider_', 'pills_')) or key == 'ticker_input':
        # Apply to session state
        if isinstance(value, str) and ',' in value:
            st.session_state[key] = value.split(',')
        else:
            st.session_state[key] = value
```

### Parameter Validation
- Invalid parameters are ignored (no errors thrown)
- Only recognized parameter names are processed
- Values are converted to appropriate types (lists for multi-select, floats for ranges)

### Session State Integration
Parameters are applied to Streamlit session state, ensuring they work with the existing filter system without conflicts.

## Best Practices

### For Bloggers/Content Creators
1. **Test URLs thoroughly** before publishing
2. **Use descriptive combinations** that tell a clear investment story
3. **Include context** in your article about what the filters represent
4. **Update URLs** when filter ranges change significantly

### For Users
1. **Bookmark useful filter combinations** for quick access
2. **Share interesting findings** with colleagues or friends
3. **Experiment with parameters** to discover new investment ideas

## Future Enhancements

Potential improvements to the URL parameter system:
- Short URL generation for cleaner links
- Parameter validation with user feedback
- Saved filter templates with names
- Integration with social sharing buttons
- URL parameter presets for common strategies

---

*This documentation covers the URL query parameter filtering system implemented in the Swedish stock analysis platform. For technical support or feature requests, refer to the project repository.*