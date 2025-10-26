# Streamlit Stock Filter - AI Agent Instructions

## Project Overview
Swedish stock analysis platform using Stockholm Stock Exchange data with three main components:
- **Frontend:** Streamlit web app (`app.py`) for interactive stock screening
- **Data Pipeline:** Automated GitHub Actions workflows for data fetching and ranking capture  
- **Analysis Engine:** Quarterly earnings impact analysis with historical trend tracking

## Architecture & Data Flow
```
GitHub Actions → yfinance/CSV → rank.py → Streamlit App
       ↓                                      ↑
ranking_history.db ← quarterly_analysis.py ←──┘
```

### Key Components
- `app.py` (2090 lines): Main Streamlit interface with user auth, portfolio management
- `rank.py` (711 lines): Core ranking engine processing 700+ Swedish stocks
- `data_fetcher.py`: yfinance integration for Stockholm Exchange (.ST suffix)
- `ranking_history_tracker.py`: SQLite time-series database for trend analysis
- `quarterly_analysis.py`: Earnings impact detection and reporting

## Environment Detection Pattern
**Critical:** All scripts use dual-path environment detection:
```python
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
# data/local/ for development, data/remote/ for GitHub Actions
CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')
```

## Development Workflow

### Setup
```bash
# Use uv (not pip) - this project is uv-native
uv sync                           # Install from pyproject.toml
uv run streamlit run app.py       # Launch development server
```

### Key Commands
```bash
# Generate stock rankings (core functionality)
uv run python rank.py

# Quarterly analysis (earnings impact detection)  
uv run python quarterly_analysis.py

# Manual ranking history capture
uv run python daily_capture.py

# Run specific GitHub Actions workflow
# See .github/workflows/ for automation patterns
```

## Git LFS Integration
**Essential:** Large data files use Git LFS (see `.gitattributes`):
- `*.csv`, `*.pkl`, `*.db` are LFS-tracked
- GitHub Actions require explicit `lfs: true` and `git lfs pull`
- Database always stored in `data/local/` even in remote environment

## Configuration System
- `rank-config.yaml`: Core ranking parameters (SMA windows, CAGR settings, 305 lines)
- Swedish market focus: `.ST` suffix for all tickers
- 4-year historical data, quarterly updates

## Database Schema
`ranking_history.db` stores normalized time-series:
```sql
(ticker, date, dimension, value)
-- dimension: ROE_ttm, Total_Revenue_ttm, ranking_* 
```

## Authentication & Subscription System
**Two-tier authentication:** Supabase login + WordPress subscription validation
- `auth.py`: Supabase integration for user login/portfolio storage
- WordPress PaidMembershipPro plugin controls subscription status via Stripe
- App checks both Supabase authentication AND WordPress subscription validity
- Portfolio data stored in Supabase, subscription data from WordPress API

## Swedish Market Specifics
- All tickers append `.ST` for Stockholm Exchange
- Swedish language in UI elements and error messages
- Sector names in Swedish: "Sällanköpsvaror", "Telekommunikation", etc.
- Currency: SEK for revenue calculations

## GitHub Actions Patterns
1. **daily_update.yml**: Main data pipeline (weekdays 04:00 UTC)
2. **daily_ranking_capture.yml**: Quarterly trigger detection (04:30 UTC)  
3. **weekly_update.yml**: Full refresh (Sundays)

All workflows use environment variables for path detection and require PAT token for LFS operations.

## Common Patterns
- **Error handling:** Swedish error messages, graceful degradation
- **Progress tracking:** tqdm with environment-based disable
- **Data validation:** Extensive null checking for financial metrics
- **File operations:** Always use pathlib.Path for cross-platform compatibility

## Future Development Pipeline
Planned automated content creation system (see `planning/to_be_done/`):
- Trigger content generation from financial data changes
- WordPress API integration for auto-publishing
- All features designed for efficiency and low maintenance
- Content automation based on yfinance data updates and ranking changes

## Development Philosophy
**Discussion-First Approach:** Always discuss features/problems/bugs before implementation
1. Analyze the problem and current system impact
2. Agree on solution approach and implementation plan  
3. Only then proceed with code changes
4. Focus on maintainable, efficient solutions

**Notebook Cell Editing Rules:**
- When editing notebook cells, **always preserve existing code** and make only targeted edits
- **Never rewrite entire cells** unless explicitly asked to replace the whole cell
- Read the full cell context before making changes
- Use surgical edits to modify specific lines while keeping the rest intact
- This ensures no accidental loss of work or context

## Testing & Debugging
- Use `quarterly_changes_analysis.txt` for output validation
- Check `logs/` directory for automated workflow debugging
- SQLite browser recommended for `ranking_history.db` inspection
- Environment variable `ENVIRONMENT=local` forces local data paths
